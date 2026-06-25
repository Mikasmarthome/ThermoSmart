"""Characterization tests for LearningEngine.async_observe (Welle 4b).

Freezes the current in-memory mutation behaviour of the main observation path:
heat-rate windowing, cool-rate guard + heat-loss EMA, dedup, norm_heat_rate,
and the "save every 50 observations" trigger.

Storage is never written: engine.async_save is replaced with an AsyncMock.
A FakeClock is injected so elapsed-time windows are deterministic without
patching dt_util.now() (LearningEngine._now_local() uses the injected clock).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from tests.helpers import make_learning_engine


FIXED_NOW = datetime(2025, 6, 16, 12, 0, 0, tzinfo=timezone.utc)  # Monday, UTC


def make_engine(monkeypatch=None, now=FIXED_NOW):
    """LearningEngine with frozen clock (FakeClock) and a mocked (no-op) save."""
    e = make_learning_engine(clock=FakeClock(start=now))
    e.async_save = AsyncMock()
    return e


def rec(current_temp=20.0, effective_target=21.0, **extra) -> dict:
    r = {"current_temp": current_temp, "effective_target": effective_target}
    r.update(extra)
    return r


# ── Early-exit guards ────────────────────────────────────────────────────────

class TestObserveGuards:
    async def test_disabled_zone_records_nothing(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.set_zone_enabled("z", False)
        await e.async_observe("z", rec(), {"temperature": 10.0})
        assert e._observations["z"] == []
        e.async_save.assert_not_called()

    async def test_missing_current_temp_records_nothing(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(current_temp=None), {"temperature": 10.0})
        assert e._observations["z"] == []

    async def test_missing_target_records_nothing(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", {"current_temp": 20.0}, {"temperature": 10.0})
        assert e._observations["z"] == []

    async def test_adjusted_target_fallback_used(self, monkeypatch):
        # effective_target missing → falls back to adjusted_target
        e = make_engine(monkeypatch)
        await e.async_observe(
            "z", {"current_temp": 20.0, "adjusted_target": 21.0}, {"temperature": 10.0}
        )
        assert len(e._observations["z"]) == 1
        assert e._observations["z"][0]["target"] == 21.0


# ── Basic observation shape ──────────────────────────────────────────────────

class TestObservationShape:
    async def test_core_fields_recorded(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(current_temp=20.0, effective_target=21.0),
                              {"temperature": 8.0})
        obs = e._observations["z"][0]
        assert obs["target"] == 21.0
        assert obs["indoor_temp"] == 20.0
        assert obs["delta"] == pytest.approx(1.0)
        assert obs["hour"] == 12
        assert obs["weekday"] == 0
        assert obs["outdoor_temp"] == 8.0

    async def test_weather_fields_mapped(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe(
            "z", rec(),
            {"temperature": 8.0, "humidity": 70.0, "wind_speed": 4.0,
             "solar_radiation": 250.0},
        )
        obs = e._observations["z"][0]
        assert obs["outdoor_temp"] == 8.0
        assert obs["outdoor_humidity"] == 70.0
        assert obs["wind_speed"] == 4.0
        assert obs["solar_radiation"] == 250.0

    async def test_indoor_humidity_recorded_when_given(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(), {"temperature": 8.0}, indoor_humidity=45.0)
        assert e._observations["z"][0]["indoor_humidity"] == 45.0

    async def test_flags_recorded(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(), {"temperature": 8.0},
                              is_active_control=True, window_open=True,
                              control_reason="preheat", vacation=True)
        obs = e._observations["z"][0]
        assert obs["active_control"] is True
        assert obs["window_open"] is True
        assert obs["control_reason"] == "preheat"
        assert obs["vacation"] is True

    async def test_forecast_high_recorded_when_present(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(forecast_high=15.0), {"temperature": 8.0})
        assert e._observations["z"][0]["forecast_high"] == 15.0

    async def test_missing_weather_temp_omits_outdoor_temp(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(), {})  # no weather data at all
        obs = e._observations["z"][0]
        assert "outdoor_temp" not in obs


# ── Heat-rate window (3–15 min) ──────────────────────────────────────────────

class TestHeatRateWindow:
    async def test_heat_rate_computed_within_window(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._last_temp["z"] = (FIXED_NOW - timedelta(minutes=5), 20.0)
        await e.async_observe("z", rec(current_temp=20.5, effective_target=21.0),
                              {"temperature": 10.0})
        obs = e._observations["z"][0]
        # (20.5-20.0)/5 = 0.1
        assert obs["heat_rate"] == pytest.approx(0.1, abs=1e-5)

    async def test_norm_heat_rate_derived(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._last_temp["z"] = (FIXED_NOW - timedelta(minutes=5), 20.0)
        await e.async_observe("z", rec(current_temp=20.5, effective_target=21.0),
                              {"temperature": 10.0})
        obs = e._observations["z"][0]
        # norm = heat_rate / max(target-outdoor,1) = 0.1 / 11
        assert obs["norm_heat_rate"] == pytest.approx(0.1 / 11.0, abs=1e-5)

    async def test_no_heat_rate_below_min_window(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._last_temp["z"] = (FIXED_NOW - timedelta(minutes=2), 20.0)
        await e.async_observe("z", rec(current_temp=20.5), {"temperature": 10.0})
        assert "heat_rate" not in e._observations["z"][0]

    async def test_no_heat_rate_above_max_window(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._last_temp["z"] = (FIXED_NOW - timedelta(minutes=20), 20.0)
        await e.async_observe("z", rec(current_temp=20.5), {"temperature": 10.0})
        assert "heat_rate" not in e._observations["z"][0]

    async def test_no_heat_rate_when_not_warming(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._last_temp["z"] = (FIXED_NOW - timedelta(minutes=5), 20.0)
        await e.async_observe("z", rec(current_temp=20.0), {"temperature": 10.0})
        assert "heat_rate" not in e._observations["z"][0]

    async def test_first_observation_has_no_heat_rate_but_sets_last_temp(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(current_temp=20.0), {"temperature": 10.0})
        assert "heat_rate" not in e._observations["z"][0]
        assert e._last_temp["z"][1] == 20.0


# ── Cool-rate guard + heat-loss EMA ──────────────────────────────────────────

class TestCoolRateAndEMA:
    async def test_cool_rate_recorded_and_ema_not_mutated(self, monkeypatch):
        # Phase 19D: cool_rate is still stored in observations for LE2 training,
        # but _heat_loss_ema must NOT be mutated (LE2 HeatLossModel is authoritative).
        e = make_engine(monkeypatch)
        # cooling: current < target-0.5; prior idle reading 20 min ago, warmer
        e._last_idle_temp["z"] = (FIXED_NOW - timedelta(minutes=20), 20.5)
        await e.async_observe("z", rec(current_temp=20.0, effective_target=21.0),
                              {"temperature": 10.0})
        obs = e._observations["z"][0]
        # (20.5-20.0)/20 = 0.025 — still recorded for LE2 PassiveCoolingEpisodes
        assert obs["cool_rate"] == pytest.approx(0.025, abs=1e-5)
        # EMA must remain empty — mutations are silenced in Phase 19D
        assert "z" not in e._heat_loss_ema

    async def test_cool_rate_guard_rejects_implausible_drop(self, monkeypatch):
        e = make_engine(monkeypatch)
        # raw_cool = (30-20)/10 = 1.0 > 0.5 → rejected
        e._last_idle_temp["z"] = (FIXED_NOW - timedelta(minutes=10), 30.0)
        await e.async_observe("z", rec(current_temp=20.0, effective_target=25.0),
                              {"temperature": 10.0})
        assert "cool_rate" not in e._observations["z"][0]
        assert "z" not in e._heat_loss_ema

    async def test_no_cool_rate_below_min_elapsed(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._last_idle_temp["z"] = (FIXED_NOW - timedelta(minutes=5), 20.5)
        await e.async_observe("z", rec(current_temp=20.0, effective_target=21.0),
                              {"temperature": 10.0})
        assert "cool_rate" not in e._observations["z"][0]

    async def test_no_cool_rate_above_max_elapsed(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._last_idle_temp["z"] = (FIXED_NOW - timedelta(minutes=90), 20.5)
        await e.async_observe("z", rec(current_temp=20.0, effective_target=21.0),
                              {"temperature": 10.0})
        assert "cool_rate" not in e._observations["z"][0]

    async def test_ema_frozen_after_mutation_silenced(self, monkeypatch):
        # Phase 19D: existing EMA data may be loaded from store (read-only compat),
        # but async_observe must NOT update it. The dict must stay frozen.
        e = make_engine(monkeypatch)
        e._heat_loss_ema["z"] = 0.10  # existing value (e.g. loaded from store)
        e._last_idle_temp["z"] = (FIXED_NOW - timedelta(minutes=20), 20.5)
        await e.async_observe("z", rec(current_temp=20.0, effective_target=21.0),
                              {"temperature": 10.0})
        # EMA must remain 0.10 — the mutation (0.15*cool_rate + 0.85*prev) is silenced
        assert e._heat_loss_ema["z"] == pytest.approx(0.10, abs=1e-5)

    async def test_idle_temp_cleared_when_not_cooling(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._last_idle_temp["z"] = (FIXED_NOW - timedelta(minutes=20), 20.5)
        # current >= target-0.5 → not cooling → idle tracker popped
        await e.async_observe("z", rec(current_temp=21.0, effective_target=21.0),
                              {"temperature": 10.0})
        assert "z" not in e._last_idle_temp

    async def test_idle_temp_set_when_cooling(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(current_temp=20.0, effective_target=21.0),
                              {"temperature": 10.0})
        assert e._last_idle_temp["z"][1] == 20.0


# ── Deduplication ────────────────────────────────────────────────────────────

class TestDeduplication:
    async def test_duplicate_within_window_skipped(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(current_temp=20.0), {"temperature": 10.0})
        await e.async_observe("z", rec(current_temp=20.0), {"temperature": 10.0})
        assert len(e._observations["z"]) == 1

    async def test_different_delta_not_duplicate(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(current_temp=20.0, effective_target=21.0),
                              {"temperature": 10.0})
        # delta changes by 1.5 (> 0.8 tol) → not a duplicate
        await e.async_observe("z", rec(current_temp=18.5, effective_target=21.0),
                              {"temperature": 10.0})
        assert len(e._observations["z"]) == 2

    async def test_old_observation_not_considered_duplicate(self, monkeypatch):
        e = make_engine(monkeypatch)
        # seed an obs older than the 30-min dedup window
        old = {
            "ts": (FIXED_NOW - timedelta(minutes=45)).isoformat(),
            "hour": 12, "weekday": 0, "outdoor_temp": 10.0, "delta": 1.0,
        }
        e._observations["z"].append(old)
        await e.async_observe("z", rec(current_temp=20.0, effective_target=21.0),
                              {"temperature": 10.0})
        assert len(e._observations["z"]) == 2


# ── Save trigger (every 50) ──────────────────────────────────────────────────

class TestSaveTrigger:
    async def test_no_save_on_single_observation(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_observe("z", rec(), {"temperature": 10.0})
        e.async_save.assert_not_called()

    async def test_save_triggered_at_fiftieth(self, monkeypatch):
        e = make_engine(monkeypatch)
        # pre-fill 49 obs with old timestamps (outside dedup window → new appends)
        e._observations["z"] = [
            {"ts": (FIXED_NOW - timedelta(hours=2)).isoformat(),
             "hour": 10, "weekday": 0, "outdoor_temp": 5.0, "delta": 0.5}
            for _ in range(49)
        ]
        await e.async_observe("z", rec(current_temp=20.0, effective_target=21.0),
                              {"temperature": 10.0})
        assert len(e._observations["z"]) == 50
        e.async_save.assert_awaited_once()
