"""Characterization tests for LearningEngine confidence logic (Welle 4a).

Covers:
  _rebuild_confidence       – 60% base / 30% TRV / 10% window composition
  get_confidence_breakdown  – per-track percentages (note: no diversity here)
  get_cold_start_phase      – phase-label thresholds

`_rebuild_confidence` and `get_confidence_breakdown` read dt_util.now(), so it
is monkeypatched to a fixed datetime and observations are timestamped fresh.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from custom_components.thermosmart import learning_engine as le
from tests.helpers import make_learning_engine


FIXED_NOW = datetime(2025, 6, 16, 12, 0, 0)


def set_now(monkeypatch, when=FIXED_NOW):
    monkeypatch.setattr(le.dt_util, "now", lambda: when)


def heat_obs(*, heat_rate=0.05, wind=None, solar=None, indoor_hum=None,
             days_ago=0, now=FIXED_NOW) -> dict:
    obs = {
        "ts": (now - timedelta(days=days_ago)).isoformat(),
        "heat_rate": heat_rate,
    }
    if wind is not None:
        obs["wind_speed"] = wind
    if solar is not None:
        obs["solar_radiation"] = solar
    if indoor_hum is not None:
        obs["indoor_humidity"] = indoor_hum
    return obs


def plain_obs(*, days_ago=0, now=FIXED_NOW) -> dict:
    """Observation WITHOUT heat_rate → does not count toward base confidence."""
    return {"ts": (now - timedelta(days=days_ago)).isoformat(), "indoor_temp": 20.0}


# ── _rebuild_confidence: base track (60%) ────────────────────────────────────

class TestRebuildConfidenceBase:
    def test_only_heat_rate_obs_count(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        # 50 plain obs (no heat_rate) → base confidence stays 0
        e._observations["z"] = [plain_obs() for _ in range(50)]
        e._rebuild_confidence("z")
        assert e.get_confidence("z") == 0.0

    def test_25_heating_obs_gives_half_base(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        # 25 fresh heating obs, no diversity → base = 25/50 = 0.5 → total = 0.3
        e._observations["z"] = [heat_obs() for _ in range(25)]
        e._rebuild_confidence("z")
        assert e.get_confidence("z") == pytest.approx(0.3, abs=0.01)

    def test_50_heating_obs_saturates_base(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        e._observations["z"] = [heat_obs() for _ in range(50)]
        e._rebuild_confidence("z")
        # base saturates at 1.0 → total = 0.6
        assert e.get_confidence("z") == pytest.approx(0.6, abs=0.01)

    def test_diversity_factor_raises_base(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        # 25 obs with wind+solar+humidity → diversity 1.15 → base 0.575
        e._observations["z"] = [
            heat_obs(wind=3.0, solar=200.0, indoor_hum=45.0) for _ in range(25)
        ]
        e._rebuild_confidence("z")
        # total = 0.575 * 0.6 = 0.345
        assert e.get_confidence("z") == pytest.approx(0.345, abs=0.01)


# ── _rebuild_confidence: all tracks ──────────────────────────────────────────

class TestRebuildConfidenceTracks:
    def test_trv_track_contributes_thirty_percent(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        e._trv_observations["z"] = [{"ts": FIXED_NOW.isoformat()} for _ in range(30)]
        e._rebuild_confidence("z")
        # only TRV saturated → 1.0 * 0.3
        assert e.get_confidence("z") == pytest.approx(0.3, abs=0.01)

    def test_window_track_contributes_ten_percent(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        e._window_cooling_obs["z"] = [{"ts": FIXED_NOW.isoformat()} for _ in range(5)]
        e._rebuild_confidence("z")
        assert e.get_confidence("z") == pytest.approx(0.1, abs=0.01)

    def test_all_tracks_saturated_gives_one(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        e._observations["z"] = [heat_obs() for _ in range(50)]
        e._trv_observations["z"] = [{"ts": FIXED_NOW.isoformat()} for _ in range(30)]
        e._window_cooling_obs["z"] = [{"ts": FIXED_NOW.isoformat()} for _ in range(5)]
        e._rebuild_confidence("z")
        assert e.get_confidence("z") == pytest.approx(1.0, abs=0.01)

    def test_rebuild_all_zones_when_no_zone_given(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        e._observations["a"] = [heat_obs() for _ in range(50)]
        e._observations["b"] = [heat_obs() for _ in range(50)]
        e._rebuild_confidence()  # no zone arg → all zones
        assert e.get_confidence("a") == pytest.approx(0.6, abs=0.01)
        assert e.get_confidence("b") == pytest.approx(0.6, abs=0.01)


# ── get_confidence_breakdown ─────────────────────────────────────────────────

class TestConfidenceBreakdown:
    def test_breakdown_keys_present(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        e._observations["z"] = [heat_obs() for _ in range(25)]
        e._rebuild_confidence("z")
        b = e.get_confidence_breakdown("z")
        for key in (
            "total_%", "room_patterns_%", "trv_efficiency_%", "window_cooling_%",
            "total_observations", "heating_observations", "trv_observations",
            "window_events", "forecast_confidence_%",
        ):
            assert key in b

    def test_room_patterns_has_no_diversity_factor(self, monkeypatch):
        # QUIRK: breakdown.room_patterns_% omits the diversity multiplier that
        # _rebuild_confidence applies → total_% and room_patterns_% can diverge.
        set_now(monkeypatch)
        e = make_learning_engine()
        e._observations["z"] = [
            heat_obs(wind=3.0, solar=200.0, indoor_hum=45.0) for _ in range(25)
        ]
        e._rebuild_confidence("z")
        b = e.get_confidence_breakdown("z")
        # room_patterns_% = 25/50 * 100 = 50.0 (no diversity)
        assert b["room_patterns_%"] == pytest.approx(50.0, abs=0.1)
        # total_% reflects diversity-boosted base → 0.345 * 100 = 34.5
        assert b["total_%"] == pytest.approx(34.5, abs=0.2)

    def test_total_observations_counts_all_obs(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        e._observations["z"] = [heat_obs() for _ in range(10)] + [plain_obs() for _ in range(15)]
        e._rebuild_confidence("z")
        b = e.get_confidence_breakdown("z")
        assert b["total_observations"] == 25
        assert b["heating_observations"] == 10

    def test_forecast_confidence_defaults_to_full(self, monkeypatch):
        set_now(monkeypatch)
        e = make_learning_engine()
        b = e.get_confidence_breakdown("z")
        # default forecast bias = FORECAST_BIAS_MAX (1.0) → 100.0
        assert b["forecast_confidence_%"] == pytest.approx(100.0, abs=0.1)


# ── get_cold_start_phase ─────────────────────────────────────────────────────

class TestColdStartPhase:
    @pytest.mark.parametrize("n", [0, 3, 4])
    def test_initializing_below_five(self, n):
        e = make_learning_engine()
        e._observations["z"] = [plain_obs() for _ in range(n)]
        assert e.get_cold_start_phase("z") == "Initializing – physics formula active"

    @pytest.mark.parametrize("n", [5, 49])
    def test_phase_one_label(self, n):
        e = make_learning_engine()
        e._observations["z"] = [plain_obs() for _ in range(n)]
        assert e.get_cold_start_phase("z") == f"Phase 1 – First patterns ({n} obs.)"

    @pytest.mark.parametrize("n", [50, 149])
    def test_phase_two_label(self, n):
        e = make_learning_engine()
        e._observations["z"] = [plain_obs() for _ in range(n)]
        assert e.get_cold_start_phase("z") == f"Phase 2 – Learning ({n} obs., 0 TRV obs.)"

    @pytest.mark.parametrize("n", [150, 300])
    def test_phase_three_label(self, n):
        e = make_learning_engine()
        e._observations["z"] = [plain_obs() for _ in range(n)]
        assert e.get_cold_start_phase("z") == f"Phase 3 – Personalized ({n} obs., 0 TRV obs.)"

    def test_phase_two_includes_trv_count(self):
        e = make_learning_engine()
        e._observations["z"] = [plain_obs() for _ in range(60)]
        e._trv_observations["z"] = [{"ts": FIXED_NOW.isoformat()} for _ in range(7)]
        assert e.get_cold_start_phase("z") == "Phase 2 – Learning (60 obs., 7 TRV obs.)"
