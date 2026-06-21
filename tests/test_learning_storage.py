"""Characterization tests for LearningEngine persistence & lifecycle (Welle 4c / Tier A).

Covers, with a MOCKED Store (no real disk I/O, Windows-runnable):
  async_save                 – payload contains exactly the 7 persisted top-level keys
  async_load                 – populates all dicts + rebuilds confidence
  save → load round-trip
  _prune_old_observations    – 3-year cutoff + 5000 per-zone/per-track cap
  prune_orphaned_zones       – empty-set safety + removal across all stores/scalars
  get_export_data            – completeness of export keys
  _schedule_debounced_save   – no-op while a save is already pending

These freeze current behaviour. Note: _forecast_decisions is NOT persisted.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.thermosmart import learning_engine as le
from tests.helpers import make_learning_engine


FIXED_NOW = datetime(2025, 6, 16, 12, 0, 0)

PERSISTED_KEYS = {
    "observations",
    "trv_observations",
    "window_cooling_obs",
    "boost_factors",
    "forecast_bias",
    "heat_loss_ema",
    "outcome_log",
}


def make_engine(monkeypatch, now=FIXED_NOW):
    monkeypatch.setattr(le.dt_util, "now", lambda: now)
    e = make_learning_engine()
    e._store.async_save = AsyncMock()
    e._store.async_load = AsyncMock(return_value=None)
    e._hass.async_create_task = lambda *a, **k: None
    return e


def fresh_obs(**extra):
    o = {"ts": FIXED_NOW.isoformat()}
    o.update(extra)
    return o


# ── async_save: payload shape ────────────────────────────────────────────────

class TestAsyncSavePayload:
    async def test_payload_has_exactly_seven_keys(self, monkeypatch):
        e = make_engine(monkeypatch)
        await e.async_save()
        payload = e._store.async_save.call_args.args[0]
        assert set(payload.keys()) == PERSISTED_KEYS

    async def test_forecast_decisions_not_persisted(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._forecast_decisions["z"].append({"ts": FIXED_NOW.isoformat()})
        await e.async_save()
        payload = e._store.async_save.call_args.args[0]
        assert "forecast_decisions" not in payload

    async def test_payload_values_round_tripped(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._observations["z"] = [fresh_obs(target=21.0)]
        e._boost_factors["z"] = 0.9
        e._forecast_bias["z"] = 0.7
        e._heat_loss_ema["z"] = 0.015
        e._outcome_log["z"] = [{"outcome_score": 0.8, "controller": "ts"}]
        await e.async_save()
        payload = e._store.async_save.call_args.args[0]
        assert payload["observations"]["z"][0]["target"] == 21.0
        assert payload["boost_factors"]["z"] == 0.9
        assert payload["forecast_bias"]["z"] == 0.7
        assert payload["heat_loss_ema"]["z"] == 0.015
        assert payload["outcome_log"]["z"][0]["outcome_score"] == 0.8

    async def test_pending_debounce_is_cancelled_on_save(self, monkeypatch):
        e = make_engine(monkeypatch)
        cancel = MagicMock()
        e._debounce_save_cancel = cancel
        await e.async_save()
        cancel.assert_called_once()
        assert e._debounce_save_cancel is None


# ── async_load ───────────────────────────────────────────────────────────────

class TestAsyncLoad:
    async def test_load_populates_all_collections(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._store.async_load = AsyncMock(return_value={
            "observations": {"z": [fresh_obs(heat_rate=0.05)]},
            "trv_observations": {"z": [{"ts": FIXED_NOW.isoformat()}]},
            "window_cooling_obs": {"z": [{"ts": FIXED_NOW.isoformat()}]},
            "boost_factors": {"z": 0.88},
            "forecast_bias": {"z": 0.6},
            "heat_loss_ema": {"z": 0.012},
            "outcome_log": {"z": [{"outcome_score": 0.9, "controller": "obs"}]},
        })
        await e.async_load()
        assert e._observations["z"][0]["heat_rate"] == 0.05
        assert e._trv_observations["z"]
        assert e._window_cooling_obs["z"]
        assert e._boost_factors["z"] == 0.88
        assert e._forecast_bias["z"] == 0.6
        assert e._heat_loss_ema["z"] == 0.012
        assert e._outcome_log["z"][0]["outcome_score"] == 0.9

    async def test_load_rebuilds_confidence(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._store.async_load = AsyncMock(return_value={
            "observations": {"z": [fresh_obs(heat_rate=0.05) for _ in range(50)]},
        })
        await e.async_load()
        # 50 fresh heating obs → base saturates → confidence ~0.6
        assert e.get_confidence("z") == pytest.approx(0.6, abs=0.02)

    async def test_load_none_does_not_crash(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._store.async_load = AsyncMock(return_value=None)
        await e.async_load()
        assert e._observations == {}

    async def test_load_empty_dict_does_not_crash(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._store.async_load = AsyncMock(return_value={})
        await e.async_load()
        assert e.get_confidence("z") == 0.0


# ── Round-trip ───────────────────────────────────────────────────────────────

class TestRoundTrip:
    async def test_save_then_load_preserves_data(self, monkeypatch):
        e1 = make_engine(monkeypatch)
        e1._observations["z"] = [fresh_obs(target=21.0, heat_rate=0.05)]
        e1._trv_observations["z"] = [{"ts": FIXED_NOW.isoformat(), "efficiency": 0.04}]
        e1._window_cooling_obs["z"] = [{"ts": FIXED_NOW.isoformat(), "cooling_rate_per_min": 0.2}]
        e1._boost_factors["z"] = 0.91
        e1._forecast_bias["z"] = 0.55
        e1._heat_loss_ema["z"] = 0.013
        e1._outcome_log["z"] = [{"outcome_score": 0.7, "controller": "ts"}]
        await e1.async_save()
        payload = e1._store.async_save.call_args.args[0]

        e2 = make_engine(monkeypatch)
        e2._store.async_load = AsyncMock(return_value=payload)
        await e2.async_load()

        assert e2._observations["z"] == e1._observations["z"]
        assert e2._trv_observations["z"] == e1._trv_observations["z"]
        assert e2._window_cooling_obs["z"] == e1._window_cooling_obs["z"]
        assert e2._boost_factors == e1._boost_factors
        assert e2._forecast_bias == e1._forecast_bias
        assert e2._heat_loss_ema == e1._heat_loss_ema
        assert e2._outcome_log["z"] == e1._outcome_log["z"]


# ── _prune_old_observations ──────────────────────────────────────────────────

class TestPruneOldObservations:
    def test_old_observations_removed(self, monkeypatch):
        monkeypatch.setattr(le.dt_util, "now", lambda: FIXED_NOW)
        e = make_learning_engine()
        old = {"ts": (FIXED_NOW - timedelta(days=1200)).isoformat()}  # > 1095
        recent = {"ts": (FIXED_NOW - timedelta(days=10)).isoformat()}
        e._observations["z"] = [old, recent]
        e._prune_old_observations()
        assert e._observations["z"] == [recent]

    def test_cutoff_boundary_keeps_recent(self, monkeypatch):
        monkeypatch.setattr(le.dt_util, "now", lambda: FIXED_NOW)
        e = make_learning_engine()
        just_inside = {"ts": (FIXED_NOW - timedelta(days=1000)).isoformat()}
        e._observations["z"] = [just_inside]
        e._prune_old_observations()
        assert e._observations["z"] == [just_inside]

    def test_prune_applies_to_all_three_obs_stores(self, monkeypatch):
        monkeypatch.setattr(le.dt_util, "now", lambda: FIXED_NOW)
        e = make_learning_engine()
        old = {"ts": (FIXED_NOW - timedelta(days=1200)).isoformat()}
        e._observations["z"] = [dict(old)]
        e._trv_observations["z"] = [dict(old)]
        e._window_cooling_obs["z"] = [dict(old)]
        e._prune_old_observations()
        assert e._observations["z"] == []
        assert e._trv_observations["z"] == []
        assert e._window_cooling_obs["z"] == []

    def test_per_zone_cap_trims_to_last_n(self, monkeypatch):
        monkeypatch.setattr(le.dt_util, "now", lambda: FIXED_NOW)
        e = make_learning_engine()
        ts = FIXED_NOW.isoformat()
        e._observations["z"] = [{"ts": ts, "i": i} for i in range(5003)]
        e._prune_old_observations(max_per_zone=5000)
        assert len(e._observations["z"]) == 5000
        # keeps the LAST 5000 → first retained index is 3
        assert e._observations["z"][0]["i"] == 3
        assert e._observations["z"][-1]["i"] == 5002

    def test_outcome_log_not_age_pruned(self, monkeypatch):
        # QUIRK: _prune_old_observations only touches the 3 obs stores;
        # outcome_log is capped elsewhere (to 200) but never age-pruned here.
        monkeypatch.setattr(le.dt_util, "now", lambda: FIXED_NOW)
        e = make_learning_engine()
        e._outcome_log["z"] = [{"ts": (FIXED_NOW - timedelta(days=2000)).isoformat()}]
        e._prune_old_observations()
        assert len(e._outcome_log["z"]) == 1


# ── prune_orphaned_zones ─────────────────────────────────────────────────────

class TestPruneOrphanedZones:
    def _populate(self, e, zone):
        e._observations[zone] = [{"ts": FIXED_NOW.isoformat()}]
        e._trv_observations[zone] = [{"ts": FIXED_NOW.isoformat()}]
        e._window_cooling_obs[zone] = [{"ts": FIXED_NOW.isoformat()}]
        e._outcome_log[zone] = [{"outcome_score": 0.5}]
        e._forecast_decisions[zone] = [{"ts": FIXED_NOW.isoformat()}]
        e._boost_factors[zone] = 0.9
        e._forecast_bias[zone] = 0.6
        e._heat_loss_ema[zone] = 0.01

    def test_empty_active_set_is_a_noop_safety_guard(self, monkeypatch):
        e = make_engine(monkeypatch)
        self._populate(e, "z")
        e.prune_orphaned_zones(set())
        # nothing removed despite empty active set
        assert e._observations["z"]
        assert e._boost_factors["z"] == 0.9

    def test_inactive_zone_removed_from_all_stores(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.async_save = MagicMock()  # removal fires save via async_create_task
        self._populate(e, "stale")
        self._populate(e, "active")
        e.prune_orphaned_zones({"active"})
        for store in (e._observations, e._trv_observations, e._window_cooling_obs,
                      e._outcome_log, e._forecast_decisions,
                      e._boost_factors, e._forecast_bias, e._heat_loss_ema):
            assert "stale" not in store
            assert "active" in store

    def test_save_triggered_when_zone_removed(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.async_save = MagicMock()  # avoid an un-awaited coroutine
        create_task = MagicMock()
        e._hass.async_create_task = create_task
        self._populate(e, "stale")
        self._populate(e, "active")
        e.prune_orphaned_zones({"active"})
        create_task.assert_called_once()

    def test_no_save_when_nothing_removed(self, monkeypatch):
        e = make_engine(monkeypatch)
        create_task = MagicMock()
        e._hass.async_create_task = create_task
        self._populate(e, "active")
        e.prune_orphaned_zones({"active"})
        create_task.assert_not_called()


# ── get_export_data ──────────────────────────────────────────────────────────

class TestExportData:
    def test_export_has_all_keys(self, monkeypatch):
        e = make_engine(monkeypatch)
        data = e.get_export_data("z")
        expected = {
            "observation_count", "trv_observation_count", "window_cooling_obs_count",
            "confidence", "boost_factor", "forecast_bias", "heat_loss_ema",
            "observations", "trv_observations", "window_cooling_obs", "outcome_log",
        }
        assert set(data.keys()) == expected

    def test_export_counts_match(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._observations["z"] = [fresh_obs(), fresh_obs()]
        e._trv_observations["z"] = [{"ts": FIXED_NOW.isoformat()}]
        e._window_cooling_obs["z"] = [{"ts": FIXED_NOW.isoformat()}] * 3
        data = e.get_export_data("z")
        assert data["observation_count"] == 2
        assert data["trv_observation_count"] == 1
        assert data["window_cooling_obs_count"] == 3

    def test_export_scalars_passed_through(self, monkeypatch):
        e = make_engine(monkeypatch)
        e._boost_factors["z"] = 0.85
        e._forecast_bias["z"] = 0.5
        e._heat_loss_ema["z"] = 0.02
        data = e.get_export_data("z")
        assert data["boost_factor"] == 0.85
        assert data["forecast_bias"] == 0.5
        assert data["heat_loss_ema"] == 0.02

    def test_export_unknown_zone_returns_empty_structures(self, monkeypatch):
        e = make_engine(monkeypatch)
        data = e.get_export_data("nope")
        assert data["observation_count"] == 0
        assert data["observations"] == []
        assert data["boost_factor"] is None


# ── _schedule_debounced_save ─────────────────────────────────────────────────

class TestScheduleDebouncedSave:
    def test_noop_when_save_already_pending(self, monkeypatch):
        e = make_engine(monkeypatch)
        called = MagicMock(return_value="NEW_CANCEL")
        monkeypatch.setattr(le, "async_call_later", called)
        sentinel = MagicMock()
        e._debounce_save_cancel = sentinel
        e._schedule_debounced_save()
        called.assert_not_called()
        assert e._debounce_save_cancel is sentinel  # unchanged

    def test_schedules_when_not_pending(self, monkeypatch):
        e = make_engine(monkeypatch)
        called = MagicMock(return_value="NEW_CANCEL")
        monkeypatch.setattr(le, "async_call_later", called)
        e._debounce_save_cancel = None
        e._schedule_debounced_save()
        called.assert_called_once()
        assert e._debounce_save_cancel == "NEW_CANCEL"
