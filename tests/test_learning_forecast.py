"""Characterization tests for LearningEngine forecast-bias logic (Welle 4b).

Covers:
  record_forecast_decision   – recording + 2h dedup + disabled-zone guard
  evaluate_forecast_decisions – 5h lookback, bias up/down/neutral,
                                min/max clamping, 7d cleanup of evaluated entries

Clock frozen via dt_util.now() monkeypatch. _forecast_decisions is in-memory
(not persisted); save (fired on change) is mocked to a no-op.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart import learning_engine as le
from custom_components.thermosmart.const import (
    FORECAST_BIAS_MIN,
    FORECAST_BIAS_MAX,
    FORECAST_BIAS_LEARNING_RATE,
)
from tests.helpers import make_learning_engine


FIXED_NOW = datetime(2025, 6, 16, 12, 0, 0)


def make_engine(monkeypatch, now=FIXED_NOW):
    monkeypatch.setattr(le.dt_util, "now", lambda: now)
    e = make_learning_engine()
    e.async_save = MagicMock()
    e._hass.async_create_task = lambda *a, **k: None
    return e


def seed_decision(e, *, decision_target, minutes_ago, evaluated=False, now=FIXED_NOW):
    e._forecast_decisions["z"].append({
        "ts": (now - timedelta(minutes=minutes_ago)).isoformat(),
        "decision_target": decision_target,
        "raw_target": decision_target,
        "forecast_high": 15.0,
        "current_temp_at_decision": 19.0,
        "suppression": 0.5,
        "outdoor_temp": 8.0,
        "evaluated": evaluated,
    })


# ── record_forecast_decision ─────────────────────────────────────────────────

class TestRecordDecision:
    def test_decision_recorded(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.record_forecast_decision("z", decision_target=20.0, raw_target=21.0,
                                   forecast_high=18.0, current_temp=19.0,
                                   suppression=0.5, outdoor_temp=8.0)
        decisions = e._forecast_decisions["z"]
        assert len(decisions) == 1
        assert decisions[0]["evaluated"] is False
        assert decisions[0]["decision_target"] == 20.0

    def test_disabled_zone_records_nothing(self, monkeypatch):
        e = make_engine(monkeypatch)
        e.set_zone_enabled("z", False)
        e.record_forecast_decision("z", 20.0, 21.0, 18.0, 19.0, 0.5, 8.0)
        assert e._forecast_decisions["z"] == []

    def test_second_decision_within_2h_deduped(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=20.0, minutes_ago=30)  # 0.5h ago
        e.record_forecast_decision("z", 20.5, 21.0, 18.0, 19.0, 0.5, 8.0)
        assert len(e._forecast_decisions["z"]) == 1  # deduped

    def test_second_decision_after_2h_recorded(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=20.0, minutes_ago=150)  # 2.5h ago
        e.record_forecast_decision("z", 20.5, 21.0, 18.0, 19.0, 0.5, 8.0)
        assert len(e._forecast_decisions["z"]) == 2


# ── evaluate_forecast_decisions: timing ──────────────────────────────────────

class TestEvaluateTiming:
    def test_decision_younger_than_lookback_not_evaluated(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=21.0, minutes_ago=120)  # 2h < 5h
        e._forecast_bias["z"] = 1.0
        e.evaluate_forecast_decisions("z", current_temp=19.0, outdoor_temp=8.0)
        assert e._forecast_decisions["z"][0]["evaluated"] is False
        assert e._forecast_bias["z"] == 1.0  # untouched

    def test_none_current_temp_skips_evaluation(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=21.0, minutes_ago=360)
        e._forecast_bias["z"] = 1.0
        e.evaluate_forecast_decisions("z", current_temp=None, outdoor_temp=8.0)
        assert e._forecast_decisions["z"][0]["evaluated"] is False


# ── evaluate_forecast_decisions: bias direction ──────────────────────────────

class TestBiasDirection:
    def test_shortfall_reduces_bias(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=21.0, minutes_ago=360)  # 6h ago
        e._forecast_bias["z"] = 1.0
        # room cold: current 20 → shortfall 1.0 → reduction 0.06*min(0.5,1)=0.03
        e.evaluate_forecast_decisions("z", current_temp=20.0, outdoor_temp=8.0)
        assert e._forecast_bias["z"] == pytest.approx(0.97, abs=1e-6)
        assert e._forecast_decisions["z"][0]["evaluated"] is True

    def test_large_shortfall_reduction_capped(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=24.0, minutes_ago=360)
        e._forecast_bias["z"] = 1.0
        # shortfall 4 → min(4/2,1)=1 → reduction = full learning rate 0.06
        e.evaluate_forecast_decisions("z", current_temp=20.0, outdoor_temp=8.0)
        assert e._forecast_bias["z"] == pytest.approx(1.0 - FORECAST_BIAS_LEARNING_RATE, abs=1e-6)

    def test_overshoot_increases_bias(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=21.0, minutes_ago=360)
        e._forecast_bias["z"] = 0.90
        # current 23 → shortfall -2 < -1.5 → +0.06*0.3 = 0.018
        e.evaluate_forecast_decisions("z", current_temp=23.0, outdoor_temp=8.0)
        assert e._forecast_bias["z"] == pytest.approx(0.918, abs=1e-6)

    def test_neutral_increases_bias_slightly(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=21.0, minutes_ago=360)
        e._forecast_bias["z"] = 0.90
        # current 21 → shortfall 0 → +0.06*0.15 = 0.009
        e.evaluate_forecast_decisions("z", current_temp=21.0, outdoor_temp=8.0)
        assert e._forecast_bias["z"] == pytest.approx(0.909, abs=1e-6)


# ── evaluate_forecast_decisions: clamping ────────────────────────────────────

class TestBiasClamping:
    def test_bias_clamped_to_min(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=24.0, minutes_ago=360)
        e._forecast_bias["z"] = FORECAST_BIAS_MIN + 0.02  # 0.32
        # large shortfall reduction 0.06 would drop below MIN → clamped
        e.evaluate_forecast_decisions("z", current_temp=20.0, outdoor_temp=8.0)
        assert e._forecast_bias["z"] == FORECAST_BIAS_MIN

    def test_bias_clamped_to_max_on_neutral(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=21.0, minutes_ago=360)
        e._forecast_bias["z"] = FORECAST_BIAS_MAX
        e.evaluate_forecast_decisions("z", current_temp=21.0, outdoor_temp=8.0)
        assert e._forecast_bias["z"] == FORECAST_BIAS_MAX

    def test_default_bias_is_max(self, monkeypatch):
        e = make_engine(monkeypatch)
        assert e.get_forecast_bias("z") == FORECAST_BIAS_MAX


# ── evaluate_forecast_decisions: cleanup ─────────────────────────────────────

class TestDecisionCleanup:
    def test_evaluated_older_than_7d_removed(self, monkeypatch):
        e = make_engine(monkeypatch)
        # already-evaluated decision, 8 days old → cleaned up
        seed_decision(e, decision_target=21.0, minutes_ago=8 * 1440, evaluated=True)
        # plus a fresh unevaluated decision to keep the list non-trivial
        seed_decision(e, decision_target=21.0, minutes_ago=360, evaluated=False)
        e._forecast_bias["z"] = 0.9
        e.evaluate_forecast_decisions("z", current_temp=21.0, outdoor_temp=8.0)
        remaining = e._forecast_decisions["z"]
        # the 8-day-old evaluated one is gone; the recent one remains (now evaluated)
        assert len(remaining) == 1
        assert remaining[0]["decision_target"] == 21.0

    def test_recent_evaluated_decision_retained(self, monkeypatch):
        e = make_engine(monkeypatch)
        seed_decision(e, decision_target=21.0, minutes_ago=360)
        e._forecast_bias["z"] = 0.9
        e.evaluate_forecast_decisions("z", current_temp=21.0, outdoor_temp=8.0)
        # evaluated but within 7d → retained
        assert len(e._forecast_decisions["z"]) == 1
        assert e._forecast_decisions["z"][0]["evaluated"] is True
