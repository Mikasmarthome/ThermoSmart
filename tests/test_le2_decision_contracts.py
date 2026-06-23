"""Phase 19A: Runtime Input Builder, Baseline Adapter, Learning Prediction Set."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.decision import (
    LearningPrediction, LearningPredictionSet)
from custom_components.thermosmart.learning.decision.baseline import (
    baseline_from_recommendation, zone_input_from_recommendation)
from tests.helpers_decision import TS, rec


class TestZoneRuntimeInput:
    def test_built_from_recommendation(self):
        zin = zone_input_from_recommendation("z", TS, rec(indoor=18.5))
        assert zin.zone_id == "z" and zin.indoor_temp_c == 18.5 and zin.indoor_temp_valid

    def test_missing_indoor_marks_invalid(self):
        zin = zone_input_from_recommendation("z", TS, {"trv_setpoint": 21.0})
        assert zin.indoor_temp_c is None and zin.indoor_temp_valid is False

    def test_flags_typed(self):
        zin = zone_input_from_recommendation("z", TS, rec(is_summer=True, window_open=True))
        assert zin.summer and zin.window_open

    def test_empty_zone_rejected(self):
        with pytest.raises(ValueError):
            zone_input_from_recommendation("", TS, rec())


class TestBaselineAdapter:
    def test_boost_offset_from_rec_key(self):
        # boost_offset_c is read directly from rec["boost_offset_c"] (coordinator sets this).
        # No reconstruction from setpoint-target (that was removed: it conflated TPI with le2).
        base = baseline_from_recommendation("z", rec(target=21.0, setpoint=23.0,
                                                     boost_offset_c=2.0))
        assert base.boost_offset_c == pytest.approx(2.0)
        # Without boost_offset_c key: defaults to 0.0 (no active le2 boost)
        base_fresh = baseline_from_recommendation("z", rec(target=21.0, setpoint=23.0))
        assert base_fresh.boost_offset_c == pytest.approx(0.0)

    def test_no_negative_boost(self):
        base = baseline_from_recommendation("z", rec(target=21.0, setpoint=20.0))
        assert base.boost_offset_c == 0.0

    def test_safety_locked_summer(self):
        base = baseline_from_recommendation("z", rec(is_summer=True))
        assert base.safety_locked

    def test_active_control_flag(self):
        base = baseline_from_recommendation("z", rec(), active_control=True)
        assert base.active_control

    def test_reads_alternate_keys(self):
        base = baseline_from_recommendation("z", {"adjusted_target": 20.0, "trv_setpoint": 21.0,
                                                  "boost_offset_c": 1.0})
        assert base.target_c == pytest.approx(20.0) and base.boost_offset_c == pytest.approx(1.0)


class TestLearningPredictionSet:
    def test_get(self):
        ps = LearningPredictionSet("z", "d", predictions={
            "boost_offset": LearningPrediction("boost_offset", 1.0, "celsius_offset", 0.8, "boost")})
        assert ps.get("boost_offset").value == 1.0 and ps.get("missing") is None

    def test_nonfinite_rejected(self):
        with pytest.raises(ValueError):
            LearningPrediction("x", float("nan"), "u", 0.5, "p")

    def test_none_value_allowed(self):
        assert LearningPrediction("x", None, "u", 0.5, "p").value is None
