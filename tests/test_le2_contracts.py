"""Phase 1 tests for the LE 2.0 contracts (capability, prediction, eligibility)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import (
    CapabilitySet,
    ConfidenceContribution,
    DataQuality,
    Measurement,
    Model,
    OutlierStatus,
    Prediction,
    PredictionType,
    Regime,
    RejectionReason,
    UpdateEligibilityResult,
)


# -- Measurement --------------------------------------------------------------

class TestMeasurement:
    def test_zero_is_usable(self):
        m = Measurement(0.0, DataQuality.OK)
        assert m.is_usable is True
        assert m.value == 0.0

    def test_none_with_ok_quality_rejected(self):
        with pytest.raises(ValueError):
            Measurement(None, DataQuality.OK)

    def test_unavailable_without_value_allowed(self):
        m = Measurement(None, DataQuality.UNAVAILABLE)
        assert m.is_usable is False

    def test_stale_value_not_usable(self):
        assert Measurement(10.0, DataQuality.STALE).is_usable is False


# -- CapabilitySet (non-linear) ----------------------------------------------

class TestCapabilitySet:
    def test_trv_only_is_tier_zero(self):
        caps = CapabilitySet()
        assert caps.tier == 0
        assert not caps.has_outdoor_temp

    def test_valve_without_weather_is_valid_and_nonlinear(self):
        # Valve opening present, but no outdoor/weather data.
        caps = CapabilitySet(has_valve_opening=True)
        assert caps.has_valve_opening is True
        assert caps.has_outdoor_temp is False
        assert caps.has_weather is False
        # tier summary may report 3, but it must NOT imply weather is present.
        assert caps.tier == 3
        assert caps.has_outdoor_temp is False  # models must check the flag, not the tier

    def test_outdoor_only_does_not_imply_valve(self):
        caps = CapabilitySet(has_outdoor_temp=True)
        assert caps.tier == 2
        assert caps.has_valve_opening is False

    def test_flags_are_authoritative(self):
        caps = CapabilitySet(has_room_sensor=True, has_solar=True)
        assert caps.has_room_sensor and caps.has_solar
        assert caps.has_window_sensor is False


# -- Prediction ---------------------------------------------------------------

def _pred(**kw) -> Prediction:
    params = dict(
        prediction_type=PredictionType.HEAT_RATE,
        values={"heat_rate": 0.05},
        units={"heat_rate": "C/min"},
        confidence=0.4,
        reliability=0.5,
        model_version=1,
        parameter_version=1,
        prior_contribution=0.0,
        learned_contribution=1.0,
        fallback_used=False,
    )
    params.update(kw)
    return Prediction(**params)


class TestPrediction:
    def test_valid_prediction(self):
        p = _pred()
        assert p.control_value("heat_rate") == 0.05

    @pytest.mark.parametrize("field_name", ["confidence", "reliability"])
    def test_confidence_reliability_out_of_range(self, field_name):
        with pytest.raises(ValueError):
            _pred(**{field_name: 1.5})

    def test_contributions_must_sum_to_one(self):
        with pytest.raises(ValueError):
            _pred(prior_contribution=0.3, learned_contribution=0.3)

    def test_fallback_contributions(self):
        p = _pred(fallback_used=True, prior_contribution=1.0, learned_contribution=0.0)
        assert p.fallback_used and p.prior_contribution == 1.0

    def test_fallback_requires_positive_prior(self):
        with pytest.raises(ValueError):
            _pred(fallback_used=True, prior_contribution=0.0, learned_contribution=1.0)

    def test_non_fallback_requires_positive_learned(self):
        with pytest.raises(ValueError):
            _pred(fallback_used=False, prior_contribution=1.0, learned_contribution=0.0)

    def test_value_without_unit_rejected(self):
        with pytest.raises(ValueError):
            _pred(values={"heat_rate": 0.05, "extra": 1.0})

    def test_empty_values_rejected(self):
        with pytest.raises(ValueError):
            _pred(values={}, units={})

    def test_confidence_cap_enforced(self):
        with pytest.raises(ValueError):
            _pred(confidence=0.9, confidence_cap=0.5)

    def test_confidence_cap_respected(self):
        p = _pred(confidence=0.4, confidence_cap=0.5, cap_reasons=("fallback",))
        assert p.confidence_cap == 0.5
        assert p.cap_reasons == ("fallback",)

    def test_control_value_missing_key(self):
        with pytest.raises(KeyError):
            _pred().control_value("nope")

    def test_zero_value_is_valid(self):
        p = _pred(values={"heat_rate": 0.0}, units={"heat_rate": "C/min"})
        assert p.control_value("heat_rate") == 0.0


# -- UpdateEligibilityResult --------------------------------------------------

class TestUpdateEligibility:
    def test_accepted_sample(self):
        r = UpdateEligibilityResult(accepted=True, reliability=0.7, weight=0.8,
                                    regime=Regime.ACTIVE_HEATING)
        assert r.accepted and r.weight == 0.8

    def test_rejected_requires_reason_and_zero_weight(self):
        with pytest.raises(ValueError):
            UpdateEligibilityResult(accepted=False, reliability=0.5, weight=0.0)

    def test_rejected_with_weight_rejected(self):
        with pytest.raises(ValueError):
            UpdateEligibilityResult(accepted=False, reliability=0.5, weight=0.3,
                                    rejection_reason=RejectionReason.OUTLIER)

    def test_accepted_with_reason_rejected(self):
        with pytest.raises(ValueError):
            UpdateEligibilityResult(accepted=True, reliability=0.5,
                                    rejection_reason=RejectionReason.OUTLIER)

    def test_rejection_carries_diagnostics(self):
        r = UpdateEligibilityResult(
            accepted=False, reliability=0.2,
            rejection_reason=RejectionReason.MISSING_REQUIRED_DATA,
            required_data_missing=("indoor_temp",),
            outlier_status=OutlierStatus.NONE,
            regime=Regime.UNKNOWN,
        )
        assert r.required_data_missing == ("indoor_temp",)
        assert r.regime is Regime.UNKNOWN

    def test_reliability_range(self):
        with pytest.raises(ValueError):
            UpdateEligibilityResult(accepted=True, reliability=2.0)


class TestConfidenceContribution:
    def test_valid(self):
        c = ConfidenceContribution(value=0.3, evidence_count=5)
        assert c.value == 0.3

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            ConfidenceContribution(value=1.2)


class TestModelProtocol:
    def test_incomplete_object_is_not_a_model(self):
        class NotAModel:
            model_name = "x"

        assert not isinstance(NotAModel(), Model)
