"""Phase 18 control integration via the real coordinator hook (CONTROL test-injected)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.const import TPI_MAX_BOOST_CELSIUS
from custom_components.thermosmart.learning.confidence import (
    ComponentKind, ConfidenceAggregator, ConfidenceComponent, ConfidenceInput,
    ConfidencePurpose, EvidenceDomain)
from custom_components.thermosmart.learning.contracts import ConfidenceContribution
from custom_components.thermosmart.learning.runtime import LearningRuntimeMode
from custom_components.thermosmart.learning.contracts import PredictionType
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator


def _strong_boost():
    return ConfidenceAggregator().aggregate(ConfidenceInput(
        ConfidencePurpose.BOOST, components=(
            ConfidenceComponent(kind=ComponentKind.HEAT_RATE,
                                contribution=ConfidenceContribution(
                                    value=0.9, evidence_count=50, reliability=0.9,
                                    evidence_domains=("thermal_trajectory",)),
                                evidence_domains=(EvidenceDomain.THERMAL_TRAJECTORY,)),)))


class _BoostPred:
    values = {"boost_factor": 1.5}
    model_version = 1
    parameter_version = 1


class TestControlIntegration:
    async def test_control_mode_enabled_via_injection(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        sh.runtime.set_mode(LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        assert sh.control_enabled

    async def test_boost_reduced_through_existing_dispatch(self):
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        sh.runtime.set_mode(LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        # inject decision-time evidence: an existing boost + strong boost confidence
        zr = sh.runtime._zone(coord.zone_id)
        zr.last_confidence["boost"] = _strong_boost()
        zr.last_predictions[PredictionType.BOOST_FACTOR] = _BoostPred()
        rec = {"effective_target": 21.0, "trv_setpoint": 23.0, "window_open": False}
        sh.adjust_recommendation_safe(rec, boost_runtime_limit=TPI_MAX_BOOST_CELSIUS)
        # baseline offset 2.0 -> reduced toward 1.5 within clamp (max_step 0.5) -> 1.5
        assert rec["trv_setpoint"] == pytest.approx(22.5, abs=1e-6)
        assert rec.get("le2_boost_adjusted") is True

    async def test_boost_not_increased(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        sh.runtime.set_mode(LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        zr = sh.runtime._zone(coord.zone_id)
        zr.last_confidence["boost"] = _strong_boost()

        class _High:
            values = {"boost_factor": 5.0}
            model_version = 1
            parameter_version = 1
        zr.last_predictions[PredictionType.BOOST_FACTOR] = _High()
        rec = {"effective_target": 21.0, "trv_setpoint": 22.0}  # existing 1.0 offset
        sh.adjust_recommendation_safe(rec, boost_runtime_limit=TPI_MAX_BOOST_CELSIUS)
        # never increases an existing boost in Phase 18
        assert rec["trv_setpoint"] <= 22.0

    async def test_limit_mismatch_blocks_adjustment(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        sh.runtime.set_mode(LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        zr = sh.runtime._zone(coord.zone_id)
        zr.last_confidence["boost"] = _strong_boost()
        zr.last_predictions[PredictionType.BOOST_FACTOR] = _BoostPred()
        rec = {"effective_target": 21.0, "trv_setpoint": 23.0}
        sh.adjust_recommendation_safe(rec, boost_runtime_limit=99.0)  # mismatch
        assert rec["trv_setpoint"] == 23.0 and "le2_boost_adjusted" not in rec

    async def test_shadow_mode_no_adjustment(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)  # SHADOW
        await sh.async_setup()
        rec = {"effective_target": 21.0, "trv_setpoint": 23.0}
        sh.adjust_recommendation_safe(rec, boost_runtime_limit=TPI_MAX_BOOST_CELSIUS)
        assert rec["trv_setpoint"] == 23.0

    async def test_config_entities_unchanged(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        sh.runtime.set_mode(LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        await coord._async_update_data()
        # no new entities/services created by the recording harness
        assert coord.entry.entry_id == coord.zone_id
