"""Phase 19A-B: confidence read transfer to LE 2.0 (display only; safe fallback)."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.confidence import ConfidencePurpose
from custom_components.thermosmart.learning.decision.confidence_adapter import (
    combined_confidence, confidence_breakdown)
from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator
from tests.helpers_control import strong_preheat, strong_boost


class _CR:
    def __init__(self, v):
        self.value = v


class TestConfidenceAdapter:
    def test_combined_mean(self):
        lc = {ConfidencePurpose.HEAT_RATE.value: _CR(0.8),
              ConfidencePurpose.FORECAST.value: _CR(0.4)}
        assert combined_confidence(lc) == pytest.approx(0.6)

    def test_combined_empty_neutral(self):
        assert combined_confidence({}) == 0.0

    def test_breakdown_keys_preserved(self):
        lc = {ConfidencePurpose.HEAT_RATE.value: _CR(0.8)}
        b = confidence_breakdown(lc, model_update_counts={"heat_rate": 5})
        for k in ("room_patterns_%", "trv_efficiency_%", "window_cooling_%",
                  "forecast_confidence_%", "total_observations", "trv_observations",
                  "window_events"):
            assert k in b
        assert b["room_patterns_%"] == 80.0

    def test_no_second_formula_uses_aggregator_values(self):
        # the breakdown % is exactly the aggregator value *100, no re-derivation
        lc = {ConfidencePurpose.FORECAST.value: _CR(0.55)}
        assert confidence_breakdown(lc)["forecast_confidence_%"] == 55.0


class TestConfidenceTransferLive:
    async def test_recommendation_confidence_from_le2(self):
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        zr = sh.runtime._zone(coord.zone_id)
        zr.last_confidence[ConfidencePurpose.HEAT_RATE.value] = _CR(0.7)
        result = await coord._async_update_data()
        assert result["zone"]["learning_confidence"] == pytest.approx(0.7, abs=1e-6)

    async def test_no_le2_neutral_fallback(self):
        # no shadow attached => neutral 0.0, never the legacy engine
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = True
        result = await coord._async_update_data()
        assert result["zone"]["learning_confidence"] == 0.0

    async def test_legacy_get_confidence_not_called_for_recommendation(self):
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        coord.learning_engine.get_confidence.reset_mock()
        await coord._async_update_data()
        # the recommendation no longer reads legacy confidence
        coord.learning_engine.get_confidence.assert_not_called()

    async def test_learning_failure_is_not_heating_failure(self):
        # a broken shadow confidence call must not raise into the cycle
        coord = make_recording_coordinator(indoor="18.0")
        coord._active_control = True
        sh = attach_shadow(coord)
        await sh.async_setup()
        sh.confidence_display = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        # coordinator guards the call; cycle still completes
        result = await coord._async_update_data()
        assert result is not None and "zone" in result
