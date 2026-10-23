"""Wiring fixes found by the simplification audit.

* The Learning switch must control Learning in the coordinator (it used to
  only change the status sensor text; control read a config flag nobody sets).
* Diagnostics runtime_state must read attributes the coordinator really has.
* The live decision record must carry the onset delay the model predicts.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.const import DOMAIN
from custom_components.thermosmart.diagnostics import async_get_config_entry_diagnostics
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
from tests.helpers import make_coordinator
from tests.helpers_ha_runtime import (
    FakeStore,
    attach_shadow,
    make_dispatching_coordinator,
)


class TestLearningSwitchControlsLearning:
    def test_switch_off_turns_learning_mode_off(self):
        coord = make_coordinator()
        coord.learning_engine.is_zone_enabled.return_value = False
        assert coord._learning_mode_enabled(coord.zone_cfg) is False

    def test_switch_on_keeps_learning_mode_on(self):
        coord = make_coordinator()
        coord.learning_engine.is_zone_enabled.return_value = True
        assert coord._learning_mode_enabled(coord.zone_cfg) is True

    def test_config_flag_off_still_wins(self):
        coord = make_coordinator({"learning_enabled": False})
        coord.learning_engine.is_zone_enabled.return_value = True
        assert coord._learning_mode_enabled(coord.zone_cfg) is False

    async def test_switch_state_selects_adaptive_or_deterministic_control(self):
        coord = make_dispatching_coordinator(
            {"schedule_enabled": False, "boost_bootstrap_prior_c": 1.0}, indoor="19.5"
        )
        shadow = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore())
        await shadow.async_setup()

        coord.learning_engine.is_zone_enabled.return_value = True
        on = (await coord._async_update_data())["zone"]
        assert on["adaptation_mode"] == "adaptive"
        assert on["applied_boost_offset_c"] == 0.5

        coord.learning_engine.is_zone_enabled.return_value = False
        off = (await coord._async_update_data())["zone"]
        assert off["adaptation_mode"] == "deterministic"
        assert off["tpi_coef_source"] == "deterministic_baseline"
        assert off["applied_boost_offset_c"] == 0.0

        coord.learning_engine.is_zone_enabled.return_value = True
        back = (await coord._async_update_data())["zone"]
        assert back["adaptation_mode"] == "adaptive"


class TestDiagnosticsRuntimeStateReadsRealAttributes:
    async def test_values_come_from_the_coordinator_not_from_missing_attributes(self):
        coord = make_coordinator()
        coord._active_control = False
        coord._mode = "eco"
        coord.learning_engine.is_zone_enabled.return_value = False
        coord.data = {"learning_confidence": 0.5}
        coord._learning_shadow = None

        entry = MagicMock()
        entry.entry_id = "zone_1"
        entry.data = {"entry_type": "zone"}
        entry.options = {}
        hass = MagicMock()
        hass.config_entries.async_entries.return_value = [entry]
        hass.data = {DOMAIN: {"zone_1": {"coordinator": coord}}}

        result = await async_get_config_entry_diagnostics(hass, entry)

        state = result["runtime_state"]
        assert state["learning_enabled"] is False
        assert state["current_mode"] == "eco"
        assert state["active_control"] is False
        assert None not in state.values()


class TestLiveDecisionRecordOnsetDelay:
    async def test_valid_onset_prediction_reaches_the_record(self):
        coord = make_dispatching_coordinator({"schedule_enabled": False}, indoor="19.5")
        shadow = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore())
        await shadow.async_setup()
        await coord._async_update_data()
        record = coord._last_live_decision
        assert record is not None

        from custom_components.thermosmart.learning.contracts import PredictionType
        zr = shadow._runtime._zone(shadow._zone)
        zr.last_predictions = {
            PredictionType.ONSET_DELAY: SimpleNamespace(
                values={"onset_delay": 7.5}, fallback_used=False, confidence=0.95
            )
        }

        enriched = shadow.enrich_live_decision_pre_dispatch(record, {})
        assert enriched.onset_delay_min == pytest.approx(7.5)
        assert enriched.onset_delay_source == "valid"
