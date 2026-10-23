"""Golden control values for the learning-driven control path.

Fixed inputs (room temperature, comfort target, weather mock, priors, boost
bootstrap prior) are run through the real coordinator cycle with an attached
Learning shadow in CONTROL mode and the real _apply_temperature actuator path.
The expected values were captured from the code before the simplification
phase 1 cleanup; removing dead learning code must not move any of them:
preheat minutes, TPI coefficients, effective target, TPI duty, TRV setpoint,
boost decision/offset, control reason and the issued service calls.

The learning_off scenario sets the zone config flag only; the Learning switch
state is covered separately in test_learning_switch_wiring.py.
"""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
from tests.helpers_ha_runtime import (
    FakeStore,
    attach_shadow,
    inject_eligible_boost_model,
    make_dispatching_coordinator,
)

_KEYS = [
    "adaptation_mode", "effective_target", "adjusted_target", "trv_setpoint", "tpi_duty_cycle",
    "tpi_coef_int", "tpi_coef_ext", "tpi_coef_source", "preheat_minutes", "preheat_active",
    "onset_delay_min", "boost_offset_c", "applied_boost_offset_c", "_boost_rejection_reason",
    "_boost_evaluation_status", "tpi_baseline_setpoint",
]


_SCENARIOS = {'priors_cold_room': {'indoor': '18.0', 'cfg': {}, 'eligible': False, 'active': True}, 'priors_near_target': {'indoor': '20.8', 'cfg': {}, 'eligible': False, 'active': True}, 'boost_applied_bootstrap_prior': {'indoor': '19.5', 'cfg': {'boost_bootstrap_prior_c': 1.0}, 'eligible': False, 'active': True}, 'boost_rejected_low_confidence': {'indoor': '19.5', 'cfg': {}, 'eligible': True, 'active': True}, 'learning_off_is_deterministic': {'indoor': '19.5', 'cfg': {'learning_enabled': False, 'boost_bootstrap_prior_c': 1.0}, 'eligible': True, 'active': True}, 'observation_writes_nothing': {'indoor': '19.5', 'cfg': {'boost_bootstrap_prior_c': 1.0}, 'eligible': True, 'active': False}}

_EXPECTED = {'priors_cold_room': [{'adaptation_mode': 'adaptive',
                       'effective_target': 21.0,
                       'adjusted_target': 21.0,
                       'trv_setpoint': 28.0,
                       'tpi_duty_cycle': 100.0,
                       'tpi_coef_int': 0.6,
                       'tpi_coef_ext': 0.01,
                       'tpi_coef_source': 'prediction_missing',
                       'preheat_minutes': 125.0,
                       'preheat_active': False,
                       'onset_delay_min': 5.0,
                       'boost_offset_c': 0.0,
                       'applied_boost_offset_c': 0.0,
                       '_boost_rejection_reason': 'insufficient_data',
                       '_boost_evaluation_status': 'evaluated_not_applied',
                       'tpi_baseline_setpoint': None,
                       'control_reason': 'schedule',
                       'calls': [('set_temperature', 28.0)]},
                      {'adaptation_mode': 'adaptive',
                       'effective_target': 21.0,
                       'adjusted_target': 21.0,
                       'trv_setpoint': 28.0,
                       'tpi_duty_cycle': 100.0,
                       'tpi_coef_int': 0.6,
                       'tpi_coef_ext': 0.01,
                       'tpi_coef_source': 'cold_start',
                       'preheat_minutes': 125.0,
                       'preheat_active': False,
                       'onset_delay_min': 5.0,
                       'boost_offset_c': 0.0,
                       'applied_boost_offset_c': 0.0,
                       '_boost_rejection_reason': 'insufficient_data',
                       '_boost_evaluation_status': 'evaluated_not_applied',
                       'tpi_baseline_setpoint': None,
                       'control_reason': 'schedule',
                       'calls': [('set_temperature', 28.0)]}],
 'priors_near_target': [{'adaptation_mode': 'adaptive',
                         'effective_target': 21.0,
                         'adjusted_target': 21.0,
                         'trv_setpoint': 22.8,
                         'tpi_duty_cycle': 23.0,
                         'tpi_coef_int': 0.6,
                         'tpi_coef_ext': 0.01,
                         'tpi_coef_source': 'prediction_missing',
                         'preheat_minutes': 0.0,
                         'preheat_active': False,
                         'onset_delay_min': 5.0,
                         'boost_offset_c': 0.0,
                         'applied_boost_offset_c': 0.0,
                         '_boost_rejection_reason': 'insufficient_data',
                         '_boost_evaluation_status': 'evaluated_not_applied',
                         'tpi_baseline_setpoint': None,
                         'control_reason': 'schedule',
                         'calls': [('set_temperature', 23.0)]},
                        {'adaptation_mode': 'adaptive',
                         'effective_target': 21.0,
                         'adjusted_target': 21.0,
                         'trv_setpoint': 22.8,
                         'tpi_duty_cycle': 23.0,
                         'tpi_coef_int': 0.6,
                         'tpi_coef_ext': 0.01,
                         'tpi_coef_source': 'cold_start',
                         'preheat_minutes': 0.0,
                         'preheat_active': False,
                         'onset_delay_min': 5.0,
                         'boost_offset_c': 0.0,
                         'applied_boost_offset_c': 0.0,
                         '_boost_rejection_reason': 'insufficient_data',
                         '_boost_evaluation_status': 'evaluated_not_applied',
                         'tpi_baseline_setpoint': None,
                         'control_reason': 'schedule',
                         'calls': [('set_temperature', 23.0)]}],
 'boost_applied_bootstrap_prior': [{'adaptation_mode': 'adaptive',
                                    'effective_target': 21.0,
                                    'adjusted_target': 21.0,
                                    'trv_setpoint': 28.5,
                                    'tpi_duty_cycle': 100.0,
                                    'tpi_coef_int': 0.6,
                                    'tpi_coef_ext': 0.01,
                                    'tpi_coef_source': 'prediction_missing',
                                    'preheat_minutes': 65.0,
                                    'preheat_active': False,
                                    'onset_delay_min': 5.0,
                                    'boost_offset_c': 0.5,
                                    'applied_boost_offset_c': 0.5,
                                    '_boost_rejection_reason': None,
                                    '_boost_evaluation_status': 'applied',
                                    'tpi_baseline_setpoint': 28.0,
                                    'control_reason': 'schedule',
                                    'calls': [('set_temperature', 28.5)]},
                                   {'adaptation_mode': 'adaptive',
                                    'effective_target': 21.0,
                                    'adjusted_target': 21.0,
                                    'trv_setpoint': 28.5,
                                    'tpi_duty_cycle': 100.0,
                                    'tpi_coef_int': 0.6,
                                    'tpi_coef_ext': 0.01,
                                    'tpi_coef_source': 'cold_start',
                                    'preheat_minutes': 65.0,
                                    'preheat_active': False,
                                    'onset_delay_min': 5.0,
                                    'boost_offset_c': 0.5,
                                    'applied_boost_offset_c': 0.5,
                                    '_boost_rejection_reason': None,
                                    '_boost_evaluation_status': 'applied',
                                    'tpi_baseline_setpoint': 28.0,
                                    'control_reason': 'schedule',
                                    'calls': [('set_temperature', 28.5)]}],
 'boost_rejected_low_confidence': [{'adaptation_mode': 'adaptive',
                                    'effective_target': 21.0,
                                    'adjusted_target': 21.0,
                                    'trv_setpoint': 28.0,
                                    'tpi_duty_cycle': 100.0,
                                    'tpi_coef_int': 0.6,
                                    'tpi_coef_ext': 0.01,
                                    'tpi_coef_source': 'prediction_missing',
                                    'preheat_minutes': 65.0,
                                    'preheat_active': False,
                                    'onset_delay_min': 5.0,
                                    'boost_offset_c': 0.0,
                                    'applied_boost_offset_c': 0.0,
                                    '_boost_rejection_reason': 'low_confidence',
                                    '_boost_evaluation_status': 'evaluated_not_applied',
                                    'tpi_baseline_setpoint': None,
                                    'control_reason': 'schedule',
                                    'calls': [('set_temperature', 28.0)]},
                                   {'adaptation_mode': 'adaptive',
                                    'effective_target': 21.0,
                                    'adjusted_target': 21.0,
                                    'trv_setpoint': 28.0,
                                    'tpi_duty_cycle': 100.0,
                                    'tpi_coef_int': 0.6,
                                    'tpi_coef_ext': 0.01,
                                    'tpi_coef_source': 'cold_start',
                                    'preheat_minutes': 65.0,
                                    'preheat_active': False,
                                    'onset_delay_min': 5.0,
                                    'boost_offset_c': 0.0,
                                    'applied_boost_offset_c': 0.0,
                                    '_boost_rejection_reason': 'low_confidence',
                                    '_boost_evaluation_status': 'evaluated_not_applied',
                                    'tpi_baseline_setpoint': None,
                                    'control_reason': 'schedule',
                                    'calls': [('set_temperature', 28.0)]}],
 'learning_off_is_deterministic': [{'adaptation_mode': 'deterministic',
                                    'effective_target': 21.0,
                                    'adjusted_target': 21.0,
                                    'trv_setpoint': 28.0,
                                    'tpi_duty_cycle': 100.0,
                                    'tpi_coef_int': 0.6,
                                    'tpi_coef_ext': 0.01,
                                    'tpi_coef_source': 'deterministic_baseline',
                                    'preheat_minutes': 65.0,
                                    'preheat_active': False,
                                    'onset_delay_min': 5.0,
                                    'boost_offset_c': 0.0,
                                    'applied_boost_offset_c': 0.0,
                                    '_boost_rejection_reason': 'learning_mode_off',
                                    '_boost_evaluation_status': 'evaluated_not_applied',
                                    'tpi_baseline_setpoint': None,
                                    'control_reason': 'schedule',
                                    'calls': [('set_temperature', 28.0)]},
                                   {'adaptation_mode': 'deterministic',
                                    'effective_target': 21.0,
                                    'adjusted_target': 21.0,
                                    'trv_setpoint': 28.0,
                                    'tpi_duty_cycle': 100.0,
                                    'tpi_coef_int': 0.6,
                                    'tpi_coef_ext': 0.01,
                                    'tpi_coef_source': 'deterministic_baseline',
                                    'preheat_minutes': 65.0,
                                    'preheat_active': False,
                                    'onset_delay_min': 5.0,
                                    'boost_offset_c': 0.0,
                                    'applied_boost_offset_c': 0.0,
                                    '_boost_rejection_reason': 'learning_mode_off',
                                    '_boost_evaluation_status': 'evaluated_not_applied',
                                    'tpi_baseline_setpoint': None,
                                    'control_reason': 'schedule',
                                    'calls': [('set_temperature', 28.0)]}],
 'observation_writes_nothing': [{'adaptation_mode': 'shadow_only',
                                 'effective_target': 21.0,
                                 'adjusted_target': 21.0,
                                 'trv_setpoint': 28.0,
                                 'tpi_duty_cycle': 100.0,
                                 'tpi_coef_int': 0.6,
                                 'tpi_coef_ext': 0.01,
                                 'tpi_coef_source': 'prediction_missing',
                                 'preheat_minutes': 65.0,
                                 'preheat_active': False,
                                 'onset_delay_min': 5.0,
                                 'boost_offset_c': 0.0,
                                 'applied_boost_offset_c': 0.0,
                                 '_boost_rejection_reason': 'active_control_off',
                                 '_boost_evaluation_status': 'evaluated_not_applied',
                                 'tpi_baseline_setpoint': None,
                                 'control_reason': 'schedule',
                                 'calls': []},
                                {'adaptation_mode': 'shadow_only',
                                 'effective_target': 21.0,
                                 'adjusted_target': 21.0,
                                 'trv_setpoint': 28.0,
                                 'tpi_duty_cycle': 100.0,
                                 'tpi_coef_int': 0.6,
                                 'tpi_coef_ext': 0.01,
                                 'tpi_coef_source': 'cold_start',
                                 'preheat_minutes': 65.0,
                                 'preheat_active': False,
                                 'onset_delay_min': 5.0,
                                 'boost_offset_c': 0.0,
                                 'applied_boost_offset_c': 0.0,
                                 '_boost_rejection_reason': 'active_control_off',
                                 '_boost_evaluation_status': 'evaluated_not_applied',
                                 'tpi_baseline_setpoint': None,
                                 'control_reason': 'schedule',
                                 'calls': []}]}


async def _run(scenario: dict) -> list[dict]:
    coord = make_dispatching_coordinator(
        {"schedule_enabled": False, **scenario["cfg"]}, indoor=scenario["indoor"]
    )
    coord.set_active_control(scenario["active"])
    shadow = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore())
    await shadow.async_setup()
    if scenario["eligible"]:
        inject_eligible_boost_model(shadow, factor_c=1.5)
    cycles = []
    for _ in range(2):
        zone = (await coord._async_update_data())["zone"]
        row = {key: getattr(zone.get(key), "value", zone.get(key)) for key in _KEYS}
        row["control_reason"] = coord._control_reason(zone)
        row["calls"] = [(c["service"], c["data"].get("temperature")) for c in coord._service_calls]
        coord._service_calls.clear()
        cycles.append(row)
    return cycles


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
async def test_control_values_match_the_golden_snapshot(name):
    assert await _run(_SCENARIOS[name]) == _EXPECTED[name]


async def test_golden_scenarios_cover_boost_applied_and_observation():
    assert _EXPECTED["boost_applied_bootstrap_prior"][0]["applied_boost_offset_c"] == 0.5
    assert all(not row["calls"] for row in _EXPECTED["observation_writes_nothing"])
