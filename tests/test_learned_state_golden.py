"""Golden contracts for control driven by *learned* thermal state.

The zone's heat rate, heat loss, afterheat and onset delay models are seeded with
a learned general state (see ``inject_learned_thermal_state``) and run through the
real coordinator cycle: real orchestrator predictions, real Read Gates, real TPI
blending, real preheat decision and the real actuator path. Nothing is patched
except the schedule's minutes-until-comfort and the base target (the schedule is
not what is under test).

The expected values were captured from the code as it behaves today. They are the
safety net for later simplification of the learning runtime: a refactor must
reproduce them, and a deliberate behaviour change must update them knowingly.

Two facts these contracts record on purpose:

* TPI coefficients move towards the learned value in bounded steps, one step per
  cycle, so the contract pins the whole transition.
* A learned afterheat does not lower the target early. The afterheat model
  publishes ``EXPECTED_OVERSHOOT``, which shortens the preheat estimate; nothing
  else acts on it. (An early-cutoff mechanism used to sit on top of it but never
  ran, because the prediction it read was never produced, and has been removed.)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
from tests.helpers import make_state
from tests.helpers_ha_runtime import (
    FakeStore,
    attach_shadow,
    inject_learned_thermal_state,
    make_dispatching_coordinator,
)

_START = datetime(2026, 1, 12, 6, 0, tzinfo=timezone.utc)
_TRV = "climate.trv_test"


async def _zone(indoor: float, *, learned: bool, hvac_action=None, base_target=None,
                minutes_until_comfort=None, **learned_kw):
    clock = FakeClock(_START)
    attrs = {"hvac_action": hvac_action} if hvac_action else {}
    coord = make_dispatching_coordinator(
        {"schedule_enabled": False}, indoor=str(indoor), climate_states={_TRV: attrs}
    )
    coord._clock = clock
    coord.set_active_control(True)
    shadow = attach_shadow(coord, mode=LearningRuntimeMode.CONTROL, store=FakeStore(),
                           clock=clock)
    await shadow.async_setup()
    if learned:
        inject_learned_thermal_state(shadow, **learned_kw)
    if base_target is not None:
        coord.learning_engine.async_get_base_target = AsyncMock(return_value=base_target)
    if minutes_until_comfort is not None:
        coord._minutes_until_next_comfort = lambda cfg: minutes_until_comfort

    original = coord.hass.states.get.side_effect
    room = {"temp": indoor}

    def _get(entity_id):
        if entity_id == "sensor.test_temp":
            return make_state(f"{room['temp']:.1f}")
        return original(entity_id)

    coord.hass.states.get = MagicMock(side_effect=_get)
    return coord, shadow, clock, room


async def _cycle(coord, clock):
    zone = (await coord._async_update_data())["zone"]
    clock.advance(300)
    return zone


def _rounded(value, ndigits=3):
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, tuple):
        return tuple(_rounded(v, ndigits) for v in value)
    return value


# ── B1  learned heat rate / heat loss drive TPI ──────────────────────────────

async def run_tpi_case(*, learned: bool) -> dict:
    coord, shadow, clock, _ = await _zone(20.4, learned=learned)
    rows = []
    for _ in range(8):
        zone = await _cycle(coord, clock)
        rows.append((
            zone["tpi_coef_source"], _rounded(zone["tpi_coef_int"]), _rounded(zone["tpi_coef_ext"]),
            _rounded(zone["tpi_duty_cycle"]), _rounded(zone["trv_setpoint"]),
        ))
    return {
        "cycles": rows,
        "heat_rate": _rounded(shadow.read_heat_rate_safe()),
        "heat_loss": _rounded(shadow.read_heat_loss_rate_safe()),
    }


# ── B2  learned preheat ──────────────────────────────────────────────────────

_PREHEAT_CASES = {
    "learned_transition_inside_window": dict(learned=True, minutes=45.0),
    "learned_transition_outside_window": dict(learned=True, minutes=120.0),
    "learned_without_afterheat": dict(learned=True, minutes=45.0, afterheat_rise_c=0.0),
    "no_evidence_uses_deterministic_baseline": dict(learned=False, minutes=45.0),
}


async def run_preheat_case(name: str) -> dict:
    case = dict(_PREHEAT_CASES[name])
    minutes = case.pop("minutes")
    learned = case.pop("learned")
    coord, shadow, clock, _ = await _zone(
        17.5, learned=learned, base_target=17.0, minutes_until_comfort=minutes, **case)
    zone = {}
    for _ in range(3):
        zone = await _cycle(coord, clock)
    return {
        "preheat_minutes": _rounded(zone["preheat_minutes"]),
        "preheat_active": zone["preheat_active"],
        "effective_target": _rounded(zone["effective_target"]),
        "trv_setpoint": _rounded(zone["trv_setpoint"]),
        "onset_delay_min": _rounded(zone["onset_delay_min"]),
        "read": _rounded(shadow.read_preheat_minutes_safe(17.5, 21.0)),
    }


# ── B3  learned afterheat does not lower the target early ────────────────────

async def run_early_cutoff_case() -> list:
    coord, shadow, clock, room = await _zone(
        18.0, learned=True, hvac_action="heating", afterheat_rise_c=0.8)
    rows = []
    for _ in range(14):
        room["temp"] += 0.4
        zone = await _cycle(coord, clock)
        rows.append((
            round(room["temp"], 1), shadow._runtime._zone(shadow._zone).last_regime,
            _rounded(zone["effective_target"]),
        ))
    return rows


# ── expected values ──────────────────────────────────────────────────────────

_EXPECTED_TPI = {True: {'cycles': [('prediction_missing', 0.6, 0.01, 47.0, 24.8),
                   ('valid_learning', 0.5, 0.01, 41.0, 24.3),
                   ('valid_learning', 0.4, 0.01, 35.0, 23.8),
                   ('valid_learning', 0.4, 0.01, 35.0, 23.8),
                   ('valid_learning', 0.4, 0.01, 35.0, 23.8),
                   ('valid_learning', 0.4, 0.01, 35.0, 23.8),
                   ('valid_learning', 0.4, 0.01, 35.0, 23.8),
                   ('valid_learning', 0.4, 0.01, 35.0, 23.8)],
        'heat_rate': (4.0, 'valid'),
        'heat_loss': (0.8, 'valid')},
 False: {'cycles': [('prediction_missing', 0.6, 0.01, 47.0, 24.8),
                    ('cold_start', 0.6, 0.01, 47.0, 24.8),
                    ('cold_start', 0.6, 0.01, 47.0, 24.8),
                    ('cold_start', 0.6, 0.01, 47.0, 24.8),
                    ('cold_start', 0.6, 0.01, 47.0, 24.8),
                    ('cold_start', 0.6, 0.01, 47.0, 24.8),
                    ('cold_start', 0.6, 0.01, 47.0, 24.8),
                    ('cold_start', 0.6, 0.01, 47.0, 24.8)],
         'heat_rate': (0.0, 'cold_start'),
         'heat_loss': (0.0, 'cold_start')}}
_EXPECTED_PREHEAT = {'learned_transition_inside_window': {'preheat_minutes': 62.4,
                                      'preheat_active': True,
                                      'effective_target': 21.0,
                                      'trv_setpoint': 28.0,
                                      'onset_delay_min': 8.0,
                                      'read': (62.4, 'valid')},
 'learned_transition_outside_window': {'preheat_minutes': 62.4,
                                       'preheat_active': False,
                                       'effective_target': 17.0,
                                       'trv_setpoint': 17.0,
                                       'onset_delay_min': 8.0,
                                       'read': (62.4, 'valid')},
 'learned_without_afterheat': {'preheat_minutes': 73.6,
                               'preheat_active': True,
                               'effective_target': 21.0,
                               'trv_setpoint': 28.0,
                               'onset_delay_min': 8.0,
                               'read': (73.6, 'valid')},
 'no_evidence_uses_deterministic_baseline': {'preheat_minutes': 145.0,
                                             'preheat_active': True,
                                             'effective_target': 21.0,
                                             'trv_setpoint': 28.0,
                                             'onset_delay_min': 5.0,
                                             'read': (145.0, 'deterministic_baseline')}}
_EXPECTED_EARLY_CUTOFF = [(18.4, 'active_heating', 21.0),
 (18.8, 'active_heating', 21.0),
 (19.2, 'active_heating', 21.0),
 (19.6, 'active_heating', 21.0),
 (20.0, 'active_heating', 21.0),
 (20.4, 'active_heating', 21.0),
 (20.8, 'active_heating', 21.0),
 (21.2, 'active_heating', 21.0),
 (21.6, 'active_heating', 21.0),
 (22.0, 'active_heating', 21.0),
 (22.4, 'active_heating', 21.0),
 (22.8, 'active_heating', 21.0),
 (23.2, 'active_heating', 21.0),
 (23.6, 'active_heating', 21.0)]


@pytest.mark.parametrize("learned", [True, False])
async def test_tpi_follows_the_learned_heat_state(learned):
    assert await run_tpi_case(learned=learned) == _EXPECTED_TPI[learned]


@pytest.mark.parametrize("name", sorted(_PREHEAT_CASES))
async def test_preheat_follows_the_learned_heat_state(name):
    assert await run_preheat_case(name) == _EXPECTED_PREHEAT[name]


async def test_learned_afterheat_does_not_lower_the_target_early():
    assert await run_early_cutoff_case() == _EXPECTED_EARLY_CUTOFF


async def test_learning_switch_off_ignores_the_learned_state():
    coord, _, clock, _ = await _zone(20.4, learned=True)
    coord.learning_engine.is_zone_enabled.return_value = False
    zone = {}
    for _ in range(4):
        zone = await _cycle(coord, clock)
    assert zone["tpi_coef_source"] == "deterministic_baseline"
    assert (zone["tpi_coef_int"], zone["tpi_duty_cycle"], zone["trv_setpoint"]) == (0.6, 47.0, 24.8)


async def test_the_zone_state_carries_no_early_cutoff_keys():
    """The early-cutoff mechanism is gone; its keys must not reappear unnoticed."""
    coord, _, clock, _ = await _zone(20.4, learned=True)
    zone = await _cycle(coord, clock)
    assert not [key for key in zone if "early_cutoff" in key]
