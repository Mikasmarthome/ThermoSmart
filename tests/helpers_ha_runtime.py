"""Harness for LE 2.0 HA shadow-integration tests.

Runs the REAL ThermoSmartCoordinator._async_update_data (with control side-effects
stubbed and recorded) so the production LE-2.0 shadow hook is exercised. Pure
Python + MagicMock hass — runs on Windows and in Docker.

``DispatchingCoordinator`` is the E2E variant: it runs the REAL ``_apply_temperature``
path (device clamping, step-snap, effective setpoint tracking) with mocked HA services.
``inject_eligible_boost_model`` seeds an eligible BoostModel state into a shadow zone
so the real ``BoostActivationReadiness`` evaluator returns ``eligibility=True``
without patching any function.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch as _patch

from custom_components.thermosmart.coordinator import ThermoSmartCoordinator
from custom_components.thermosmart.learning.runtime.ha_integration import (
    LearningShadowController,
)
from custom_components.thermosmart.trv_control import _DispatchStats
from tests.helpers import make_coordinator, make_state, set_hass_states


class RecordingCoordinator(ThermoSmartCoordinator):
    """Coordinator that records (not performs) every control side-effect."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.control_calls: list = []

    async def _reset_valve_opening_degree(self) -> bool:
        return True

    async def _async_apply_quirks(self, cfg):
        pass

    async def _watchdog_hvac(self, cfg, recommendation):
        pass

    async def _async_calibrate_trvs(self, cfg, recommendation):
        pass

    async def _apply_temperature(self, cfg, recommendation):
        self.control_calls.append(("apply_temperature", recommendation.get("trv_setpoint"),
                                   recommendation.get("effective_target")))
        return _DispatchStats()

    async def _async_set_valve_percent(self, cfg, duty):
        self.control_calls.append(("set_valve_percent", duty))
        return _DispatchStats()

    async def _apply_frost_protection(self, cfg):
        self.control_calls.append(("frost_protection",))

    async def _async_valve_maintenance(self, cfg, recommendation):
        pass

    async def _async_write_external_temp(self, cfg, recommendation):
        pass

    async def _async_manage_temp_source(self, cfg, recommendation):
        pass

    async def _async_observe_trv_setpoints(self, cfg, recommendation, weather):
        pass

    def _check_heating_failure(self, recommendation):
        pass


def make_recording_coordinator(zone_cfg_overrides=None, *, indoor="19.0"):
    base = make_coordinator(zone_cfg_overrides)
    with _patch("homeassistant.helpers.frame.report_usage"):
        coord = RecordingCoordinator(base.hass, base.entry, base.weather_engine,
                                     base.learning_engine)
    coord._valve_reset_done = True
    set_hass_states(coord, {"sensor.test_temp": make_state(indoor)})
    return coord


class FakeStore:
    def __init__(self, data=None, *, fail_load=False, fail_save=False):
        self.data = data
        self.saves = 0
        self._fail_load = fail_load
        self._fail_save = fail_save

    async def async_load(self):
        if self._fail_load:
            raise IOError("load boom")
        return self.data

    async def async_save(self, data):
        if self._fail_save:
            raise IOError("save boom")
        self.data = data
        self.saves += 1


def attach_shadow(coord, *, store=None, zone_id=None):
    shadow = LearningShadowController(hass=coord.hass, zone_id=zone_id or coord.zone_id,
                                     store=store or FakeStore())
    coord.attach_le2_shadow(shadow)
    return shadow


# ── DispatchingCoordinator — real _apply_temperature, mocked HA services ─────

class DispatchingCoordinator(ThermoSmartCoordinator):
    """Real _apply_temperature execution with mocked HA service backend.

    Stubs all side-effects EXCEPT _apply_temperature so the real device
    clamping, step-snap, effective-setpoint tracking and _DispatchStats
    population run unchanged.  hass.services.async_call is replaced by
    a configurable mock so tests can simulate success, failure, or partial.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.service_calls: list[dict] = []
        self.control_calls: list = []
        # _boost_active is not initialized in coordinator.py (production uses HA setup path).
        # We initialize it here so the real _apply_temperature can run in tests.
        self._boost_active: dict = {}

    async def _reset_valve_opening_degree(self) -> bool:
        return True

    async def _async_apply_quirks(self, cfg):
        pass

    async def _watchdog_hvac(self, cfg, recommendation):
        pass

    async def _async_calibrate_trvs(self, cfg, recommendation):
        pass

    # _apply_temperature NOT overridden → real TRVControlMixin implementation runs

    async def _async_set_valve_percent(self, cfg, duty):
        self.control_calls.append(("set_valve_percent", duty))
        return _DispatchStats()

    async def _apply_frost_protection(self, cfg):
        pass

    async def _async_valve_maintenance(self, cfg, recommendation):
        pass

    async def _async_write_external_temp(self, cfg, recommendation):
        pass

    async def _async_manage_temp_source(self, cfg, recommendation):
        pass

    async def _async_observe_trv_setpoints(self, cfg, recommendation, weather):
        pass

    def _check_heating_failure(self, recommendation):
        pass


def make_dispatching_coordinator(zone_cfg_overrides=None, *, indoor="18.0",
                                 climate_entities=None,
                                 climate_states=None,
                                 service_side_effect=None):
    """Create a DispatchingCoordinator with configured HA-service mock.

    Args:
        indoor: indoor temperature sensor value string.
        climate_entities: list of entity IDs to include in zone config.
        climate_states: dict mapping entity_id → state attributes dict. If not
            provided, defaults to a heat-mode entity with min=5, max=35, temp=21.
        service_side_effect: callable(domain, service, data, **kw) or list of
            exceptions/None aligned with call order (None = success, exc = failure).
            If None, all service calls succeed.
    Returns:
        (coord, service_mock) tuple.
    """
    from homeassistant.core import State as _HaState

    _climate_entities = climate_entities or ["climate.trv_test"]
    overrides = {"climate_entities": _climate_entities, **(zone_cfg_overrides or {})}
    base = make_coordinator(overrides)

    with _patch("homeassistant.helpers.frame.report_usage"):
        coord = DispatchingCoordinator(base.hass, base.entry,
                                       base.weather_engine, base.learning_engine)
    coord._valve_reset_done = True
    coord._last_written_setpoints = {}

    # Build per-entity climate state dicts.
    # Note: _apply_temperature reads "target_temp_step" (not "temperature_step") for step-snap.
    # Default attributes include both for compatibility, but counterfactual tests must use
    # target_temp_step to exercise the real step-snap path.
    _climate_attrs: dict[str, dict] = {}
    for _eid in _climate_entities:
        _a = {"temperature": 21.0, "min_temp": 5.0, "max_temp": 35.0,
              "hvac_mode": "heat", "target_temp_step": 0.5}
        if climate_states and _eid in climate_states:
            _a.update(climate_states[_eid])
        _climate_attrs[_eid] = _a

    # Sensor state (non-climate entities use make_state/set_hass_states)
    _sensor_state = make_state(indoor)
    set_hass_states(coord, {"sensor.test_temp": _sensor_state})

    # Override hass.states.get to return proper mock states with attributes
    def _states_get(entity_id):
        if entity_id in _climate_attrs:
            _st = MagicMock()
            _st.state = "heat"
            _st.attributes = _climate_attrs[entity_id]
            return _st
        if entity_id == "sensor.test_temp":
            return _sensor_state
        return None

    coord.hass.states.get = MagicMock(side_effect=_states_get)

    # Service call mock
    _service_calls = []

    if callable(service_side_effect):
        async def _service_call(domain, service, data=None, *, blocking=True, **kw):
            _service_calls.append({"domain": domain, "service": service, "data": data})
            return service_side_effect(domain, service, data, blocking=blocking)
    elif isinstance(service_side_effect, list):
        _call_idx = [0]
        async def _service_call(domain, service, data=None, *, blocking=True, **kw):
            _service_calls.append({"domain": domain, "service": service, "data": data})
            idx = _call_idx[0]
            _call_idx[0] += 1
            if idx < len(service_side_effect):
                exc = service_side_effect[idx]
                if exc is not None:
                    raise exc
    else:
        async def _service_call(domain, service, data=None, *, blocking=True, **kw):
            _service_calls.append({"domain": domain, "service": service, "data": data})

    coord.hass.services.async_call = _service_call
    coord._service_calls = _service_calls

    # Activate both switches and mark initialized so restore barrier passes
    coord.set_active_control(True)
    coord.set_adaptive_boost_enabled(True)

    return coord


def inject_eligible_boost_model(shadow, *, factor_c: float = 1.5):
    """Seed a BoostModel with eligible state in the zone's orchestrator.

    The real ``BoostActivationReadiness`` evaluator runs on this state and
    returns ``eligibility=True``.  No function is patched — only model state
    is injected (equivalent to seeding a DB before a test).

    Args:
        shadow: a ``LearningShadowController`` after ``async_setup()`` has run.
        factor_c: boost offset to inject (default 1.5 °C).
    """
    from custom_components.thermosmart.learning.models.boost import (
        BoostBucket, BoostState, _Dim)
    from custom_components.thermosmart.learning.models.boost_degradation import (
        BoostDegradationState, BoostLearningReadiness, BoostScopedDegradation)

    # effective_n=15 → confidence=15/20=0.75 >= 0.55; reliability=0.75 >= 0.5
    _ef = _Dim(value=factor_c, effective_n=15.0, dispersion=0.05, sample_count=20)
    _general = BoostBucket(key="general", effective_factor=_ef, sample_count=20)
    # factor_history with 3 identical values → _factor_stable=True (spread=0.0 ≤ 0.5)
    _degrad = BoostDegradationState(
        readiness=BoostLearningReadiness.ELIGIBLE,
        factor_history=(factor_c, factor_c, factor_c),
        degradation_score=0.0,
        overshoot_evidence=0.0,
    )
    _state = BoostState(
        learning_zone_id=shadow._zone,
        general=_general,
        degradation=BoostScopedDegradation(general=_degrad),
    )
    model = shadow._boost_model()
    if model is None:
        raise RuntimeError("inject_eligible_boost_model: no boost model in zone — "
                           "call async_setup() before injecting state")
    model._state = _state
