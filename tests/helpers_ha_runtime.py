"""Harness for Learning HA shadow-integration tests.

Runs the REAL ThermoSmartCoordinator._async_update_data (with control side-effects
stubbed and recorded) so the production Learning shadow hook is exercised. Pure
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


def attach_shadow(coord, *, store=None, zone_id=None, clock=None, mode=None):
    from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
    _clock = clock if clock is not None else getattr(coord, "_clock", None)
    _mode = mode if mode is not None else LearningRuntimeMode.SHADOW
    shadow = LearningShadowController(
        hass=coord.hass, zone_id=zone_id or coord.zone_id,
        store=store or FakeStore(),
        clock=_clock,
        mode=_mode,
    )
    coord.attach_learning_shadow(shadow)
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

    # Mark active_control initialized so restore barrier passes
    coord.set_active_control(True)

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


def inject_learned_thermal_state(shadow, *, heat_rate_c_per_h: float = 4.0,
                                 heat_loss_c_per_h: float = 0.8,
                                 afterheat_rise_c: float = 0.6,
                                 onset_delay_min: float = 8.0,
                                 samples: int = 25):
    """Seed the thermal models with a learned (non-fallback, high confidence) state.

    Equivalent to restoring a snapshot of a zone that has seen ``samples`` clean
    heating cycles: the *general* bucket of the heat rate, heat loss, afterheat and
    onset delay models carries the given values. The predictions the real
    orchestrator derives from that state are what the control path reads — nothing
    is patched. Call after ``shadow.async_setup()``.
    """
    import dataclasses

    from custom_components.thermosmart.learning.models.afterheat import AfterheatBucket, _Dim
    from custom_components.thermosmart.learning.models.heat_loss import HeatLossBucket
    from custom_components.thermosmart.learning.models.heat_rate import HeatRateBucket
    from custom_components.thermosmart.learning.models.onset_delay import OnsetDelayBucket

    ts = shadow._utcnow_iso()
    models = shadow._runtime._zone(shadow._zone).orchestrator.models
    n = float(samples)

    st = models["heat_rate"]._state
    models["heat_rate"]._state = dataclasses.replace(
        st, general=HeatRateBucket("general", heat_rate_c_per_h, n, 0.05 * heat_rate_c_per_h,
                                   samples, ts),
        aggregate_reliability=0.9, last_update_ts=ts)

    st = models["heat_loss"]._state
    loss = HeatLossBucket("general", heat_loss_c_per_h, n, 0.1, samples, ts)
    models["heat_loss"]._state = dataclasses.replace(
        st, general=loss, normalized=dataclasses.replace(loss, key="normalized"),
        aggregate_reliability=0.9, last_update_ts=ts)

    st = models["afterheat"]._state
    models["afterheat"]._state = dataclasses.replace(
        st, general=AfterheatBucket(
            "general", rise=_Dim(afterheat_rise_c, n, 0.05, samples),
            time_to_peak=_Dim(900.0, n, 60.0, samples),
            duration=_Dim(1800.0, n, 60.0, samples),
            decay=_Dim(0.5, n, 0.05, samples), sample_count=samples),
        aggregate_reliability=0.9, last_update_ts=ts)

    st = models["onset_delay"]._state
    models["onset_delay"]._state = dataclasses.replace(
        st, general=OnsetDelayBucket("general", onset_delay_min, n, 1.0, samples, ts),
        aggregate_reliability=0.9, last_update_ts=ts)
