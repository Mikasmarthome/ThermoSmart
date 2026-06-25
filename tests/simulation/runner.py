"""ScenarioRunner — drives ThermoSmartCoordinator through a simulated environment.

Architecture
------------
SimulationClock patches homeassistant.util.dt globally → all production code
sees deterministic simulation time.

VirtualRoomPhysics models one room thermally. The coordinator reads sensor
state and dispatches setpoints; the runner applies dispatched setpoints back
to the physics model and advances the room temperature.

Only the physical environment is simulated. All ThermoSmart logic (TPI, LE2,
gates, episode logic, dispatch, device clamping) runs as-is from production.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
from unittest.mock import AsyncMock, MagicMock, patch as _patch

from tests.helpers import make_coordinator, make_state, make_zone_config
from tests.helpers_ha_runtime import (
    DispatchingCoordinator,
    FakeStore,
    attach_shadow,
    inject_eligible_boost_model,
)
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
from tests.simulation.clock import SimulationClock
from tests.simulation.metrics import MetricsCollector, MetricsSummary
from tests.simulation.physics import RoomProfile, VirtualRoomPhysics


@dataclass
class ScenarioConfig:
    name: str
    room_profile: RoomProfile = field(default_factory=RoomProfile)
    initial_room_temp: float = 18.0
    start_utc: datetime = field(
        default_factory=lambda: datetime(2024, 1, 8, 0, 0, 0, tzinfo=timezone.utc)
    )
    tz_offset_hours: float = 1.0
    step_s: float = 300.0
    duration_s: float = 7 * 24 * 3600
    zone_cfg_overrides: dict = field(default_factory=dict)
    climate_entities: list = field(default_factory=lambda: ["climate.trv_sim"])
    sensor_entity: str = "sensor.room_sim"
    outdoor_temp_fn: Callable = field(default_factory=lambda: (lambda _t: 5.0))
    solar_gain_fn: Callable = field(default_factory=lambda: (lambda _t: 0.0))
    seed_boost_model: bool = False
    boost_seed_factor_c: float = 1.5
    # ControlAdaptationMode switches
    learning_mode: bool = True
    active_control: bool = True
    # List of (step_idx, entity_id, available: bool) — applied at that step
    fault_schedule: list = field(default_factory=list)
    # Set of step indices to simulate an HA restart at
    restart_steps: set = field(default_factory=set)

    @property
    def n_steps(self) -> int:
        return int(self.duration_s / self.step_s)


class ScenarioRunner:
    """Drives ThermoSmartCoordinator through the simulated environment.

    Call ``await runner.run()`` inside an asyncio event loop.
    """

    def __init__(self, cfg: ScenarioConfig) -> None:
        self.cfg = cfg
        self._clock: Optional[SimulationClock] = None
        self._room: Optional[VirtualRoomPhysics] = None
        self._coord: Optional[DispatchingCoordinator] = None
        self._shadow = None
        self._fake_store: Optional[FakeStore] = None
        # Mutable state shared with the hass.states.get closure
        self._trv_cmds: dict = {}
        self._sensor_value: list = [cfg.initial_room_temp]
        self._metrics: Optional[MetricsCollector] = None

    # ── Coordinator factory ──────────────────────────────────────────────

    def _build_zone_cfg_overrides(self) -> dict:
        overrides = {
            "climate_entities": self.cfg.climate_entities,
            "temp_sensors": [self.cfg.sensor_entity],
            "learning_enabled": self.cfg.learning_mode,
        }
        overrides.update(self.cfg.zone_cfg_overrides)
        return overrides

    def _build_coord(self) -> tuple[DispatchingCoordinator, FakeStore]:
        """Create a fresh DispatchingCoordinator wired to the simulation state."""
        from homeassistant.util import dt as dt_util

        zone_overrides = self._build_zone_cfg_overrides()
        base = make_coordinator(zone_overrides)
        zone_cfg_data = base.entry.data   # used inside closures below

        # -- Schedule-aware base target (uses patched dt_util.now) --
        async def _sched_base_target(
            zone_id, mode,
            comfort_temp=21.0, night_temp=18.0,
            away_temp=17.0, vacation_temp=12.0, eco_temp=19.0,
            schedule_cfg=None, **_kw,
        ):
            if mode == "auto":
                now = dt_util.now()
                is_we = now.weekday() >= 5
                cur = now.hour * 60 + now.minute
                src = schedule_cfg or zone_cfg_data

                def _t(key, default):
                    try:
                        parts = str(src.get(key, default)).split(":")
                        return int(parts[0]) * 60 + int(parts[1])
                    except Exception:
                        h, m = default.split(":")
                        return int(h) * 60 + int(m)

                morning = _t("sched_we_morning" if is_we else "sched_wd_morning", "06:00")
                night_start = _t("sched_we_night" if is_we else "sched_wd_night", "22:00")
                return comfort_temp if morning <= cur < night_start else night_temp
            return {
                "comfort": comfort_temp, "night": night_temp,
                "eco": eco_temp, "away": away_temp, "vacation": vacation_temp,
            }.get(mode, comfort_temp)

        base.learning_engine.async_get_base_target = AsyncMock(
            side_effect=_sched_base_target)

        # -- Dynamic outdoor temperature via weather engine --
        clock_cell = [self._clock]   # indirection so restart sees new clock ref

        async def _weather_data():
            t = clock_cell[0].now_utc() if clock_cell[0] else datetime.now(timezone.utc)
            return {
                "temperature": self.cfg.outdoor_temp_fn(t),
                "humidity": 70.0,
                "condition": "overcast",
            }

        base.weather_engine.async_get_data = AsyncMock(side_effect=_weather_data)

        # -- Build DispatchingCoordinator --
        # Pass simulation clock via the official constructor parameter so the coordinator's
        # fachliche time authority (EC hold timers, etc.) uses simulated time.
        _inject_clock = self._clock._fake if self._clock is not None else None
        with _patch("homeassistant.helpers.frame.report_usage"):
            coord = DispatchingCoordinator(
                base.hass, base.entry, base.weather_engine, base.learning_engine,
                clock=_inject_clock)
        coord._valve_reset_done = True
        coord._last_written_setpoints = {}
        coord._boost_active = {}

        # Initialise TRV state store (mutable, read by hass.states.get closure)
        for eid in self.cfg.climate_entities:
            if eid not in self._trv_cmds:
                self._trv_cmds[eid] = {
                    "temperature": self.cfg.initial_room_temp,
                    "min_temp": 5.0,
                    "max_temp": 35.0,
                    "hvac_mode": "heat",
                    "target_temp_step": 0.5,
                    "_state": "heat",   # internal: drives state.state
                }

        trv_cmds = self._trv_cmds
        sensor_value = self._sensor_value
        sensor_eid = self.cfg.sensor_entity

        def _states_get(entity_id):
            if entity_id in trv_cmds:
                attrs = trv_cmds[entity_id]
                mock_st = MagicMock()
                mock_st.state = attrs.get("_state", "heat")
                # Expose all attrs except internal _state key
                mock_st.attributes = {k: v for k, v in attrs.items() if not k.startswith("_")}
                return mock_st
            if entity_id == sensor_eid:
                return make_state(str(sensor_value[0]))
            return None

        coord.hass.states.get = MagicMock(side_effect=_states_get)

        # -- Service call capture --
        service_calls: list[dict] = []

        async def _service_call(domain, service, data=None, *, blocking=True, **_kw):
            service_calls.append({"domain": domain, "service": service, "data": data})

        coord.hass.services.async_call = _service_call
        coord._service_calls = service_calls

        # -- Initialise control switch --
        active = getattr(self.cfg, "active_control", True)
        coord.set_active_control(active)

        fake_store = FakeStore()
        return coord, fake_store

    # ── Simulation loop ──────────────────────────────────────────────────

    async def run(self) -> MetricsSummary:
        cfg = self.cfg
        resolved_cfg = make_zone_config(**self._build_zone_cfg_overrides())
        comfort_temp = resolved_cfg.get("comfort_temp", 21.0)
        night_temp = resolved_cfg.get("night_temp", 18.0)

        self._room = VirtualRoomPhysics(cfg.room_profile, cfg.initial_room_temp)
        self._metrics = MetricsCollector(comfort_temp=comfort_temp, night_temp=night_temp)
        self._clock = SimulationClock(cfg.start_utc, cfg.tz_offset_hours)
        self._coord, self._fake_store = self._build_coord()

        _shadow_mode = LearningRuntimeMode.CONTROL if getattr(self.cfg, "active_control", True) else LearningRuntimeMode.SHADOW
        self._shadow = attach_shadow(self._coord, store=self._fake_store, mode=_shadow_mode)
        if cfg.seed_boost_model:
            inject_eligible_boost_model(self._shadow, factor_c=cfg.boost_seed_factor_c)
        await self._shadow.async_setup()

        # Build fault map: step_idx → [(entity_id, available), ...]
        fault_by_step: dict[int, list] = {}
        for step_idx, eid, available in cfg.fault_schedule:
            fault_by_step.setdefault(step_idx, []).append((eid, available))

        with self._clock:
            for step_idx in range(cfg.n_steps):
                await self._step(step_idx, fault_by_step)

        return self._metrics.summarize()

    async def _step(self, step_idx: int, fault_by_step: dict) -> None:
        cfg = self.cfg

        # 1. Advance simulation time
        self._clock.advance(cfg.step_s)
        sim_time = self._clock.now_utc()

        # 2. Apply fault injections
        for eid, available in fault_by_step.get(step_idx, []):
            new_state = "heat" if available else "unavailable"
            self._trv_cmds[eid]["_state"] = new_state
            self._trv_cmds[eid]["hvac_mode"] = new_state

        # 3. Update sensor reading from current room temperature
        self._sensor_value[0] = self._room.read_sensor(noise=True)

        # 4. Run real coordinator cycle
        pre_count = len(self._coord._service_calls)
        result = await self._coord._async_update_data()
        zone = (result or {}).get("zone", {})

        # 5. Apply dispatched setpoints to TRV state and physics
        new_calls = self._coord._service_calls[pre_count:]
        for call in new_calls:
            if call.get("service") == "set_temperature":
                data = call.get("data") or {}
                eid = data.get("entity_id")
                new_sp = data.get("temperature")
                if eid and new_sp is not None and eid in self._trv_cmds:
                    self._trv_cmds[eid]["temperature"] = float(new_sp)

        # Physics uses highest available setpoint (handles multi-TRV and partial failure)
        active_sps = [
            self._trv_cmds[eid]["temperature"]
            for eid in cfg.climate_entities
            if self._trv_cmds[eid].get("_state", "heat") == "heat"
        ]
        if active_sps:
            self._room.set_setpoint(max(active_sps))

        # 6. Advance room physics
        outdoor = cfg.outdoor_temp_fn(sim_time)
        solar = cfg.solar_gain_fn(sim_time)
        self._room.step(cfg.step_s, outdoor, solar)

        # 7. Record metrics
        self._metrics.record(
            sim_time, zone, self._room, outdoor, cfg.step_s,
            service_calls_this_step=len(new_calls),
        )

        # 8. Simulate HA restart (S1-E)
        if step_idx in cfg.restart_steps:
            await self._do_restart()

    async def _do_restart(self) -> None:
        """Flush LE2 state, rebuild coordinator from saved store data."""
        # Force save current LE2 state
        await self._coord._le2_shadow._runtime.async_flush()
        saved = dict(self._fake_store.data or {})

        # Rebuild with restored state
        new_store = FakeStore(data=saved)
        new_coord, _ = self._build_coord()
        new_coord._service_calls = new_coord._service_calls  # already set in _build_coord

        _shadow_mode = LearningRuntimeMode.CONTROL if getattr(self.cfg, "active_control", True) else LearningRuntimeMode.SHADOW
        new_shadow = attach_shadow(new_coord, store=new_store, mode=_shadow_mode)
        await new_shadow.async_setup()

        self._coord = new_coord
        self._fake_store = new_store
        self._shadow = new_shadow
