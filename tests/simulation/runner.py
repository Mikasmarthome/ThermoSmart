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

Item 6 extensions:
  - ScenarioConfig.seed: applied via random.seed() before run() for reproducibility
  - ScenarioConfig.window_schedule: list of (step_idx, is_open) for window-open events
  - ScenarioConfig.model_snapshot_interval_steps: take ModelLearningSnapshot every N steps
  - ScenarioConfig.baseline_mode: when True, use SHADOW mode and disable active control
  - Per-step timing recorded via time.perf_counter() → MetricsSummary.perf
  - _extract_model_snapshot() pulls learned values from shadow controller
"""
from __future__ import annotations

import json
import math
import random
import time as _time
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
from tests.simulation.metrics import MetricsCollector, MetricsSummary, ModelLearningSnapshot
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

    # ── Item 6 extensions ─────────────────────────────────────────────────

    # RNG seed for deterministic noise; applied via random.seed() at run() start.
    # None means "do not touch the global random state" (caller manages the seed,
    # as existing S1 tests do with explicit random.seed() before ScenarioRunner).
    seed: int = None

    # List of (step_idx, is_open: bool) — window state changes
    window_schedule: list = field(default_factory=list)

    # Capture a ModelLearningSnapshot every N steps (0 = disabled; 288 = every day at 5-min)
    model_snapshot_interval_steps: int = 0

    # When True: run in SHADOW mode without active boost control
    # (used to isolate learning quality from control side-effects)
    baseline_mode: bool = False

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
        # Cumulative save count across ALL fake stores (survives restarts)
        self._total_saves: int = 0

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
                _mock_state = attrs.get("_state", "heat")
                mock_st.state = _mock_state
                exposed = {k: v for k, v in attrs.items() if not k.startswith("_")}
                # Add hvac_action to match real TRV behavior.
                # Real TRVs report hvac_action="heating" when in heat mode with setpoint
                # above current room temperature; the LE2 regime classifier uses this to
                # confirm ACTIVE_HEATING without needing a positive temperature trend.
                # Without this attribute the classifier falls back to trend-only detection,
                # which fails for Profile A/B during supply-delay ramps.
                if _mock_state == "heat":
                    _sp = float(exposed.get("temperature", 0.0))
                    _room_t = float(sensor_value[0])
                    exposed["hvac_action"] = (
                        "heating" if _sp > _room_t + 0.3 else "idle"
                    )
                else:
                    exposed["hvac_action"] = "off"
                mock_st.attributes = exposed
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
        # Deterministic RNG — applied only when the scenario explicitly requests a seed.
        # None means the caller manages the seed (backward-compatible with S1 tests that
        # call random.seed() before ScenarioRunner).
        if cfg.seed is not None:
            random.seed(cfg.seed)

        resolved_cfg = make_zone_config(**self._build_zone_cfg_overrides())
        comfort_temp = resolved_cfg.get("comfort_temp", 21.0)
        night_temp = resolved_cfg.get("night_temp", 18.0)

        self._room = VirtualRoomPhysics(cfg.room_profile, cfg.initial_room_temp)
        self._metrics = MetricsCollector(comfort_temp=comfort_temp, night_temp=night_temp)
        self._clock = SimulationClock(cfg.start_utc, cfg.tz_offset_hours)
        self._coord, self._fake_store = self._build_coord()

        # baseline_mode → SHADOW (observe only); otherwise CONTROL for active learning
        if cfg.baseline_mode:
            _shadow_mode = LearningRuntimeMode.SHADOW
        elif getattr(cfg, "active_control", True):
            _shadow_mode = LearningRuntimeMode.CONTROL
        else:
            _shadow_mode = LearningRuntimeMode.SHADOW

        self._shadow = attach_shadow(self._coord, store=self._fake_store, mode=_shadow_mode)
        if cfg.seed_boost_model and not cfg.baseline_mode:
            inject_eligible_boost_model(self._shadow, factor_c=cfg.boost_seed_factor_c)
        await self._shadow.async_setup()

        # Build fault map: step_idx → [(entity_id, available), ...]
        fault_by_step: dict[int, list] = {}
        for step_idx, eid, available in cfg.fault_schedule:
            fault_by_step.setdefault(step_idx, []).append((eid, available))

        # Build window map: step_idx → is_open
        window_by_step: dict[int, bool] = {}
        for step_idx, is_open in cfg.window_schedule:
            window_by_step[step_idx] = is_open

        with self._clock:
            for step_idx in range(cfg.n_steps):
                await self._step(step_idx, fault_by_step, window_by_step)

        # Final flush so FakeStore.data is non-None after any run (including no-restart runs).
        # _total_saves is NOT updated here; callers use _total_saves + _fake_store.saves.
        await self._shadow._runtime.async_flush()

        return self._metrics.summarize()

    async def _step(self, step_idx: int, fault_by_step: dict, window_by_step: dict = None) -> None:
        cfg = self.cfg
        _t0 = _time.perf_counter()

        # 1. Advance simulation time
        self._clock.advance(cfg.step_s)
        sim_time = self._clock.now_utc()

        # 2. Apply window schedule
        if window_by_step and step_idx in window_by_step:
            self._room.set_window_open(window_by_step[step_idx])

        # 3. Apply fault injections
        for eid, available in fault_by_step.get(step_idx, []):
            new_state = "heat" if available else "unavailable"
            self._trv_cmds[eid]["_state"] = new_state
            self._trv_cmds[eid]["hvac_mode"] = new_state

        # 4. Update sensor reading from current room temperature
        noisy_reading = self._room.read_sensor(noise=True)
        self._sensor_value[0] = noisy_reading
        self._metrics._summary.sensor_readings_sum += noisy_reading

        # 5. Run real coordinator cycle
        pre_count = len(self._coord._service_calls)
        result = await self._coord._async_update_data()
        zone = (result or {}).get("zone", {})

        # 6. Apply dispatched setpoints to TRV state and physics
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

        # 7. Advance room physics (count window-open contribution)
        outdoor = cfg.outdoor_temp_fn(sim_time)
        solar = cfg.solar_gain_fn(sim_time)
        if self._room.window_open:
            self._metrics._summary.window_open_steps += 1
        self._room.step(cfg.step_s, outdoor, solar)

        # 8. Record metrics
        elapsed_ms = (_time.perf_counter() - _t0) * 1000.0
        self._metrics.record(
            sim_time, zone, self._room, outdoor, cfg.step_s,
            service_calls_this_step=len(new_calls),
        )
        self._metrics._summary.perf.add(elapsed_ms)

        # 9. Model snapshot at configured interval
        interval = cfg.model_snapshot_interval_steps
        if interval > 0 and step_idx > 0 and step_idx % interval == 0:
            day_idx = len(self._metrics._summary.daily) - 1
            snap = self._extract_model_snapshot(day_idx=day_idx, step_idx=step_idx)
            self._metrics._summary.model_snapshots.append(snap)

        # 10. Simulate HA restart
        if step_idx in cfg.restart_steps:
            await self._do_restart()
            self._metrics._summary.restart_count += 1

    def _extract_model_snapshot(self, day_idx: int, step_idx: int) -> ModelLearningSnapshot:
        """Extract current learned model state from the shadow controller.

        Accesses RuntimeHealth for high-level stats and model-specific _state via
        defensive attribute access. All values are Optional — missing attributes
        return None without raising.
        """
        snap = ModelLearningSnapshot(
            day_idx=day_idx,
            step_idx=step_idx,
            gt_heat_rate_c_per_h=self._room.ground_truth_heat_rate_c_per_h,
            gt_heat_loss_w_per_k=self._room.ground_truth_heat_loss_w_per_k,
            gt_supply_delay_s=self._room.ground_truth_supply_delay_s,
            gt_afterheat_tau_s=self._room.ground_truth_afterheat_tau_s,
        )

        # Store stats — cumulative saves across all fake stores (restarts reset per-store counter)
        snap.save_count = self._total_saves + self._fake_store.saves
        try:
            blob = json.dumps(self._fake_store.data or {})
            snap.blob_bytes = len(blob.encode("utf-8"))
        except Exception:
            snap.blob_bytes = 0

        # RuntimeHealth for high-level counters
        try:
            health = self._shadow._runtime.health()
            snap.model_update_total = health.model_update_total
            counts = dict(health.model_update_counts)
            snap.model_update_counts = counts
            snap.control_fallback = health.control_fallback
            snap.control_applied = health.control_applied
            # Per-model breakdown — None-safe; keys absent when model never updated
            snap.heat_rate_update_count = counts.get("heat_rate", 0)
            snap.heat_loss_update_count = counts.get("heat_loss", 0)
            snap.onset_delay_update_count = counts.get("onset_delay", 0)
            snap.afterheat_update_count = counts.get("afterheat", 0)
            snap.outcome_update_count = counts.get("outcome", 0)
        except Exception:
            pass

        # Model-specific learned values — defensive attribute chain access
        try:
            zones = self._shadow._runtime._zones
            zone_key = next(iter(zones), None)
            if zone_key:
                zone_rt = zones[zone_key]
                models = zone_rt.orchestrator.models if hasattr(zone_rt, "orchestrator") else {}

                def _safe_get(model_name, *attrs):
                    try:
                        obj = models.get(model_name)
                        if obj is None:
                            return None
                        s = obj._state
                        for a in attrs:
                            s = getattr(s, a)
                        return float(s) if s is not None else None
                    except Exception:
                        return None

                snap.learned_heat_rate_c_per_h = _safe_get(
                    "heat_rate", "general", "rate_c_per_h")
                snap.learned_heat_loss_w_per_k = _safe_get(
                    "heat_loss", "general", "rate_c_per_h")
                # onset_delay_min → convert to seconds for ground-truth comparison
                onset_min = _safe_get("onset_delay", "general", "onset_delay_min")
                snap.learned_supply_delay_s = (
                    onset_min * 60.0 if onset_min is not None else None
                )
                snap.learned_afterheat_rise_c = _safe_get(
                    "afterheat", "general", "rise")
                snap.learned_boost_offset_c = _safe_get(
                    "boost", "general", "effective_factor", "value")
        except Exception:
            pass

        return snap

    async def _do_restart(self) -> None:
        """Flush LE2 state, rebuild coordinator from saved store data."""
        # Force save current LE2 state
        await self._shadow._runtime.async_flush()
        # Accumulate saves from the current store before replacing it
        self._total_saves += self._fake_store.saves
        saved = dict(self._fake_store.data or {})

        # Rebuild with restored state
        new_store = FakeStore(data=saved)
        new_coord, _ = self._build_coord()
        new_coord._service_calls = new_coord._service_calls  # already set in _build_coord

        cfg = self.cfg
        if cfg.baseline_mode:
            _shadow_mode = LearningRuntimeMode.SHADOW
        elif getattr(cfg, "active_control", True):
            _shadow_mode = LearningRuntimeMode.CONTROL
        else:
            _shadow_mode = LearningRuntimeMode.SHADOW

        new_shadow = attach_shadow(new_coord, store=new_store, mode=_shadow_mode)
        await new_shadow.async_setup()

        self._coord = new_coord
        self._fake_store = new_store
        self._shadow = new_shadow
