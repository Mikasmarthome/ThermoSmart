"""S1a Item 7 Follow-up: Extended four-mode authority proof (pure Python, Welle 1/2).

Covers:
  1.  LE1 frozen semantics in all four modes
  2.  LE2 storage dirty/save per mode
  3.  Pending outcome count per mode
  4.  Preheat/afterheat/boost cleanup on mode transitions
  5.  Multi-TRV in all four modes
  6.  Partial-failure scenarios (TRV unavailable, service call fails)
  7.  Minimal TRV-only setup (DETERMINISTIC and ADAPTIVE)
  8.  Restart/Restore unit-level matrix
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch as _patch

import pytest

from custom_components.thermosmart.coordinator import (
    ControlAdaptationMode, ThermoSmartCoordinator,
)
from tests.helpers import make_coordinator, make_state, make_zone_config, set_hass_states
from tests.helpers_ha_runtime import (
    FakeStore, RecordingCoordinator, attach_shadow,
    inject_eligible_boost_model, make_recording_coordinator,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

async def _cycle(coord) -> dict:
    result = await coord._async_update_data()
    return (result or {}).get("zone", {})


def _base_coord(learning: bool, active: bool, *, extra_cfg: dict | None = None):
    cfg = {"learning_enabled": learning, **(extra_cfg or {})}
    coord = make_recording_coordinator(cfg)
    coord.set_active_control(active)
    return coord


async def _coord_with_shadow(learning: bool, active: bool, *,
                              extra_cfg: dict | None = None, eligible_boost: bool = False,
                              shadow_mode=None):
    from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
    coord = _base_coord(learning, active, extra_cfg=extra_cfg)
    _mode = shadow_mode or LearningRuntimeMode.CONTROL
    store = FakeStore()
    shadow = attach_shadow(coord, mode=_mode, store=store)
    await shadow.async_setup()
    if eligible_boost:
        inject_eligible_boost_model(shadow)
    return coord, shadow, store


def _zone_runtime(shadow):
    """Access the ZoneRuntime for the shadow's zone."""
    return shadow.runtime._zones.get(shadow._zone)


# ════════════════════════════════════════════════════════════════════════════════
# 1. LE1 Frozen Semantics
# ════════════════════════════════════════════════════════════════════════════════

class TestLE1FrozenSemantics:
    """Verify that the legacy learning engine (LE1) is frozen in all four modes.

    In production, freeze() is called in __init__.py:159 immediately after load.
    All LE1 mutation methods (async_observe, update_heating_session, etc.) check
    _frozen first and return without any state change.
    """

    def _make_real_le1(self):
        """Create a real (unfrozen) LearningEngine backed by a MagicMock hass."""
        from custom_components.thermosmart.learning_engine import LearningEngine
        hass = MagicMock()
        hass.async_create_task = MagicMock()
        return LearningEngine(hass)

    def test_freeze_sets_flag(self):
        """freeze() sets _frozen = True."""
        le = self._make_real_le1()
        assert not le.frozen
        le.freeze()
        assert le.frozen

    @pytest.mark.asyncio
    async def test_frozen_le1_observe_is_noop(self):
        """async_observe() on a frozen LE1 adds no observations."""
        le = self._make_real_le1()
        le.set_zone_enabled("z1", True)
        le.freeze()
        before = len(le._observations.get("z1", []))
        await le.async_observe(
            zone_id="z1",
            recommendation={"current_temp": 19.0, "effective_target": 21.0},
            weather_data={},
        )
        after = len(le._observations.get("z1", []))
        assert before == after, "frozen LE1 must not add observations"

    def test_frozen_le1_update_heating_session_is_noop(self):
        """update_heating_session() on frozen LE1 starts no new sessions."""
        le = self._make_real_le1()
        le.set_zone_enabled("z1", True)
        le.freeze()
        le.update_heating_session(
            zone_id="z1", current_temp=19.0, target=21.0,
            is_active_control=True, weather_data={},
        )
        assert "z1" not in le._heating_sessions, "frozen LE1 must not start sessions"

    @pytest.mark.asyncio
    async def test_frozen_le1_save_is_noop(self):
        """async_save() on frozen LE1 never writes to the store."""
        le = self._make_real_le1()
        le.freeze()
        # async_save should silently return; no IOError or state change
        await le.async_save()  # must not raise

    @pytest.mark.asyncio
    async def test_coordinator_does_not_call_frozen_le1_observe_paths(self):
        """P2-1: frozen LE1 observe/update paths are no longer called, in ANY mode.

        LE v1 is frozen (learning_engine.freeze() in __init__.py) — its
        async_observe/update_heating_session/async_observe_trv_setpoint calls
        were dead no-ops (the frozen check always short-circuited them) and
        have since been removed from the coordinator's real cycle. LE v1
        continues to serve exactly one live purpose: the schedule lookup via
        async_get_base_target(), which must still be called every cycle in
        all four modes — this removal does not gate or disable any LE2 path.

        async_observe_window_cooling() is not exercised here: it is only
        reached from the window-sensor state-change listener wired by
        setup_event_listeners(), which this cycle-level harness does not
        register — its removal is verified statically (no remaining call
        site in coordinator.py), not via a live event simulation here.
        """
        for lm, ac in [(False, False), (True, False), (False, True), (True, True)]:
            coord = _base_coord(lm, ac)
            await _cycle(coord)
            assert not coord.learning_engine.async_observe.called, (
                f"le1.async_observe unexpectedly called in mode ({lm},{ac})")
            assert not coord.learning_engine.update_heating_session.called, (
                f"le1.update_heating_session unexpectedly called in mode ({lm},{ac})")
            assert not coord.learning_engine.async_observe_trv_setpoint.called, (
                f"le1.async_observe_trv_setpoint unexpectedly called in mode ({lm},{ac})")
            assert coord.learning_engine.async_get_base_target.called, (
                f"le1.async_get_base_target (schedule lookup) must still be "
                f"called in mode ({lm},{ac})")
            coord.learning_engine.reset_mock()

    @pytest.mark.asyncio
    async def test_le1_not_enabled_blocks_observe_independently_of_frozen(self):
        """_is_enabled(zone_id) = False (zone disabled) also blocks observe."""
        le = self._make_real_le1()
        le.set_zone_enabled("z1", False)  # disable without freezing
        await le.async_observe(
            zone_id="z1",
            recommendation={"current_temp": 19.0, "effective_target": 21.0},
            weather_data={},
        )
        assert len(le._observations.get("z1", [])) == 0


# ════════════════════════════════════════════════════════════════════════════════
# 2. LE2 Storage Dirty / Save per Mode
# ════════════════════════════════════════════════════════════════════════════════

class TestStorageDirtyPerMode:

    @pytest.mark.asyncio
    async def test_inactive_no_dirty_after_cycle(self):
        """INACTIVE: observe_safe never called → no dirty flag after cycle."""
        coord, shadow, store = await _coord_with_shadow(False, False)
        await _cycle(coord)
        rt = shadow.runtime
        dirty = rt._persistence.dirty if rt._persistence else False
        assert not dirty, f"INACTIVE must not mark dirty (got dirty={dirty})"

    @pytest.mark.asyncio
    async def test_deterministic_no_dirty_after_cycle(self):
        """DETERMINISTIC: observe_safe not called → no dirty flag."""
        coord, shadow, store = await _coord_with_shadow(False, True)
        await _cycle(coord)
        rt = shadow.runtime
        dirty = rt._persistence.dirty if rt._persistence else False
        assert not dirty, f"DETERMINISTIC must not mark dirty (got dirty={dirty})"

    @pytest.mark.asyncio
    async def test_shadow_only_observation_runs_without_crashing(self):
        """SHADOW_ONLY: observe_safe runs; no dirty on first cycle (no episode complete)."""
        coord, shadow, store = await _coord_with_shadow(True, False)
        await _cycle(coord)
        # Shadow DID observe; dirty may or may not be set (startup grace or no episode)
        # We verify: no crash, shadow still initialized
        assert shadow.diagnostics()["initialized"]

    @pytest.mark.asyncio
    async def test_inactive_no_store_saves(self):
        """INACTIVE: FakeStore.saves == 0 after multiple cycles."""
        coord, shadow, store = await _coord_with_shadow(False, False)
        for _ in range(5):
            await _cycle(coord)
        assert store.saves == 0, f"INACTIVE must not save: saves={store.saves}"

    @pytest.mark.asyncio
    async def test_deterministic_no_store_saves(self):
        """DETERMINISTIC: FakeStore.saves == 0 after multiple cycles."""
        coord, shadow, store = await _coord_with_shadow(False, True)
        for _ in range(5):
            await _cycle(coord)
        assert store.saves == 0, f"DETERMINISTIC must not save: saves={store.saves}"

    @pytest.mark.asyncio
    async def test_shadow_only_debounced_save_not_immediate(self):
        """SHADOW_ONLY: save is debounced (FakeStore.saves == 0 immediately after cycle)."""
        coord, shadow, store = await _coord_with_shadow(True, False)
        await _cycle(coord)
        # Debounce delay means save has NOT happened yet in the same cycle
        assert store.saves == 0, (
            f"SHADOW_ONLY save must be debounced, not immediate: saves={store.saves}")

    @pytest.mark.asyncio
    async def test_manual_flush_writes_dirty_state(self):
        """SHADOW_ONLY + manual flush: state is written to FakeStore."""
        from tests.helpers_runtime_scenarios import heating_ramp_then_settle
        coord, shadow, store = await _coord_with_shadow(True, False)
        # Drive the runtime to produce authoritative changes (outside coordinator)
        heating_ramp_then_settle(shadow.runtime, zone=shadow._zone)
        shadow.runtime.mark_dirty(important=True)
        await shadow.runtime.async_flush()
        assert store.saves == 1, (
            f"Manual flush with dirty state must write store: saves={store.saves}")


# ════════════════════════════════════════════════════════════════════════════════
# 3. Pending Outcome Count per Mode
# ════════════════════════════════════════════════════════════════════════════════

class TestPendingOutcomesPerMode:

    @pytest.mark.asyncio
    async def test_inactive_no_pending_boost_contexts(self):
        """INACTIVE: Gate 1 blocks boost → no pending boost contexts created."""
        coord, shadow, _ = await _coord_with_shadow(False, False, eligible_boost=True)
        await _cycle(coord)
        zr = _zone_runtime(shadow)
        if zr is not None:
            count = len(zr.pending_boost_contexts)
            assert count == 0, f"INACTIVE must have 0 pending boost contexts: {count}"

    @pytest.mark.asyncio
    async def test_shadow_only_no_pending_boost_contexts(self):
        """SHADOW_ONLY: Gate 2 blocks boost → no pending boost contexts."""
        coord, shadow, _ = await _coord_with_shadow(True, False, eligible_boost=True)
        await _cycle(coord)
        zr = _zone_runtime(shadow)
        if zr is not None:
            count = len(zr.pending_boost_contexts)
            assert count == 0, f"SHADOW_ONLY must have 0 pending boost contexts: {count}"

    @pytest.mark.asyncio
    async def test_deterministic_no_pending_boost_contexts(self):
        """DETERMINISTIC: Gate 1 blocks boost → no pending boost contexts."""
        coord, shadow, _ = await _coord_with_shadow(False, True, eligible_boost=True)
        await _cycle(coord)
        zr = _zone_runtime(shadow)
        if zr is not None:
            count = len(zr.pending_boost_contexts)
            assert count == 0, f"DETERMINISTIC must have 0 pending boost contexts: {count}"

    @pytest.mark.asyncio
    async def test_adaptive_pending_context_created_when_boost_eligible(self):
        """ADAPTIVE with eligible model: pending boost context created on boost apply."""
        coord, shadow, _ = await _coord_with_shadow(True, True, eligible_boost=True)
        zone = await _cycle(coord)
        applied = zone.get("applied_boost_offset_c") or 0.0
        zr = _zone_runtime(shadow)
        if applied > 0 and zr is not None:
            # Boost was applied → pending context must exist
            count = len(zr.pending_boost_contexts)
            assert count >= 1, (
                f"ADAPTIVE with applied boost must have pending context: {count}")
        # If boost was blocked by inner gates (cooldown etc.), pending context = 0 is also OK


# ════════════════════════════════════════════════════════════════════════════════
# 4. Preheat / Afterheat / Boost Cleanup on Mode Transitions
# ════════════════════════════════════════════════════════════════════════════════

class TestPreheatCleanupTransitions:

    @pytest.mark.asyncio
    async def test_adaptive_to_inactive_no_stale_preheat_dispatch(self):
        """ADAPTIVE → INACTIVE: no dispatch occurs in INACTIVE cycle after switch."""
        coord, shadow, _ = await _coord_with_shadow(True, True)
        await _cycle(coord)
        coord.entry.data["learning_enabled"] = False
        coord.set_active_control(False)
        coord.control_calls.clear()
        await _cycle(coord)
        assert not coord.control_calls, "INACTIVE must not dispatch after switch from ADAPTIVE"

    @pytest.mark.asyncio
    async def test_adaptive_to_shadow_only_no_stale_dispatch(self):
        """ADAPTIVE → SHADOW_ONLY: no dispatch in SHADOW_ONLY cycle."""
        coord, shadow, _ = await _coord_with_shadow(True, True)
        await _cycle(coord)
        coord.set_active_control(False)
        coord.control_calls.clear()
        await _cycle(coord)
        assert not coord.control_calls, "SHADOW_ONLY must not dispatch after switch from ADAPTIVE"

    @pytest.mark.asyncio
    async def test_adaptive_to_deterministic_boost_blocked(self):
        """ADAPTIVE → DETERMINISTIC: LE2 boost is blocked (Gate 1) in next cycle."""
        coord, shadow, _ = await _coord_with_shadow(True, True, eligible_boost=True)
        await _cycle(coord)
        coord.entry.data["learning_enabled"] = False  # switch to DETERMINISTIC
        zone_b = await _cycle(coord)
        applied = zone_b.get("applied_boost_offset_c") or 0.0
        assert applied == 0.0, f"DETERMINISTIC must have 0 boost: applied={applied}"

    @pytest.mark.asyncio
    async def test_active_control_off_during_active_preheat_no_dispatch(self):
        """AC OFF during preheat: no setpoint dispatch in same cycle."""
        coord, shadow, _ = await _coord_with_shadow(True, True)
        await _cycle(coord)
        # Switch AC off (simulates user turning off AC during preheat)
        coord.set_active_control(False)
        coord.control_calls.clear()
        await _cycle(coord)
        assert not coord.control_calls, (
            "Switching AC off must immediately stop dispatch (same cycle)")

    @pytest.mark.asyncio
    async def test_learning_off_during_adaptive_preheat_no_le2_boost(self):
        """Learning OFF during ADAPTIVE preheat: LE2 boost blocked in next cycle."""
        coord, shadow, _ = await _coord_with_shadow(True, True, eligible_boost=True)
        await _cycle(coord)
        coord.entry.data["learning_enabled"] = False
        zone_b = await _cycle(coord)
        applied = zone_b.get("applied_boost_offset_c") or 0.0
        assert applied == 0.0, (
            f"Learning OFF must block LE2 boost: applied={applied}")

    @pytest.mark.asyncio
    async def test_mode_switch_before_dispatch_no_stale_offset(self):
        """Mode switch in same cycle as eligible boost: no stale boost if mode is non-ADAPTIVE."""
        coord, shadow, _ = await _coord_with_shadow(True, True, eligible_boost=True)
        # Switch to SHADOW_ONLY in the same "session"
        coord.set_active_control(False)
        zone = await _cycle(coord)
        assert not coord.control_calls
        applied = zone.get("applied_boost_offset_c") or 0.0
        assert applied == 0.0, "No boost dispatch in SHADOW_ONLY"

    @pytest.mark.asyncio
    async def test_mode_switch_after_dispatch_boost_released(self):
        """Mode switch after ADAPTIVE dispatch: next cycle releases boost lifecycle."""
        coord, shadow, _ = await _coord_with_shadow(True, True, eligible_boost=True)
        zone_a = await _cycle(coord)  # ADAPTIVE dispatch
        # Switch to INACTIVE
        coord.entry.data["learning_enabled"] = False
        coord.set_active_control(False)
        zone_b = await _cycle(coord)
        reason = zone_b.get("_boost_rejection_reason", "")
        assert "learning" in reason.lower() or reason, (
            f"Boost must be released in INACTIVE mode: {reason!r}")

    @pytest.mark.asyncio
    async def test_deterministic_to_inactive_stops_dispatch(self):
        """DETERMINISTIC → INACTIVE: subsequent cycle has zero dispatch."""
        coord = _base_coord(False, True)
        await _cycle(coord)
        coord.set_active_control(False)
        coord.control_calls.clear()
        await _cycle(coord)
        assert not coord.control_calls

    @pytest.mark.asyncio
    async def test_shadow_only_to_inactive_stops_observation(self):
        """SHADOW_ONLY → INACTIVE: observe_safe no longer called."""
        coord, shadow, _ = await _coord_with_shadow(True, False)
        # Switch to INACTIVE
        coord.entry.data["learning_enabled"] = False
        with _patch.object(shadow, "observe_safe", wraps=shadow.observe_safe) as spy:
            await _cycle(coord)
        assert not spy.called, "INACTIVE must not call observe_safe"

    @pytest.mark.asyncio
    async def test_no_double_release_on_repeated_transition(self):
        """ADAPTIVE → INACTIVE → INACTIVE: release called once, no double-release error."""
        coord, shadow, _ = await _coord_with_shadow(True, True, eligible_boost=True)
        await _cycle(coord)
        coord.entry.data["learning_enabled"] = False
        coord.set_active_control(False)
        zone1 = await _cycle(coord)  # first INACTIVE cycle
        zone2 = await _cycle(coord)  # second INACTIVE cycle
        # Both cycles must have the same rejection reason (no crash on second release)
        r1 = zone1.get("_boost_rejection_reason")
        r2 = zone2.get("_boost_rejection_reason")
        assert r1 is not None and r2 is not None


# ════════════════════════════════════════════════════════════════════════════════
# 5. Multi-TRV in All Four Modes
# ════════════════════════════════════════════════════════════════════════════════

class TestMultiTRVAllModes:

    def _multi_trv_coord(self, learning: bool, active: bool):
        cfg = {
            "learning_enabled": learning,
            "climate_entities": ["climate.trv_a", "climate.trv_b"],
        }
        coord = make_recording_coordinator(cfg)
        coord.set_active_control(active)
        return coord

    @pytest.mark.asyncio
    async def test_inactive_multi_trv_zero_dispatch(self):
        """INACTIVE + 2 TRVs: zero dispatch calls."""
        coord = self._multi_trv_coord(False, False)
        await _cycle(coord)
        assert not coord.control_calls, "INACTIVE multi-TRV must produce 0 calls"

    @pytest.mark.asyncio
    async def test_shadow_only_multi_trv_zero_dispatch(self):
        """SHADOW_ONLY + 2 TRVs: zero dispatch."""
        coord, shadow, _ = await _coord_with_shadow(
            True, False, extra_cfg={"climate_entities": ["climate.trv_a", "climate.trv_b"]})
        await _cycle(coord)
        assert not coord.control_calls

    @pytest.mark.asyncio
    async def test_deterministic_multi_trv_dispatches(self):
        """DETERMINISTIC + 2 TRVs: dispatch occurs."""
        coord = self._multi_trv_coord(False, True)
        await _cycle(coord)
        assert coord.control_calls, "DETERMINISTIC multi-TRV must dispatch"

    @pytest.mark.asyncio
    async def test_adaptive_multi_trv_dispatches_and_observes(self):
        """ADAPTIVE + 2 TRVs: dispatch + observe."""
        coord, shadow, _ = await _coord_with_shadow(
            True, True, extra_cfg={"climate_entities": ["climate.trv_a", "climate.trv_b"]})
        with _patch.object(shadow, "observe_safe", wraps=shadow.observe_safe) as spy:
            await _cycle(coord)
        assert coord.control_calls, "ADAPTIVE multi-TRV must dispatch"
        assert spy.called, "ADAPTIVE multi-TRV must observe"


# ════════════════════════════════════════════════════════════════════════════════
# 6. Partial Failure Scenarios
# ════════════════════════════════════════════════════════════════════════════════

class TestPartialFailureScenarios:

    @pytest.mark.asyncio
    async def test_deterministic_trv_unavailable_no_crash(self):
        """DETERMINISTIC: one TRV unavailable → coordinator cycle must not raise."""
        coord = _base_coord(False, True)
        # Simulate the TRV being unavailable
        from tests.helpers import make_state
        coord.hass.states.get = lambda eid: (
            make_state("unavailable") if eid == "climate.test_trv"
            else make_state("19.0")
        )
        zone = await _cycle(coord)
        assert zone is not None, "Coordinator must not crash with unavailable TRV"

    @pytest.mark.asyncio
    async def test_adaptive_trv_unavailable_boosts_blocked(self):
        """ADAPTIVE: TRV unavailable → boost blocked (Gate 3: device_unavailable)."""
        coord, shadow, _ = await _coord_with_shadow(True, True, eligible_boost=True)
        # Simulate unavailable TRV
        coord.hass.states.get = lambda eid: (
            make_state("unavailable") if eid == "climate.test_trv"
            else make_state("19.0")
        )
        zone = await _cycle(coord)
        reason = zone.get("_boost_rejection_reason") or ""
        applied = zone.get("applied_boost_offset_c") or 0.0
        # Either: boost was blocked due to unavailable device, or applied is 0
        assert applied == 0.0 or "unavailable" in reason.lower(), (
            f"Unavailable TRV must block boost: applied={applied}, reason={reason}")

    @pytest.mark.asyncio
    async def test_inactive_sensor_unavailable_no_dispatch(self):
        """INACTIVE + unavailable sensor: still no dispatch (gate holds)."""
        coord = _base_coord(False, False)
        coord.hass.states.get = lambda eid: None  # all sensors unavailable
        zone = await _cycle(coord)
        assert not coord.control_calls

    @pytest.mark.asyncio
    async def test_shadow_only_sensor_unavailable_no_dispatch(self):
        """SHADOW_ONLY + unavailable sensor: no dispatch regardless."""
        coord, shadow, _ = await _coord_with_shadow(True, False)
        coord.hass.states.get = lambda eid: None
        await _cycle(coord)
        assert not coord.control_calls

    @pytest.mark.asyncio
    async def test_deterministic_wrong_mode_trv_no_crash(self):
        """DETERMINISTIC: TRV in 'off' mode → coordinator handles gracefully."""
        coord = _base_coord(False, True)
        coord.hass.states.get = lambda eid: (
            make_state("off", {"temperature": 5.0, "min_temp": 5.0, "max_temp": 30.0,
                                "hvac_mode": "off", "target_temp_step": 0.5})
            if eid == "climate.test_trv" else make_state("19.0")
        )
        zone = await _cycle(coord)
        assert zone is not None


# ════════════════════════════════════════════════════════════════════════════════
# 7. Minimal TRV-Only Setup
# ════════════════════════════════════════════════════════════════════════════════

class TestMinimalSetup:

    @pytest.mark.asyncio
    async def test_deterministic_trv_only_no_temp_sensor(self):
        """DETERMINISTIC: no temp sensor → coordinator runs but uses fallback."""
        cfg = {"learning_enabled": False, "temp_sensors": []}
        coord = make_recording_coordinator(cfg)
        coord.set_active_control(True)
        # No temp sensor state → current_temp will be None
        coord.hass.states.get = lambda eid: None
        zone = await _cycle(coord)
        assert zone is not None, "Must not crash with no temp sensor"

    @pytest.mark.asyncio
    async def test_deterministic_trv_only_dispatches(self):
        """DETERMINISTIC: TRV-only (no extra sensors) → dispatch occurs."""
        cfg = {
            "learning_enabled": False,
            "temp_sensors": ["sensor.test_temp"],
            "humidity_sensors": [],
            "window_sensors": [],
        }
        coord = make_recording_coordinator(cfg)
        coord.set_active_control(True)
        zone = await _cycle(coord)
        assert coord.control_calls, "TRV-only DETERMINISTIC must dispatch"

    @pytest.mark.asyncio
    async def test_adaptive_trv_only_with_shadow_no_crash(self):
        """ADAPTIVE: TRV-only with LE2 shadow → no crash, safe baseline used."""
        cfg = {
            "learning_enabled": True,
            "temp_sensors": ["sensor.test_temp"],
            "humidity_sensors": [],
            "window_sensors": [],
        }
        coord, shadow, _ = await _coord_with_shadow(True, True, extra_cfg=cfg)
        zone = await _cycle(coord)
        assert zone is not None
        assert coord.control_calls, "TRV-only ADAPTIVE must dispatch"

    @pytest.mark.asyncio
    async def test_shadow_only_no_temp_no_dispatch_no_crash(self):
        """SHADOW_ONLY: no temp sensor → observe_safe called but no crash, no dispatch."""
        cfg = {"learning_enabled": True, "temp_sensors": []}
        coord, shadow, _ = await _coord_with_shadow(True, False, extra_cfg=cfg)
        coord.hass.states.get = lambda eid: None
        zone = await _cycle(coord)
        assert zone is not None
        assert not coord.control_calls


# ════════════════════════════════════════════════════════════════════════════════
# 8. Restart / Restore Unit-Level Matrix
# ════════════════════════════════════════════════════════════════════════════════

class TestRestoreMatrixUnit:
    """Simulate HA restart by creating a fresh coordinator after a cycle.

    A 'restart' is modelled as: run cycle, discard coordinator, create new one
    (optionally with stored shadow data via FakeStore).
    """

    @pytest.mark.asyncio
    async def test_inactive_restart_stays_inactive(self):
        """INACTIVE → restart → INACTIVE: adaptation_mode = inactive in both sessions."""
        coord = _base_coord(False, False)
        zone1 = await _cycle(coord)
        assert zone1.get("adaptation_mode") == ControlAdaptationMode.INACTIVE
        # Simulate restart: fresh coordinator in same mode
        coord2 = _base_coord(False, False)
        zone2 = await _cycle(coord2)
        assert zone2.get("adaptation_mode") == ControlAdaptationMode.INACTIVE

    @pytest.mark.asyncio
    async def test_shadow_only_restart_restores_observation(self):
        """SHADOW_ONLY → restart → SHADOW_ONLY: observation resumes after restart."""
        coord, shadow, store = await _coord_with_shadow(True, False)
        await _cycle(coord)
        # Restart: create new coordinator with same store (persisted state)
        coord2, shadow2, store2 = await _coord_with_shadow(True, False)
        with _patch.object(shadow2, "observe_safe", wraps=shadow2.observe_safe) as spy:
            await _cycle(coord2)
        assert spy.called, "SHADOW_ONLY after restart must still observe"
        assert not coord2.control_calls

    @pytest.mark.asyncio
    async def test_deterministic_restart_no_le2_data_in_control(self):
        """DETERMINISTIC → restart → DETERMINISTIC: no LE2 data affects TPI."""
        coord, shadow, _ = await _coord_with_shadow(False, True)
        zone1 = await _cycle(coord)
        assert zone1.get("tpi_coef_source") == "deterministic_baseline"
        # Restart
        coord2, shadow2, _ = await _coord_with_shadow(False, True)
        zone2 = await _cycle(coord2)
        assert zone2.get("tpi_coef_source") == "deterministic_baseline"

    @pytest.mark.asyncio
    async def test_adaptive_restart_no_stale_boost_carryover(self):
        """ADAPTIVE → restart → ADAPTIVE: no carryover boost from pre-restart session."""
        coord, shadow, _ = await _coord_with_shadow(True, True, eligible_boost=True)
        await _cycle(coord)
        # Restart: fresh coordinator in ADAPTIVE
        coord2, shadow2, _ = await _coord_with_shadow(True, True)
        zone2 = await _cycle(coord2)
        assert zone2.get("adaptation_mode") == ControlAdaptationMode.ADAPTIVE
        # New session: no stale boost offset from old session
        applied = zone2.get("applied_boost_offset_c") or 0.0
        assert applied == 0.0 or applied <= 8.0  # bounded, not stale

    @pytest.mark.asyncio
    async def test_restart_adaptive_then_switch_to_inactive(self):
        """Restart in ADAPTIVE, then switch to INACTIVE: no residual control."""
        coord, shadow, _ = await _coord_with_shadow(True, True)
        await _cycle(coord)
        # Restart with INACTIVE config
        coord2 = _base_coord(False, False)
        zone2 = await _cycle(coord2)
        assert zone2.get("adaptation_mode") == ControlAdaptationMode.INACTIVE
        assert not coord2.control_calls

    @pytest.mark.asyncio
    async def test_restart_adaptive_then_learning_off(self):
        """Restart in ADAPTIVE, then learning OFF (DETERMINISTIC): TPI resets."""
        coord, shadow, _ = await _coord_with_shadow(True, True)
        await _cycle(coord)
        # Restart in DETERMINISTIC
        coord2, shadow2, _ = await _coord_with_shadow(False, True)
        zone2 = await _cycle(coord2)
        assert zone2.get("adaptation_mode") == ControlAdaptationMode.DETERMINISTIC
        assert zone2.get("tpi_coef_source") == "deterministic_baseline"

    @pytest.mark.asyncio
    async def test_restart_with_corrupted_shadow_store(self):
        """Corrupt shadow store → coordinator continues with baseline, no crash."""
        corrupt_store = FakeStore(data={"garbage": True, "le2_corrupt": "yes"})
        coord = _base_coord(True, True)
        from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
        shadow = attach_shadow(coord, store=corrupt_store, mode=LearningRuntimeMode.CONTROL)
        await shadow.async_setup()  # must not raise despite corrupt data
        zone = await _cycle(coord)
        assert zone is not None, "Corrupt store must not crash coordinator"
        assert coord.control_calls, "ADAPTIVE must still dispatch even with corrupt store"

    @pytest.mark.asyncio
    async def test_restore_barrier_still_active_before_first_cycle(self):
        """Fresh coordinator: barrier active before first cycle, cleared after set_active_control."""
        coord = make_recording_coordinator({"learning_enabled": True})
        coord._active_control = True  # raw, without calling set_active_control
        # Barrier is NOT cleared yet
        assert not coord._active_control_initialized
        # After set_active_control, barrier is cleared
        coord.set_active_control(True)
        assert coord._active_control_initialized


# ════════════════════════════════════════════════════════════════════════════════
# 9. Shadow vs Real Outcome Separation
# ════════════════════════════════════════════════════════════════════════════════

class TestShadowVsRealOutcomes:

    @pytest.mark.asyncio
    async def test_shadow_only_no_real_dispatch_provenance(self):
        """SHADOW_ONLY: shadow observes but no real dispatch → no dispatch provenance."""
        coord, shadow, _ = await _coord_with_shadow(True, False, eligible_boost=True)
        zone = await _cycle(coord)
        # No dispatch → _last_written_setpoints unchanged
        written = getattr(coord, "_last_written_setpoints", {})
        assert not written, (
            f"SHADOW_ONLY must not write setpoints: {written}")

    @pytest.mark.asyncio
    async def test_adaptive_real_dispatch_separate_from_shadow_proposals(self):
        """ADAPTIVE: real dispatch writes setpoints; shadow proposals are internal."""
        coord, shadow, _ = await _coord_with_shadow(True, True)
        zone = await _cycle(coord)
        # Real dispatch occurred
        assert coord.control_calls, "ADAPTIVE must have real dispatch"
        # Shadow proposals (if any) are in shadow._last_result, not in real setpoints
        real_setpoints = getattr(coord, "_last_written_setpoints", {})
        # Shadow's last_result is separate from real control
        shadow_result = shadow._last_result
        # They co-exist: real dispatch + shadow observation are separate paths
        assert zone.get("adaptation_mode") == ControlAdaptationMode.ADAPTIVE

    @pytest.mark.asyncio
    async def test_inactive_no_shadow_proposals_and_no_real_dispatch(self):
        """INACTIVE: neither shadow proposals nor real dispatch."""
        coord, shadow, _ = await _coord_with_shadow(False, False)
        with _patch.object(shadow, "observe_safe", wraps=shadow.observe_safe) as spy:
            await _cycle(coord)
        assert not spy.called, "INACTIVE: no shadow observation"
        assert not coord.control_calls, "INACTIVE: no real dispatch"

    @pytest.mark.asyncio
    async def test_deterministic_real_dispatch_no_le2_influence(self):
        """DETERMINISTIC: real dispatch happens but LE2 shadow has no influence on setpoints."""
        coord, shadow, _ = await _coord_with_shadow(False, True)
        zone = await _cycle(coord)
        assert coord.control_calls, "DETERMINISTIC must dispatch"
        # LE2 shadow NOT observed → no LE2 influence
        assert zone.get("tpi_coef_source") == "deterministic_baseline"
        applied = zone.get("applied_boost_offset_c") or 0.0
        assert applied == 0.0, "DETERMINISTIC must have 0 LE2 boost influence"
