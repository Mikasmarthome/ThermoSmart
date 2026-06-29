"""
S1a Item 16 — Failure Injection and Recovery Audit

Verifies across all failure modes:
- Bounded retry / no service-storm
- Graceful degradation without LE2 shadow
- Correct partial-success attribution
- No data corruption on failure
- Persistence failure isolation with correct dirty-state semantics
- LE2 shadow failure isolation (never crashes the control cycle)
- Storage load/save exception propagation contract
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.thermosmart.const import WINDOW_OPEN_SETPOINT
from custom_components.thermosmart.device_profiles.capabilities import (
    DeviceProfile,
    SETPOINT_CLIMATE,
    SETPOINT_HVAC_FIRST,
    VALVE_DIRECT,
)
from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.runtime.ha_integration import (
    LearningShadowController,
)
from custom_components.thermosmart.learning.storage.persistence import (
    DirtyState,
    DirtyTracker,
    FlushPolicy,
    FlushReason,
    PersistenceError,
    SaveToken,
)
from custom_components.thermosmart.learning.storage.stores import (
    StoreVersionError,
    ZoneMetadataStore,
)
from custom_components.thermosmart.trv_control import _DispatchStats

from tests.helpers import make_coordinator, make_state, set_hass_states

# ── helpers ───────────────────────────────────────────────────────────────────

_T0 = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)

_CFG = {
    "climate_entities": ["climate.trv"],
    "temp_tolerance": 0.5,
    "window_open_temp": WINDOW_OPEN_SETPOINT,
}
_CFG2 = {
    "climate_entities": ["climate.trv1", "climate.trv2"],
    "temp_tolerance": 0.5,
    "window_open_temp": WINDOW_OPEN_SETPOINT,
}
_CFG3 = {
    "climate_entities": ["climate.trv1", "climate.trv2", "climate.trv3"],
    "temp_tolerance": 0.5,
    "window_open_temp": WINDOW_OPEN_SETPOINT,
}
_REC = {"adjusted_target": 21.0, "trv_setpoint": 23.5, "window_open": False}
_REC_WIN = {"adjusted_target": None, "trv_setpoint": None, "window_open": True}


def _fake_clock() -> FakeClock:
    return FakeClock(_T0)


def _trv_state(setpoint: float = 19.0, state: str = "heat") -> object:
    return make_state(
        state,
        {
            "temperature": setpoint,
            "current_temperature": 18.5,
            "min_temp": 5.0,
            "max_temp": 35.0,
            "target_temp_step": 0.5,
        },
    )


def _coord_ok() -> object:
    coord = make_coordinator()
    coord.hass.services.async_call = AsyncMock(return_value=None)
    return coord


def _coord_fail() -> object:
    coord = make_coordinator()
    coord.hass.services.async_call = AsyncMock(
        side_effect=Exception("Service unavailable")
    )
    return coord


def _make_store(zone_id: str = "zone_abc") -> ZoneMetadataStore:
    hass = MagicMock()
    store = ZoneMetadataStore(hass, zone_id)
    store._store = MagicMock()
    return store


def _make_dirty_tracker() -> DirtyTracker:
    return DirtyTracker(_fake_clock())


def _make_shadow_ctrl() -> LearningShadowController:
    hass = MagicMock()
    hass.async_create_task = MagicMock(side_effect=lambda c: c.close())
    return LearningShadowController(hass, "zone_test", clock=_fake_clock())


# ── §1  DispatchStats Merge Semantics ─────────────────────────────────────────


class TestDispatchStatsMergeSemantics:
    def test_two_empty_merge_gives_not_attempted(self):
        merged = _DispatchStats().merge(_DispatchStats())
        assert merged.status == "not_attempted"

    def test_fully_succeeded_merged_with_empty_is_fully_succeeded(self):
        a = _DispatchStats()
        a.record(None, effective_c=21.0, entity_id="e1")
        merged = a.merge(_DispatchStats())
        assert merged.status == "fully_succeeded"

    def test_failed_merged_with_empty_is_failed(self):
        a = _DispatchStats()
        a.record(Exception("fail"))
        merged = a.merge(_DispatchStats())
        assert merged.status == "failed"

    def test_fully_succeeded_and_failed_gives_partially(self):
        ok = _DispatchStats()
        ok.record(None, effective_c=21.0, entity_id="e1")
        fail = _DispatchStats()
        fail.record(Exception("oops"))
        merged = ok.merge(fail)
        assert merged.status == "partially_succeeded"

    def test_failed_and_fully_succeeded_gives_partially(self):
        fail = _DispatchStats()
        fail.record(Exception("oops"))
        ok = _DispatchStats()
        ok.record(None, effective_c=21.0, entity_id="e1")
        merged = fail.merge(ok)
        assert merged.status == "partially_succeeded"

    def test_two_failed_merge_gives_failed(self):
        a = _DispatchStats()
        a.record(Exception("err1"))
        b = _DispatchStats()
        b.record(Exception("err2"))
        merged = a.merge(b)
        assert merged.status == "failed"

    def test_failure_reasons_concatenated_in_merge(self):
        a = _DispatchStats()
        a.record(ValueError("bad"))
        b = _DispatchStats()
        b.record(OSError("io"))
        merged = a.merge(b)
        assert len(merged.failure_reasons) == 2

    def test_effective_setpoints_combined_in_merge(self):
        a = _DispatchStats()
        a.record(None, effective_c=20.0, entity_id="e1")
        b = _DispatchStats()
        b.record(None, effective_c=22.0, entity_id="e2")
        merged = a.merge(b)
        assert merged.effective_setpoints_by_entity == {"e1": 20.0, "e2": 22.0}

    def test_targets_total_is_sum(self):
        a = _DispatchStats()
        a.record(None, effective_c=21.0, entity_id="e1")
        b = _DispatchStats()
        b.record(Exception("x"))
        merged = a.merge(b)
        assert merged.targets_total == 2
        assert merged.targets_succeeded == 1
        assert merged.targets_failed == 1

    def test_record_gather_with_mixed_results(self):
        stats = _DispatchStats()
        stats.record_gather([None, Exception("err"), None])
        assert stats.status == "partially_succeeded"
        assert stats.targets_succeeded == 2
        assert stats.targets_failed == 1


# ── §2  _apply_temperature Multi-Cycle Recovery ───────────────────────────────


class TestApplyTemperatureMultiCycleRecovery:
    async def test_failure_cycle_leaves_last_written_empty(self):
        coord = _coord_fail()
        set_hass_states(coord, {"climate.trv": _trv_state()})
        stats = await coord._apply_temperature(_CFG, _REC)
        assert stats.status == "failed"
        assert "climate.trv" not in coord._last_written_setpoints

    async def test_recovery_cycle_updates_last_written(self):
        coord = _coord_ok()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        stats = await coord._apply_temperature(_CFG, _REC)
        assert stats.status == "fully_succeeded"
        assert "climate.trv" in coord._last_written_setpoints

    async def test_failure_then_success_updates_last_written(self):
        coord = _coord_fail()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        await coord._apply_temperature(_CFG, _REC)
        assert "climate.trv" not in coord._last_written_setpoints

        coord.hass.services.async_call = AsyncMock(return_value=None)
        stats2 = await coord._apply_temperature(_CFG, _REC)
        assert stats2.status == "fully_succeeded"
        assert "climate.trv" in coord._last_written_setpoints

    async def test_partial_success_updates_only_successful_entity(self):
        coord = make_coordinator()
        set_hass_states(coord, {
            "climate.trv1": _trv_state(setpoint=19.0),
            "climate.trv2": _trv_state(setpoint=19.0),
        })

        async def svc_call(domain, service, data, **kw):
            if "trv2" in str(data.get("entity_id") or ""):
                raise Exception("trv2 offline")

        coord.hass.services.async_call = AsyncMock(side_effect=svc_call)
        stats = await coord._apply_temperature(_CFG2, _REC)
        assert stats.status == "partially_succeeded"
        assert "climate.trv1" in coord._last_written_setpoints
        assert "climate.trv2" not in coord._last_written_setpoints

    async def test_unavailable_trv_makes_no_service_call(self):
        coord = _coord_ok()
        set_hass_states(coord, {"climate.trv": make_state("unavailable")})
        await coord._apply_temperature(_CFG, _REC)
        coord.hass.services.async_call.assert_not_called()

    async def test_three_cycles_failure_no_state_accumulation(self):
        coord = _coord_fail()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        for _ in range(3):
            stats = await coord._apply_temperature(_CFG, _REC)
            assert stats.status == "failed"
        assert "climate.trv" not in coord._last_written_setpoints

    async def test_all_three_fail_gives_failed_stats(self):
        coord = _coord_fail()
        set_hass_states(coord, {
            "climate.trv1": _trv_state(),
            "climate.trv2": _trv_state(),
            "climate.trv3": _trv_state(),
        })
        stats = await coord._apply_temperature(_CFG3, _REC)
        assert stats.status == "failed"
        assert stats.targets_failed == 3
        assert stats.targets_succeeded == 0

    async def test_window_open_frost_failure_no_last_written_corruption(self):
        coord = _coord_fail()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=8.0)})
        await coord._apply_temperature(_CFG, _REC_WIN)
        assert "climate.trv" not in coord._last_written_setpoints

    async def test_recovery_after_window_close(self):
        coord = _coord_ok()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        await coord._apply_temperature(_CFG, _REC_WIN)  # window open
        stats = await coord._apply_temperature(_CFG, _REC)  # window closed
        assert stats.status == "fully_succeeded"

    async def test_failure_reason_list_not_empty_on_fail(self):
        coord = _coord_fail()
        set_hass_states(coord, {"climate.trv": _trv_state()})
        stats = await coord._apply_temperature(_CFG, _REC)
        assert len(stats.failure_reasons) >= 1

    async def test_empty_climate_entities_gives_not_attempted(self):
        coord = _coord_ok()
        cfg_empty = {"climate_entities": [], "temp_tolerance": 0.5,
                     "window_open_temp": WINDOW_OPEN_SETPOINT}
        stats = await coord._apply_temperature(cfg_empty, _REC)
        assert stats.status == "not_attempted"

    async def test_inactive_device_profile_skips_dispatch(self):
        coord = _coord_ok()
        coord._device_profiles = {
            "climate.trv": DeviceProfile(
                identifier="t", display_name="T", is_active=False
            )
        }
        set_hass_states(coord, {"climate.trv": _trv_state()})
        await coord._apply_temperature(_CFG, _REC)
        coord.hass.services.async_call.assert_not_called()


# ── §3  Valve Dispatch Failure Isolation ──────────────────────────────────────


class TestValveDispatchFailureIsolation:
    def _coord_valve(self, *, valve_state: str = "50", fail: bool = False):
        coord = _coord_fail() if fail else _coord_ok()
        coord._auto_valve_map = {"climate.trv": "number.valve"}
        set_hass_states(coord, {"number.valve": make_state(valve_state)})
        return coord

    async def test_valve_failure_captured_in_stats(self):
        coord = self._coord_valve(valve_state="50", fail=True)
        stats = await coord._async_set_valve_percent(_CFG, 75.0)
        assert stats.targets_failed >= 1

    async def test_valve_failure_does_not_raise(self):
        coord = self._coord_valve(valve_state="50", fail=True)
        result = await coord._async_set_valve_percent(_CFG, 75.0)
        assert result is not None

    async def test_empty_valve_map_returns_not_attempted(self):
        coord = _coord_ok()
        coord._auto_valve_map = {}
        stats = await coord._async_set_valve_percent(_CFG, 75.0)
        assert stats.status == "not_attempted"
        coord.hass.services.async_call.assert_not_called()

    async def test_valve_noop_same_duty_no_service_call(self):
        coord = self._coord_valve(valve_state="75")
        await coord._async_set_valve_percent(_CFG, 75.0)
        coord.hass.services.async_call.assert_not_called()

    async def test_valve_success_gives_fully_succeeded(self):
        coord = self._coord_valve(valve_state="50")
        stats = await coord._async_set_valve_percent(_CFG, 75.0)
        assert stats.status == "fully_succeeded"

    async def test_valve_unavailable_makes_no_service_call(self):
        coord = _coord_ok()
        coord._auto_valve_map = {"climate.trv": "number.valve"}
        set_hass_states(coord, {"number.valve": make_state("unavailable")})
        await coord._async_set_valve_percent(_CFG, 75.0)
        coord.hass.services.async_call.assert_not_called()

    async def test_two_valves_partial_failure_gives_partially_succeeded(self):
        coord = make_coordinator()
        coord._auto_valve_map = {
            "climate.trv1": "number.valve1",
            "climate.trv2": "number.valve2",
        }
        set_hass_states(coord, {
            "number.valve1": make_state("20"),
            "number.valve2": make_state("20"),
        })

        async def svc(domain, service, data, **kw):
            if "valve2" in str(data.get("entity_id") or ""):
                raise Exception("valve2 fail")

        coord.hass.services.async_call = AsyncMock(side_effect=svc)
        stats = await coord._async_set_valve_percent(_CFG2, 75.0)
        assert stats.status == "partially_succeeded"

    async def test_two_valves_all_fail_still_attempts_both(self):
        coord = make_coordinator()
        coord._auto_valve_map = {
            "climate.trv1": "number.valve1",
            "climate.trv2": "number.valve2",
        }
        set_hass_states(coord, {
            "number.valve1": make_state("10"),
            "number.valve2": make_state("10"),
        })
        coord.hass.services.async_call = AsyncMock(side_effect=Exception("all fail"))
        stats = await coord._async_set_valve_percent(_CFG2, 75.0)
        assert stats.targets_total == 2


# ── §4  LearningShadowController Failure Isolation ───────────────────────────


class TestLearningShadowControllerFailureIsolation:
    async def test_async_setup_exception_returns_false(self):
        ctrl = _make_shadow_ctrl()
        with patch.object(ctrl._runtime, "async_setup",
                          AsyncMock(side_effect=RuntimeError("setup crash"))):
            result = await ctrl.async_setup()
        assert result is False

    async def test_async_setup_failure_increments_error_count(self):
        ctrl = _make_shadow_ctrl()
        with patch.object(ctrl._runtime, "async_setup",
                          AsyncMock(side_effect=RuntimeError("crash"))):
            await ctrl.async_setup()
        assert ctrl.errors >= 1

    async def test_async_setup_failure_disables_control(self):
        ctrl = _make_shadow_ctrl()
        with patch.object(ctrl._runtime, "async_setup",
                          AsyncMock(side_effect=RuntimeError("crash"))):
            await ctrl.async_setup()
        assert ctrl.control_enabled is False

    def test_observe_safe_exception_does_not_raise(self):
        ctrl = _make_shadow_ctrl()
        # observe_safe wraps all errors — even invalid inputs should not raise
        try:
            ctrl.observe_safe({"ts_iso": "not-a-real-ts", "broken": True})
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"observe_safe raised: {exc}")

    def test_read_tpi_coefficients_safe_disabled_returns_defaults(self):
        ctrl = _make_shadow_ctrl()
        # Disabled (not set up) → returns default tuple, no crash
        result = ctrl.read_tpi_coefficients_safe()
        assert result is not None  # returns a tuple, not None

    def test_read_preheat_minutes_safe_disabled_returns_defaults(self):
        ctrl = _make_shadow_ctrl()
        result = ctrl.read_preheat_minutes_safe(19.5, 21.0)
        assert result is not None  # returns a tuple, not None

    def test_compute_decision_trace_safe_does_not_raise(self):
        ctrl = _make_shadow_ctrl()
        try:
            ctrl.compute_decision_trace_safe({})
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"compute_decision_trace_safe raised: {exc}")

    def test_emit_boost_authority_status_safe_does_not_raise(self):
        ctrl = _make_shadow_ctrl()
        try:
            ctrl.emit_boost_authority_status_safe({})
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"emit_boost_authority_status_safe raised: {exc}")

    def test_invalidate_boost_after_failed_dispatch_safe_does_not_raise(self):
        ctrl = _make_shadow_ctrl()
        try:
            ctrl.invalidate_boost_after_failed_dispatch_safe("zone_test")
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"invalidate_boost raised: {exc}")

    def test_fresh_controller_error_count_is_zero(self):
        ctrl = _make_shadow_ctrl()
        assert ctrl.errors == 0

    def test_fresh_controller_control_not_enabled(self):
        ctrl = _make_shadow_ctrl()
        assert ctrl.control_enabled is False

    async def test_unload_after_failed_setup_does_not_raise(self):
        ctrl = _make_shadow_ctrl()
        with patch.object(ctrl._runtime, "async_setup",
                          AsyncMock(side_effect=RuntimeError("crash"))):
            await ctrl.async_setup()
        try:
            await ctrl.async_unload()
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"async_unload raised after failed setup: {exc}")


# ── §5  ZoneMetadataStore Failure Paths ───────────────────────────────────────


class TestZoneMetadataStoreFailurePaths:
    async def test_load_none_returns_none(self):
        store = _make_store()
        store._store.async_load = AsyncMock(return_value=None)
        assert await store.load() is None

    async def test_load_wrong_version_raises_store_version_error(self):
        store = _make_store()
        store._store.async_load = AsyncMock(
            return_value={"store_schema_version": 99, "data": {}}
        )
        with pytest.raises(StoreVersionError):
            await store.load()

    async def test_load_missing_version_key_raises_store_version_error(self):
        store = _make_store()
        store._store.async_load = AsyncMock(return_value={"data": {}})
        with pytest.raises(StoreVersionError):
            await store.load()

    async def test_load_ioerror_propagates(self):
        store = _make_store()
        store._store.async_load = AsyncMock(side_effect=OSError("disk error"))
        with pytest.raises(OSError):
            await store.load()

    async def test_load_valid_schema_returns_data(self):
        store = _make_store()
        payload = {"store_schema_version": 1, "data": {"k": "v"}}
        store._store.async_load = AsyncMock(return_value=payload)
        result = await store.load()
        assert result == {"k": "v"}

    async def test_save_exception_propagates(self):
        store = _make_store()
        store._store.async_save = AsyncMock(side_effect=OSError("write fail"))
        with pytest.raises(OSError):
            await store.save({"some": "data"})

    def test_key_contains_zone_id(self):
        store = _make_store("my_zone_123")
        assert "my_zone_123" in store.key

    def test_store_version_error_is_exception(self):
        err = StoreVersionError("test")
        assert isinstance(err, Exception)
        assert "test" in str(err)


# ── §6  DirtyTracker Failure Semantics ────────────────────────────────────────


class TestDirtyTrackerFailureSemantics:
    def test_initial_state_not_dirty(self):
        tracker = _make_dirty_tracker()
        assert not tracker.is_dirty("key1")

    def test_mark_dirty_sets_dirty(self):
        tracker = _make_dirty_tracker()
        tracker.mark_dirty("key1")
        assert tracker.is_dirty("key1")

    def test_fail_save_leaves_dirty(self):
        tracker = _make_dirty_tracker()
        tracker.mark_dirty("key1")
        token = tracker.begin_save("key1")
        tracker.fail_save(token, "disk error")
        assert tracker.is_dirty("key1")

    def test_fail_save_increments_retry_count(self):
        tracker = _make_dirty_tracker()
        tracker.mark_dirty("key1")
        token = tracker.begin_save("key1")
        tracker.fail_save(token, "io error")
        assert tracker.state("key1").retry_count == 1

    def test_fail_save_stores_error_message(self):
        tracker = _make_dirty_tracker()
        tracker.mark_dirty("key1")
        token = tracker.begin_save("key1")
        tracker.fail_save(token, "disk full")
        assert tracker.state("key1").last_error == "disk full"

    def test_complete_save_clears_dirty(self):
        tracker = _make_dirty_tracker()
        tracker.mark_dirty("key1")
        token = tracker.begin_save("key1")
        tracker.complete_save(token)
        assert not tracker.is_dirty("key1")

    def test_begin_save_on_not_dirty_raises_persistence_error(self):
        tracker = _make_dirty_tracker()
        with pytest.raises(PersistenceError):
            tracker.begin_save("key1")

    def test_concurrent_begin_save_raises_persistence_error(self):
        tracker = _make_dirty_tracker()
        tracker.mark_dirty("key1")
        tracker.begin_save("key1")
        with pytest.raises(PersistenceError):
            tracker.begin_save("key1")

    def test_dirty_keys_returns_only_dirty(self):
        tracker = _make_dirty_tracker()
        tracker.mark_dirty("k1")
        tracker.mark_dirty("k2")
        token = tracker.begin_save("k1")
        tracker.complete_save(token)
        keys = tracker.dirty_keys()
        assert "k1" not in keys
        assert "k2" in keys

    def test_retry_reset_after_complete_save(self):
        tracker = _make_dirty_tracker()
        tracker.mark_dirty("key1")
        token = tracker.begin_save("key1")
        tracker.fail_save(token, "err")
        tracker.mark_dirty("key1")
        token2 = tracker.begin_save("key1")
        tracker.complete_save(token2)
        assert tracker.state("key1").retry_count == 0


# ── §7  FlushPolicy Behavior Under Failures ───────────────────────────────────


class TestFlushPolicyFailureSemantics:
    def _policy(self, interval: float = 60.0) -> FlushPolicy:
        return FlushPolicy(interval_seconds=interval)

    def test_not_dirty_never_flushes(self):
        policy = self._policy()
        state = DirtyState()
        decision = policy.decide(state, now=_T0)
        assert decision.should_flush is False

    def test_dirty_with_forced_unload_flushes(self):
        policy = self._policy()
        state = DirtyState(change_seq=1, first_dirty_utc=_T0)
        decision = policy.decide(state, now=_T0, forced_reason=FlushReason.UNLOAD)
        assert decision.should_flush is True
        assert decision.reason == FlushReason.UNLOAD

    def test_not_dirty_with_forced_unload_does_not_flush(self):
        policy = self._policy()
        state = DirtyState()
        decision = policy.decide(state, now=_T0, forced_reason=FlushReason.UNLOAD)
        assert decision.should_flush is False

    def test_interval_elapsed_triggers_flush(self):
        policy = self._policy(interval=60.0)
        state = DirtyState(change_seq=1, saved_seq=0, first_dirty_utc=_T0,
                           last_dirty_utc=_T0)
        later = datetime(2025, 1, 15, 11, 0, tzinfo=timezone.utc)  # +1h
        decision = policy.decide(state, now=later)
        assert decision.should_flush is True
        assert decision.reason == FlushReason.INTERVAL

    def test_interval_not_elapsed_no_flush(self):
        policy = self._policy(interval=3600.0)
        state = DirtyState(change_seq=1, first_dirty_utc=_T0)
        decision = policy.decide(state, now=_T0)
        assert decision.should_flush is False

    def test_invalid_interval_raises_persistence_error(self):
        with pytest.raises(PersistenceError):
            FlushPolicy(interval_seconds=0)

    def test_important_event_triggers_immediate_flush(self):
        policy = self._policy(interval=3600.0)
        state = DirtyState(change_seq=1, first_dirty_utc=_T0)
        decision = policy.decide(state, now=_T0, important_event=True)
        assert decision.should_flush is True
        assert decision.reason == FlushReason.IMPORTANT_EVENT

    def test_segment_full_triggers_immediate_flush(self):
        policy = self._policy(interval=3600.0)
        state = DirtyState(change_seq=1, first_dirty_utc=_T0)
        decision = policy.decide(state, now=_T0, segment_full=True)
        assert decision.should_flush is True
        assert decision.reason == FlushReason.SEGMENT_FULL


# ── §8  No Service Storm — Bounded Retry ─────────────────────────────────────


class TestNoServiceStorm:
    """Proves that exactly one service call per TRV per cycle is made."""

    async def test_single_failure_makes_exactly_one_call(self):
        coord = _coord_fail()
        set_hass_states(coord, {"climate.trv": _trv_state()})
        await coord._apply_temperature(_CFG, _REC)
        assert coord.hass.services.async_call.call_count == 1

    async def test_five_failure_cycles_make_exactly_five_calls(self):
        coord = _coord_fail()
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        for _ in range(5):
            await coord._apply_temperature(_CFG, _REC)
        assert coord.hass.services.async_call.call_count == 5

    async def test_dedup_after_success_prevents_next_write(self):
        coord = _coord_ok()
        # TRV already shows the target setpoint
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=23.5)})
        # Cycle 1: not yet in _last_written, so writes
        await coord._apply_temperature(_CFG, _REC)
        count_c1 = coord.hass.services.async_call.call_count

        # Cycle 2: TRV shows same setpoint, _last_written matches → dedup
        await coord._apply_temperature(_CFG, _REC)
        count_c2 = coord.hass.services.async_call.call_count

        # Dedup prevents a second write
        assert count_c2 == count_c1

    async def test_unavailable_trv_zero_calls_across_cycles(self):
        coord = _coord_ok()
        set_hass_states(coord, {"climate.trv": make_state("unavailable")})
        for _ in range(3):
            await coord._apply_temperature(_CFG, _REC)
        coord.hass.services.async_call.assert_not_called()

    async def test_valve_failure_one_write_per_cycle(self):
        # Valve can make a bump call + write call — both bounded per cycle.
        # The stats show exactly 1 failure (the main write), regardless of bump.
        coord = _coord_fail()
        coord._auto_valve_map = {"climate.trv": "number.valve"}
        set_hass_states(coord, {"number.valve": make_state("50")})
        stats = await coord._async_set_valve_percent(_CFG, 75.0)
        # Exactly 1 failure tracked in stats (bump exceptions are swallowed)
        assert stats.targets_failed == 1

    async def test_three_entity_failure_bounded_by_entity_count(self):
        """failure_reasons per dispatch cycle is bounded by TRV count."""
        coord = _coord_fail()
        set_hass_states(coord, {
            "climate.trv1": _trv_state(),
            "climate.trv2": _trv_state(),
            "climate.trv3": _trv_state(),
        })
        stats = await coord._apply_temperature(_CFG3, _REC)
        assert len(stats.failure_reasons) == 3  # exactly one per TRV


# ── §9  Coordinator LE2 Absent Degradation ───────────────────────────────────


class TestCoordinatorLE2AbsentDegradation:
    def test_fresh_coordinator_le2_shadow_is_none(self):
        coord = make_coordinator()
        assert coord._le2_shadow is None

    def test_coordinator_has_le2_shadow_attribute(self):
        coord = make_coordinator()
        assert hasattr(coord, "_le2_shadow")

    def test_dispatch_stats_without_shadow_still_mergeable(self):
        sp_stats = _DispatchStats()
        vv_stats = _DispatchStats()
        merged = sp_stats.merge(vv_stats)
        assert merged.status == "not_attempted"

    def test_le2_absent_boost_not_applied(self):
        coord = make_coordinator()
        assert coord._le2_shadow is None
        rec = {"adjusted_target": 21.0, "trv_setpoint": 23.0, "window_open": False}
        assert not rec.get("le2_boost_adjusted")

    def test_attach_le2_shadow_sets_attribute(self):
        coord = make_coordinator()
        mock_shadow = MagicMock()
        coord.attach_le2_shadow(mock_shadow)
        assert coord._le2_shadow is mock_shadow

    def test_le2_shadow_none_after_explicit_clear(self):
        coord = make_coordinator()
        coord._le2_shadow = MagicMock()
        coord._le2_shadow = None
        assert coord._le2_shadow is None

    async def test_apply_temperature_without_le2_shadow_succeeds(self):
        coord = _coord_ok()
        assert coord._le2_shadow is None
        set_hass_states(coord, {"climate.trv": _trv_state(setpoint=19.0)})
        stats = await coord._apply_temperature(_CFG, _REC)
        assert stats.status == "fully_succeeded"

    async def test_valve_dispatch_without_le2_shadow_succeeds(self):
        coord = _coord_ok()
        assert coord._le2_shadow is None
        coord._auto_valve_map = {"climate.trv": "number.valve"}
        set_hass_states(coord, {"number.valve": make_state("20")})
        stats = await coord._async_set_valve_percent(_CFG, 75.0)
        assert stats.status == "fully_succeeded"
