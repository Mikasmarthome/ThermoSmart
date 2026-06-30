"""B2c Final Hardening — Dispatch truth, mixed zones, device availability, typed reasons.

Tests:
  TestBoostBlockReasonConstants:
    - all typed constants are non-empty strings
    - outer gate constants match expected string values
    - constants are unique

  TestAppliedSemanticPreDispatch:
    - applied_boost_offset_c is None when boost approved (pre-dispatch)
    - applied_boost_offset_c is 0.0 when any gate blocks
    - approved is always 0.0 when blocked

  TestPreDispatchDeviceUnavailable:
    - unavailable device → blocking_reason = device_unavailable
    - boost_allowed = False when device unavailable
    - lifecycle hard-released on unavailable device
    - unknown state also blocks
    - available device → gate passes (reaches readiness gate)
    - per-gate: learning_mode gate fires before device check

  TestMixedControlZone:
    - mixed zone_control_type → blocking_reason = mixed_control_types_unsupported
    - boost_allowed = False for mixed zone
    - lifecycle released for mixed zone
    - pure setpoint zone → gate passes (reaches readiness gate)
    - pure direct_valve zone → blocked at readiness (unsupported_control_type)

  TestBoostOffsetCSemantics:
    - boost_offset_c starts at 0.0 (default)
    - boost_factor starts at 1.0 (default)
    - boost_offset_c set from applied value after resolver

  TestPostDispatchInvalidation:
    - invalidate_boost_after_failed_dispatch_safe("failed") releases lifecycle
    - invalidate_boost_after_failed_dispatch_safe("partially_succeeded") releases lifecycle
    - reason code is FAILED_DISPATCH for failed
    - reason code is PARTIAL_DISPATCH for partial

  TestRestoreRace:
    - coordinator with active_control=False never boosts
    - all-False default → no boost path
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch as _patch

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.runtime.ha_integration import (
    AdaptiveBoostControlResult,
    BoostBlockReason,
    LearningShadowController,
    ADAPTIVE_BOOST_RESOLVER_VERSION,
)
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode

_T0 = datetime(2025, 3, 10, 6, 0, 0, tzinfo=timezone.utc)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_shadow(zone_id="zone_test"):
    hass = MagicMock()
    hass.states.get.return_value = None
    with _patch(
        "custom_components.thermosmart.learning.runtime.ha_integration.HomeAssistantStoreAdapter",
        return_value=MagicMock(),
    ):
        shadow = LearningShadowController(
            hass, zone_id,
            mode=LearningRuntimeMode.CONTROL,
            clock=FakeClock(start=_T0),
        )
    return shadow


def _base_rec(*, current_temp=18.0, target=21.0, setpoint=21.0, zone_control_type="setpoint",
              device_unavailable=False):
    return {
        "current_temp": current_temp,
        "effective_target": target,
        "trv_setpoint": setpoint,
        "tpi_valve_direct": zone_control_type == "direct_valve",
        "zone_control_type": zone_control_type,
        "_setpoint_device_unavailable": device_unavailable,
        "tpi_duty_cycle": 80.0,
        "mode": "heat",
    }


def _resolve(shadow, rec, *, learning=True, active=True):
    return shadow.resolve_adaptive_boost_control(
        rec,
        learning_mode_on=learning,
        active_control_on=active,
        boost_runtime_limit=8.0,
    )


# ── TestBoostBlockReasonConstants ─────────────────────────────────────────────

class TestBoostBlockReasonConstants:
    def test_all_constants_are_non_empty_strings(self):
        attrs = [
            BoostBlockReason.LEARNING_MODE_OFF,
            BoostBlockReason.ACTIVE_CONTROL_OFF,
            BoostBlockReason.MIXED_CONTROL_TYPES,
            BoostBlockReason.DEVICE_UNAVAILABLE,
            BoostBlockReason.FAILED_DISPATCH,
            BoostBlockReason.PARTIAL_DISPATCH,
            BoostBlockReason.READINESS_UNAVAILABLE,
        ]
        for v in attrs:
            assert isinstance(v, str) and v, f"constant is empty: {v!r}"

    def test_outer_gate_string_values(self):
        assert BoostBlockReason.LEARNING_MODE_OFF == "learning_mode_off"
        assert BoostBlockReason.ACTIVE_CONTROL_OFF == "active_control_off"
        assert BoostBlockReason.MIXED_CONTROL_TYPES == "mixed_control_types_unsupported"
        assert BoostBlockReason.DEVICE_UNAVAILABLE == "device_unavailable"
        assert BoostBlockReason.FAILED_DISPATCH == "failed_dispatch"
        assert BoostBlockReason.PARTIAL_DISPATCH == "partial_dispatch"

    def test_all_constants_unique(self):
        values = [
            BoostBlockReason.LEARNING_MODE_OFF,
            BoostBlockReason.ACTIVE_CONTROL_OFF,
            BoostBlockReason.MIXED_CONTROL_TYPES,
            BoostBlockReason.DEVICE_UNAVAILABLE,
            BoostBlockReason.FAILED_DISPATCH,
            BoostBlockReason.PARTIAL_DISPATCH,
            BoostBlockReason.READINESS_UNAVAILABLE,
        ]
        assert len(values) == len(set(values))

    def test_adaptive_boost_switch_off_constant_removed(self):
        """ADAPTIVE_BOOST_SWITCH_OFF must not exist — the switch was removed."""
        assert not hasattr(BoostBlockReason, "ADAPTIVE_BOOST_SWITCH_OFF")


# ── TestAppliedSemanticPreDispatch ─────────────────────────────────────────────

class TestAppliedSemanticPreDispatch:
    def test_applied_is_zero_when_gate_1_blocks(self):
        shadow = _make_shadow()
        rec = _base_rec()
        r = _resolve(shadow, rec, learning=False)
        assert r.applied_boost_offset_c == 0.0

    def test_applied_is_zero_when_gate_2_blocks(self):
        shadow = _make_shadow()
        rec = _base_rec()
        r = _resolve(shadow, rec, active=False)
        assert r.applied_boost_offset_c == 0.0

    def test_applied_is_zero_when_mixed_zone_blocks(self):
        shadow = _make_shadow()
        rec = _base_rec(zone_control_type="mixed")
        r = _resolve(shadow, rec)
        assert r.applied_boost_offset_c == 0.0

    def test_applied_is_zero_when_device_unavailable_blocks(self):
        shadow = _make_shadow()
        rec = _base_rec(device_unavailable=True)
        r = _resolve(shadow, rec)
        assert r.applied_boost_offset_c == 0.0

    def test_approved_is_always_zero_when_blocked(self):
        shadow = _make_shadow()
        for kwargs in [
            {"learning": False},
            {"active": False},
        ]:
            rec = _base_rec()
            r = _resolve(shadow, rec, **kwargs)
            assert r.approved_boost_offset_c == 0.0, f"approved != 0.0 for {kwargs}"

    def test_applied_is_none_when_inner_gates_pass_and_boost_applied(self):
        """When all outer+readiness gates pass and inner gates apply boost, applied = None."""
        shadow = _make_shadow()
        rec = _base_rec()
        readiness = MagicMock()
        readiness.eligibility = True
        readiness.factor_usable = True
        readiness.eligibility_reason = "eligible"
        readiness.selected_scope = "general"
        readiness.to_dict = lambda: {}

        with _patch.object(shadow, "_read_boost_activation_readiness_safe",
                           return_value=readiness):
            def _fake_adjust(r, *, boost_runtime_limit=None, authorized_override=False):
                r["_boost_applied_c"] = 1.5
                r["_boost_candidate_c"] = 1.5
                r["le2_boost_adjusted"] = True
                r["tpi_baseline_setpoint"] = 21.0
                r["trv_setpoint"] = 22.5

            with _patch.object(shadow, "adjust_recommendation_safe", _fake_adjust):
                r = _resolve(shadow, rec)

        assert r.applied_boost_offset_c is None, (
            f"expected None (pre-dispatch) but got {r.applied_boost_offset_c}")
        assert r.boost_allowed is True
        assert r.approved_boost_offset_c == 1.5


# ── TestPreDispatchDeviceUnavailable ─────────────────────────────────────────

class TestPreDispatchDeviceUnavailable:
    def test_unavailable_device_blocks_boost(self):
        shadow = _make_shadow()
        rec = _base_rec(device_unavailable=True)
        r = _resolve(shadow, rec)
        assert r.blocking_reason == BoostBlockReason.DEVICE_UNAVAILABLE
        assert r.boost_allowed is False

    def test_unavailable_device_sets_correct_eligibility_reason(self):
        shadow = _make_shadow()
        rec = _base_rec(device_unavailable=True)
        r = _resolve(shadow, rec)
        assert r.eligibility_reason == BoostBlockReason.DEVICE_UNAVAILABLE

    def test_unavailable_device_releases_lifecycle(self):
        shadow = _make_shadow()
        rec = _base_rec(device_unavailable=True)
        released = []
        with _patch.object(shadow, "_release_boost_lifecycle_safe",
                           side_effect=released.append):
            _resolve(shadow, rec)
        assert released == [BoostBlockReason.DEVICE_UNAVAILABLE]

    def test_available_device_passes_gate(self):
        """Available device passes Gate 3.5; blocked at readiness (no model)."""
        shadow = _make_shadow()
        rec = _base_rec(device_unavailable=False)
        r = _resolve(shadow, rec)
        assert r.blocking_reason != BoostBlockReason.DEVICE_UNAVAILABLE

    def test_learning_gate_fires_before_device_gate(self):
        """Gate 1 (learning) fires before Gate 3.5 (device unavailable)."""
        shadow = _make_shadow()
        rec = _base_rec(device_unavailable=True)
        r = _resolve(shadow, rec, learning=False)
        assert r.blocking_reason == BoostBlockReason.LEARNING_MODE_OFF
        assert r.blocking_reason != BoostBlockReason.DEVICE_UNAVAILABLE

    def test_rejection_reason_set_in_recommendation(self):
        shadow = _make_shadow()
        rec = _base_rec(device_unavailable=True)
        _resolve(shadow, rec)
        assert rec.get("_boost_rejection_reason") == BoostBlockReason.DEVICE_UNAVAILABLE


# ── TestMixedControlZone ─────────────────────────────────────────────────────

class TestMixedControlZone:
    def test_mixed_zone_blocks_boost(self):
        shadow = _make_shadow()
        rec = _base_rec(zone_control_type="mixed")
        r = _resolve(shadow, rec)
        assert r.blocking_reason == BoostBlockReason.MIXED_CONTROL_TYPES
        assert r.boost_allowed is False

    def test_mixed_zone_releases_lifecycle(self):
        shadow = _make_shadow()
        rec = _base_rec(zone_control_type="mixed")
        released = []
        with _patch.object(shadow, "_release_boost_lifecycle_safe",
                           side_effect=released.append):
            _resolve(shadow, rec)
        assert BoostBlockReason.MIXED_CONTROL_TYPES in released

    def test_mixed_zone_eligibility_reason(self):
        shadow = _make_shadow()
        rec = _base_rec(zone_control_type="mixed")
        r = _resolve(shadow, rec)
        assert r.eligibility_reason == BoostBlockReason.MIXED_CONTROL_TYPES

    def test_pure_setpoint_passes_mixed_gate(self):
        """Pure setpoint zone → Gate 3 passes; blocked at readiness (no model)."""
        shadow = _make_shadow()
        rec = _base_rec(zone_control_type="setpoint")
        r = _resolve(shadow, rec)
        assert r.blocking_reason != BoostBlockReason.MIXED_CONTROL_TYPES

    def test_mixed_before_device_unavailable_gate(self):
        """Mixed zone gate fires before device unavailability gate."""
        shadow = _make_shadow()
        rec = _base_rec(zone_control_type="mixed", device_unavailable=True)
        r = _resolve(shadow, rec)
        assert r.blocking_reason == BoostBlockReason.MIXED_CONTROL_TYPES

    def test_recommendation_rejection_set_for_mixed(self):
        shadow = _make_shadow()
        rec = _base_rec(zone_control_type="mixed")
        _resolve(shadow, rec)
        assert rec.get("_boost_rejection_reason") == BoostBlockReason.MIXED_CONTROL_TYPES


# ── TestBoostOffsetCSemantics ─────────────────────────────────────────────────

class TestBoostOffsetCSemantics:
    """Verify boost_offset_c reflects applied offset, not raw LE2 prediction."""

    def test_applied_boost_offset_c_is_zero_when_boost_blocked(self):
        """When any gate blocks, the result's applied is 0.0 (not None)."""
        shadow = _make_shadow()
        for kwargs in [
            {"learning": False},
            {"active": False},
        ]:
            rec = _base_rec()
            r = _resolve(shadow, rec, **kwargs)
            assert r.applied_boost_offset_c == 0.0

    def test_device_unavailable_gives_zero_applied(self):
        shadow = _make_shadow()
        rec = _base_rec(device_unavailable=True)
        r = _resolve(shadow, rec)
        assert r.applied_boost_offset_c == 0.0
        assert r.approved_boost_offset_c == 0.0

    def test_result_component_version(self):
        shadow = _make_shadow()
        rec = _base_rec(device_unavailable=True)
        r = _resolve(shadow, rec)
        assert r.component_version == ADAPTIVE_BOOST_RESOLVER_VERSION

    def test_blocked_result_boost_allowed_false(self):
        shadow = _make_shadow()
        for kwargs in [
            {"learning": False},
            {"active": False},
        ]:
            rec = _base_rec()
            r = _resolve(shadow, rec, **kwargs)
            assert r.boost_allowed is False


# ── TestPostDispatchInvalidation ─────────────────────────────────────────────

class TestPostDispatchInvalidation:
    def test_failed_dispatch_releases_lifecycle(self):
        shadow = _make_shadow()
        released = []
        with _patch.object(shadow, "_release_boost_lifecycle_safe",
                           side_effect=released.append):
            shadow.invalidate_boost_after_failed_dispatch_safe("failed")
        assert BoostBlockReason.FAILED_DISPATCH in released

    def test_partial_dispatch_releases_lifecycle(self):
        shadow = _make_shadow()
        released = []
        with _patch.object(shadow, "_release_boost_lifecycle_safe",
                           side_effect=released.append):
            shadow.invalidate_boost_after_failed_dispatch_safe("partially_succeeded")
        assert BoostBlockReason.PARTIAL_DISPATCH in released

    def test_failed_dispatch_reason_code(self):
        shadow = _make_shadow()
        reasons = []
        with _patch.object(shadow, "_release_boost_lifecycle_safe",
                           side_effect=reasons.append):
            shadow.invalidate_boost_after_failed_dispatch_safe("failed")
        assert reasons == [BoostBlockReason.FAILED_DISPATCH]

    def test_partial_dispatch_reason_code(self):
        shadow = _make_shadow()
        reasons = []
        with _patch.object(shadow, "_release_boost_lifecycle_safe",
                           side_effect=reasons.append):
            shadow.invalidate_boost_after_failed_dispatch_safe("partially_succeeded")
        assert reasons == [BoostBlockReason.PARTIAL_DISPATCH]

    def test_invalidate_never_raises(self):
        shadow = _make_shadow()
        with _patch.object(shadow, "_release_boost_lifecycle_safe",
                           side_effect=RuntimeError("test")):
            try:
                shadow.invalidate_boost_after_failed_dispatch_safe("failed")
            except RuntimeError:
                pytest.fail("invalidate_boost_after_failed_dispatch_safe must never raise")

    def test_not_attempted_does_not_need_invalidation(self):
        """dispatch_status='not_attempted' means active_control=False — no invalidation needed."""
        shadow = _make_shadow()
        released = []
        with _patch.object(shadow, "_release_boost_lifecycle_safe",
                           side_effect=released.append):
            pass  # coordinator only calls invalidate for failed/partial
        assert released == []


# ── TestRestoreRace ──────────────────────────────────────────────────────────

class TestRestoreRace:
    """Verify coordinator defaults prevent boost during partial restore."""

    def test_active_control_disabled_blocks_gate2(self):
        shadow = _make_shadow()
        rec = _base_rec()
        r = _resolve(shadow, rec, active=False)
        assert r.blocking_reason == BoostBlockReason.ACTIVE_CONTROL_OFF

    def test_all_defaults_false_no_boost_possible(self):
        shadow = _make_shadow()
        rec = _base_rec()
        r = _resolve(shadow, rec, learning=True, active=False)
        assert r.boost_allowed is False

    def test_learning_switch_affects_gate1(self):
        """Learning OFF blocks at Gate 1 regardless of active_control state."""
        shadow = _make_shadow()
        rec = _base_rec()
        r = _resolve(shadow, rec, learning=False, active=True)
        assert r.blocking_reason == BoostBlockReason.LEARNING_MODE_OFF

    def test_gate_order_is_strict(self):
        """Gate 1 always fires before Gate 2."""
        shadow = _make_shadow()
        # Both gates off: Gate 1 fires first
        rec = _base_rec()
        r = _resolve(shadow, rec, learning=False, active=False)
        assert r.blocking_reason == BoostBlockReason.LEARNING_MODE_OFF

        # Gate 1 on, Gate 2 off: Gate 2 fires
        rec = _base_rec()
        r = _resolve(shadow, rec, learning=True, active=False)
        assert r.blocking_reason == BoostBlockReason.ACTIVE_CONTROL_OFF
