"""B2c: Adaptive Boost Activation — Double Gate, Resolver, semantics.

Tests:
  - Double gate: all 4 combinations of learning_mode / active_control
  - Resolver result typing: blocking_reason, eligibility_reason, applied_c
  - Hard release on each gate change mid-boost
  - Readiness NOT eligible → blocked + lifecycle released
  - Direct valve → blocked at readiness (control_type gate)
  - Setpoint TRV → approved when all gates pass + eligible model
  - boost_allowed only when all gates pass
  - Legacy adjust_recommendation_safe still blocked without resolver
  - Soft step-down only on TPI sufficient (not on active_control-off)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch as _patch

from datetime import datetime, timezone

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.runtime.ha_integration import (
    AdaptiveBoostControlResult,
    ADAPTIVE_BOOST_RESOLVER_VERSION,
    LearningShadowController,
)
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
from tests.helpers import make_coordinator
from tests.helpers_ha_runtime import (
    FakeStore,
    RecordingCoordinator,
    attach_shadow,
    make_recording_coordinator,
)

_T0 = datetime(2025, 3, 10, 6, 0, 0, tzinfo=timezone.utc)


# ── helpers ──────────────────────────────────────────────────────────────────

def _rec(target=21.0, current=18.0, setpoint=21.0, *, valve_direct=False, window_open=False,
         heating_failure=False, preheat=False, ec_state=None, mode="heat"):
    r = {
        "effective_target": target,
        "adjusted_target": target,
        "current_temp": current,
        "trv_setpoint": setpoint,
        "tpi_valve_direct": valve_direct,
        "window_open": window_open,
        "heating_failure": heating_failure,
        "preheat_minutes": 30 if preheat else 0,
        "mode": mode,
        "boost_active": False,
    }
    if ec_state is not None:
        r["early_cutoff_state"] = ec_state
    return r


def _shadow(*, zone_id="z1", mode=LearningRuntimeMode.CONTROL):
    return LearningShadowController(
        hass=MagicMock(), zone_id=zone_id, store=FakeStore(),
        mode=mode, clock=FakeClock(start=_T0),
    )


# ── AdaptiveBoostControlResult dataclass ─────────────────────────────────────

class TestAdaptiveBoostControlResult:
    def test_fields_present(self):
        r = AdaptiveBoostControlResult(
            requested_boost_offset_c=1.5, approved_boost_offset_c=1.0,
            applied_boost_offset_c=1.0, boost_allowed=True,
            selected_scope="general", eligibility_reason="eligible",
            blocking_reason=None, release_reason=None, clamp_applied=False,
            source_decision_id="did1",
        )
        assert r.requested_boost_offset_c == 1.5
        assert r.approved_boost_offset_c == 1.0
        assert r.boost_allowed is True
        assert r.component_version == ADAPTIVE_BOOST_RESOLVER_VERSION
        # Trace fields with defaults
        assert r.authorization_source is None
        assert r.bypassed_gates == ()

    def test_trace_fields_when_authorization_present(self):
        r = AdaptiveBoostControlResult(
            requested_boost_offset_c=1.5, approved_boost_offset_c=1.0,
            applied_boost_offset_c=None, boost_allowed=True,
            selected_scope="general", eligibility_reason="eligible",
            blocking_reason=None, release_reason=None, clamp_applied=False,
            source_decision_id="did1",
            authorization_source="bootstrap_activation",
            bypassed_gates=("confidence", "reliability"),
        )
        assert r.authorization_source == "bootstrap_activation"
        assert "confidence" in r.bypassed_gates

    def test_zero_result_has_no_none_applied(self):
        r = AdaptiveBoostControlResult(
            requested_boost_offset_c=None, approved_boost_offset_c=0.0,
            applied_boost_offset_c=0.0, boost_allowed=False,
            selected_scope=None, eligibility_reason="learning_mode_off",
            blocking_reason="learning_mode_off", release_reason=None,
            clamp_applied=False, source_decision_id=None,
        )
        assert r.approved_boost_offset_c == 0.0   # never None when gate blocked
        assert r.applied_boost_offset_c == 0.0


# ── Double Gate: all 4 combinations ──────────────────────────────────────────

class TestDoubleGate:
    """Every combination of the two outer gates must behave deterministically."""

    async def _run(self, coord, *, learning, active):
        sh = attach_shadow(coord)
        await sh.async_setup()
        rec = _rec()
        result = sh.resolve_adaptive_boost_control(
            rec,
            learning_mode_on=learning,
            active_control_on=active,
        )
        return result, rec

    async def test_all_off_blocked(self, make_rc):
        r, rec = await self._run(make_rc, learning=False, active=False)
        assert r.boost_allowed is False
        assert r.blocking_reason == "learning_mode_off"
        assert r.approved_boost_offset_c == 0.0

    async def test_learning_off_active_on_blocked(self, make_rc):
        r, rec = await self._run(make_rc, learning=False, active=True)
        assert r.boost_allowed is False
        assert r.blocking_reason == "learning_mode_off"

    async def test_learning_on_active_off_blocked(self, make_rc):
        r, rec = await self._run(make_rc, learning=True, active=False)
        assert r.boost_allowed is False
        assert r.blocking_reason == "active_control_off"

    async def test_both_on_reaches_readiness_gate(self, make_rc):
        # With both gates open but no trained model, readiness gate blocks
        r, rec = await self._run(make_rc, learning=True, active=True)
        assert r.boost_allowed is False
        assert r.approved_boost_offset_c == 0.0
        assert r.applied_boost_offset_c == 0.0
        # Blocking reason comes from readiness, not from the outer gates
        assert r.blocking_reason not in (
            "learning_mode_off", "active_control_off",
        ), "readiness gate should fire, not outer gates"

    @pytest.fixture
    def make_rc(self):
        return make_recording_coordinator()


# ── Readiness gate ────────────────────────────────────────────────────────────

class TestReadinessGate:
    """BoostActivationReadiness must block when eligibility == False."""

    async def test_direct_valve_not_eligible(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        rec = _rec(valve_direct=True)
        r = sh.resolve_adaptive_boost_control(
            rec,
            learning_mode_on=True, active_control_on=True,
        )
        # direct_valve: control_type="direct_valve" → readiness UNSUPPORTED_CONTROL_TYPE
        assert r.approved_boost_offset_c == 0.0
        assert r.boost_allowed is False
        assert r.blocking_reason is not None

    async def test_readiness_unavailable_returns_zero(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        sh._enabled = False  # disable LE2 to simulate unavailable readiness
        rec = _rec()
        r = sh.resolve_adaptive_boost_control(
            rec,
            learning_mode_on=True, active_control_on=True,
        )
        # _enabled=False is caught by gate 1 (learning_mode_on=True but not self._enabled)
        assert r.blocking_reason == "learning_mode_off"
        assert r.approved_boost_offset_c == 0.0


# ── Hard release on gate change ───────────────────────────────────────────────

class TestHardRelease:
    """Turning off any gate must trigger lifecycle release with the right reason."""

    def _shadow_with_release_tracker(self):
        sh = _shadow()
        sh._releases: list[str] = []
        original = sh._release_boost_lifecycle_safe
        def _track(reason):
            sh._releases.append(reason)
            original(reason)
        sh._release_boost_lifecycle_safe = _track
        return sh

    async def test_learning_off_triggers_release(self):
        sh = self._shadow_with_release_tracker()
        await sh.async_setup()
        rec = _rec()
        sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=False, active_control_on=True)
        assert "learning_mode_off" in sh._releases

    async def test_active_control_off_triggers_release(self):
        sh = self._shadow_with_release_tracker()
        await sh.async_setup()
        rec = _rec()
        sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=True, active_control_on=False)
        assert "active_control_off" in sh._releases


# ── Gate isolation: adjust_recommendation_safe still blocked without resolver ─

class TestControlGateIsolation:
    """Mode gate (control_enabled) is non-bypassable by authorized_override."""

    async def test_shadow_mode_blocks_even_with_authorized_override(self):
        sh = _shadow(mode=LearningRuntimeMode.SHADOW)
        await sh.async_setup()
        rec = _rec()
        # authorized_override=True must NOT bypass the shadow-mode gate
        sh.adjust_recommendation_safe(rec, authorized_override=True)
        assert rec.get("le2_boost_adjusted") is not True
        assert rec.get("_boost_applied_c") is None

    async def test_shadow_mode_blocks_direct_call_without_override(self):
        sh = _shadow(mode=LearningRuntimeMode.SHADOW)
        await sh.async_setup()
        rec = _rec()
        sh.adjust_recommendation_safe(rec)
        assert rec.get("le2_boost_adjusted") is not True
        assert rec.get("_boost_applied_c") is None

    async def test_no_residual_authorization_after_resolver_in_shadow_mode(self):
        sh = _shadow(mode=LearningRuntimeMode.SHADOW)
        await sh.async_setup()
        rec = _rec()
        sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=True, active_control_on=True)
        rec2 = _rec()
        sh.adjust_recommendation_safe(rec2, authorized_override=True)
        assert rec2.get("le2_boost_adjusted") is not True
        assert rec2.get("_boost_applied_c") is None


# ── Resolver result invariants ────────────────────────────────────────────────

class TestResolverResultInvariants:
    """approved_boost_offset_c and applied_boost_offset_c are always 0.0 when blocked."""

    async def test_blocked_result_never_has_none_applied(self):
        sh = _shadow()
        await sh.async_setup()
        rec = _rec()
        for learning, active in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            r = sh.resolve_adaptive_boost_control(
                rec, learning_mode_on=learning,
                active_control_on=active,
            )
            assert r.approved_boost_offset_c == 0.0, f"gate {learning}/{active}"
            assert r.applied_boost_offset_c == 0.0, f"gate {learning}/{active}"
            assert r.boost_allowed is False

    async def test_result_is_frozen_dataclass(self):
        sh = _shadow()
        await sh.async_setup()
        rec = _rec()
        r = sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=False, active_control_on=False)
        assert isinstance(r, AdaptiveBoostControlResult)
        with pytest.raises(Exception):  # frozen dataclass → attribute assignment must fail
            r.boost_allowed = True  # type: ignore[misc]

    async def test_never_raises(self):
        sh = _shadow()
        await sh.async_setup()
        # Poison the recommendation with garbage
        rec = {"effective_target": "not_a_number", "trv_setpoint": None}
        # Must not raise — returns typed zero result
        r = sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=True, active_control_on=True)
        assert isinstance(r, AdaptiveBoostControlResult)


# ── Coordinator-level integration ─────────────────────────────────────────────

class TestCoordinatorIntegration:
    """resolve_adaptive_boost_control is called every coordinator cycle."""

    async def test_update_data_uses_resolver_not_direct_adjust(self):
        """Coordinator calls resolve, not adjust_recommendation_safe directly."""
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        coord._active_control = True

        resolved_calls: list = []
        original = sh.resolve_adaptive_boost_control
        def _track(*a, **kw):
            resolved_calls.append(kw)
            return original(*a, **kw)
        sh.resolve_adaptive_boost_control = _track

        await coord._async_update_data()
        assert len(resolved_calls) == 1, "resolver must be called exactly once per cycle"
        assert "adaptive_boost_switch_on" not in resolved_calls[0]

    async def test_update_data_passes_active_control_state(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        coord._active_control = True

        captured: list = []
        original = sh.resolve_adaptive_boost_control
        def _capture(*a, **kw):
            captured.append(kw)
            return original(*a, **kw)
        sh.resolve_adaptive_boost_control = _capture

        await coord._async_update_data()
        assert captured[0]["active_control_on"] is True

    async def test_update_data_active_control_off(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        coord._active_control = False

        captured: list = []
        original = sh.resolve_adaptive_boost_control
        def _capture(*a, **kw):
            captured.append(kw)
            return original(*a, **kw)
        sh.resolve_adaptive_boost_control = _capture

        await coord._async_update_data()
        assert captured[0]["active_control_on"] is False

    async def test_no_shadow_no_crash(self):
        """If no shadow, resolve is never called — no crash."""
        coord = make_recording_coordinator()
        coord._active_control = True
        await coord._async_update_data()   # must not raise


# ── boost_offset_c attribute: 0.0 when boost blocked ─────────────────────────

class TestBoostOffsetAttributeZero:
    """boost_offset_c and boost_factor compat attributes remain neutral when blocked."""

    async def test_boost_offset_c_zero_when_learning_off(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        coord._active_control = True
        # zone_cfg has learning_enabled=True by default; override to False
        coord.entry.data = {**coord.entry.data, "learning_enabled": False}

        data = await coord._async_update_data()
        z = data["zone"]
        assert z.get("boost_offset_c", 0.0) == 0.0

    async def test_boost_factor_compat_is_one_when_no_boost(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()

        data = await coord._async_update_data()
        z = data["zone"]
        assert z.get("boost_factor", 1.0) == 1.0   # compat attribute: 1.0 = neutral
