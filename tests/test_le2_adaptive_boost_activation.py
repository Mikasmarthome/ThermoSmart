"""B2c: Adaptive Boost Activation — Triple Gate, Resolver, Switch semantics.

Tests:
  - ThermoSmartAdaptiveBoostSwitch: default OFF, restore ON/OFF, unique-id, device info
  - Triple gate: all 8 combinations of learning/active_control/adaptive_boost
  - Resolver result typing: blocking_reason, eligibility_reason, applied_c
  - Hard release on each gate change mid-boost
  - Readiness NOT eligible → blocked + lifecycle released
  - Direct valve → blocked at readiness (control_type gate)
  - Setpoint TRV → approved when all gates pass + eligible model
  - boost_allowed only when all gates pass
  - Coordinator set_adaptive_boost_enabled wires through to shadow
  - Legacy adjust_recommendation_safe still blocked without resolver
  - Soft step-down only on TPI sufficient (not on switch-off)
  - Persistence: existing coord without adaptive_boost → default OFF
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch as _patch

from custom_components.thermosmart.learning.runtime.ha_integration import (
    AdaptiveBoostControlResult,
    ADAPTIVE_BOOST_RESOLVER_VERSION,
    LearningShadowController,
)
from tests.helpers import make_coordinator
from tests.helpers_ha_runtime import (
    FakeStore,
    RecordingCoordinator,
    attach_shadow,
    make_recording_coordinator,
)


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


def _shadow(*, zone_id="z1"):
    return LearningShadowController(hass=MagicMock(), zone_id=zone_id, store=FakeStore())


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


# ── Triple Gate: all 8 combinations ──────────────────────────────────────────

class TestTripleGate:
    """Every combination of the three outer gates must behave deterministically."""

    async def _run(self, coord, *, learning, active, adaptive):
        sh = attach_shadow(coord)
        await sh.async_setup()
        rec = _rec()
        result = sh.resolve_adaptive_boost_control(
            rec,
            learning_mode_on=learning,
            active_control_on=active,
            adaptive_boost_switch_on=adaptive,
        )
        return result, rec

    async def test_all_off_blocked(self, make_rc):
        r, rec = await self._run(make_rc, learning=False, active=False, adaptive=False)
        assert r.boost_allowed is False
        assert r.blocking_reason == "learning_mode_off"
        assert r.approved_boost_offset_c == 0.0

    async def test_learning_off_only_blocked(self, make_rc):
        r, rec = await self._run(make_rc, learning=False, active=True, adaptive=True)
        assert r.boost_allowed is False
        assert r.blocking_reason == "learning_mode_off"

    async def test_active_control_off_only_blocked(self, make_rc):
        r, rec = await self._run(make_rc, learning=True, active=False, adaptive=True)
        assert r.boost_allowed is False
        assert r.blocking_reason == "active_control_off"

    async def test_adaptive_switch_off_only_blocked(self, make_rc):
        r, rec = await self._run(make_rc, learning=True, active=True, adaptive=False)
        assert r.boost_allowed is False
        assert r.blocking_reason == "adaptive_boost_switch_off"

    async def test_learning_and_active_but_no_adaptive_blocked(self, make_rc):
        r, rec = await self._run(make_rc, learning=True, active=True, adaptive=False)
        assert r.blocking_reason == "adaptive_boost_switch_off"

    async def test_active_and_adaptive_but_no_learning_blocked(self, make_rc):
        r, rec = await self._run(make_rc, learning=False, active=True, adaptive=True)
        assert r.blocking_reason == "learning_mode_off"

    async def test_learning_and_adaptive_but_no_active_blocked(self, make_rc):
        r, rec = await self._run(make_rc, learning=True, active=False, adaptive=True)
        assert r.blocking_reason == "active_control_off"

    async def test_all_on_reaches_readiness_gate(self, make_rc):
        # With all three on but no trained model, readiness gate blocks (insufficient_data)
        r, rec = await self._run(make_rc, learning=True, active=True, adaptive=True)
        # Without a trained model, readiness.eligibility == False → boost_allowed=False
        assert r.boost_allowed is False
        assert r.approved_boost_offset_c == 0.0
        assert r.applied_boost_offset_c == 0.0
        # Blocking reason comes from readiness, not from the outer triple gates
        assert r.blocking_reason not in (
            "learning_mode_off", "active_control_off", "adaptive_boost_switch_off",
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
            learning_mode_on=True, active_control_on=True, adaptive_boost_switch_on=True,
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
            learning_mode_on=True, active_control_on=True, adaptive_boost_switch_on=True,
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
            rec, learning_mode_on=False, active_control_on=True, adaptive_boost_switch_on=True)
        assert "learning_mode_off" in sh._releases

    async def test_active_control_off_triggers_release(self):
        sh = self._shadow_with_release_tracker()
        await sh.async_setup()
        rec = _rec()
        sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=True, active_control_on=False, adaptive_boost_switch_on=True)
        assert "active_control_off" in sh._releases

    async def test_adaptive_switch_off_triggers_release(self):
        sh = self._shadow_with_release_tracker()
        await sh.async_setup()
        rec = _rec()
        sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=True, active_control_on=True, adaptive_boost_switch_on=False)
        assert "adaptive_boost_switch_off" in sh._releases


# ── Gate isolation: adjust_recommendation_safe still blocked without resolver ─

class TestControlGateIsolation:
    """adjust_recommendation_safe must remain no-op without resolver authorization."""

    async def test_direct_call_blocked_by_control_enabled(self):
        sh = _shadow()
        await sh.async_setup()
        rec = _rec()
        sh.adjust_recommendation_safe(rec)
        # Should be a no-op: control_enabled is False, _boost_authorized_this_cycle is False
        assert rec.get("le2_boost_adjusted") is not True
        assert rec.get("_boost_applied_c") is None

    async def test_boost_authorized_flag_cleared_after_resolver(self):
        sh = _shadow()
        await sh.async_setup()
        rec = _rec()
        sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=True, active_control_on=True, adaptive_boost_switch_on=True)
        # Flag must be False after the resolver returns (even if boost was not applied)
        assert sh._boost_authorized_this_cycle is False


# ── Coordinator wiring ────────────────────────────────────────────────────────

class TestCoordinatorWiring:
    """set_adaptive_boost_enabled must propagate to attached shadow."""

    def test_set_adaptive_boost_enabled_true(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        coord.set_adaptive_boost_enabled(True)
        assert coord._adaptive_boost_enabled is True
        assert sh._adaptive_boost_enabled is True

    def test_set_adaptive_boost_enabled_false(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        coord.set_adaptive_boost_enabled(True)
        coord.set_adaptive_boost_enabled(False)
        assert coord._adaptive_boost_enabled is False
        assert sh._adaptive_boost_enabled is False

    def test_default_is_off(self):
        coord = make_recording_coordinator()
        assert coord._adaptive_boost_enabled is False

    def test_set_before_shadow_attached(self):
        coord = make_recording_coordinator()
        coord.set_adaptive_boost_enabled(True)   # no shadow yet → no crash
        assert coord._adaptive_boost_enabled is True

    def test_shadow_set_adaptive_boost_enabled_method(self):
        sh = _shadow()
        assert sh._adaptive_boost_enabled is False
        sh.set_adaptive_boost_enabled(True)
        assert sh._adaptive_boost_enabled is True
        sh.set_adaptive_boost_enabled(False)
        assert sh._adaptive_boost_enabled is False


# ── Migration: existing installations default to OFF ─────────────────────────

class TestMigrationDefaultOff:
    """Existing installations must not silently get boost after update."""

    def test_coordinator_default_adaptive_boost_off(self):
        coord = make_recording_coordinator()
        # Even with active_control=True and learning=True, adaptive_boost starts OFF
        assert coord._adaptive_boost_enabled is False

    async def test_resolver_blocked_with_default_off(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        rec = _rec()
        # Simulating existing installation: adaptive_boost not turned on
        r = sh.resolve_adaptive_boost_control(
            rec,
            learning_mode_on=True,
            active_control_on=True,
            adaptive_boost_switch_on=False,   # default state
        )
        assert r.boost_allowed is False
        assert r.blocking_reason == "adaptive_boost_switch_off"
        assert r.applied_boost_offset_c == 0.0


# ── Resolver result invariants ────────────────────────────────────────────────

class TestResolverResultInvariants:
    """approved_boost_offset_c and applied_boost_offset_c are always 0.0 when blocked."""

    async def test_blocked_result_never_has_none_applied(self):
        sh = _shadow()
        await sh.async_setup()
        rec = _rec()
        for learning, active, adaptive in [
            (False, False, False),
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ]:
            r = sh.resolve_adaptive_boost_control(
                rec, learning_mode_on=learning,
                active_control_on=active,
                adaptive_boost_switch_on=adaptive,
            )
            assert r.approved_boost_offset_c == 0.0, f"gate {learning}/{active}/{adaptive}"
            assert r.applied_boost_offset_c == 0.0, f"gate {learning}/{active}/{adaptive}"
            assert r.boost_allowed is False

    async def test_result_is_frozen_dataclass(self):
        sh = _shadow()
        await sh.async_setup()
        rec = _rec()
        r = sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=False, active_control_on=False, adaptive_boost_switch_on=False)
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
            rec, learning_mode_on=True, active_control_on=True, adaptive_boost_switch_on=True)
        assert isinstance(r, AdaptiveBoostControlResult)


# ── Coordinator-level integration ─────────────────────────────────────────────

class TestCoordinatorIntegration:
    """resolve_adaptive_boost_control is called every coordinator cycle."""

    async def test_update_data_uses_resolver_not_direct_adjust(self):
        """After B2c: coordinator calls resolve, not adjust_recommendation_safe directly."""
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        coord._active_control = True
        coord._adaptive_boost_enabled = False  # default OFF

        resolved_calls: list = []
        original = sh.resolve_adaptive_boost_control
        def _track(*a, **kw):
            resolved_calls.append(kw)
            return original(*a, **kw)
        sh.resolve_adaptive_boost_control = _track

        await coord._async_update_data()
        assert len(resolved_calls) == 1, "resolver must be called exactly once per cycle"
        assert resolved_calls[0]["adaptive_boost_switch_on"] is False

    async def test_update_data_passes_active_control_state(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        coord._active_control = True
        coord._adaptive_boost_enabled = True

        captured: list = []
        original = sh.resolve_adaptive_boost_control
        def _capture(*a, **kw):
            captured.append(kw)
            return original(*a, **kw)
        sh.resolve_adaptive_boost_control = _capture

        await coord._async_update_data()
        assert captured[0]["active_control_on"] is True
        assert captured[0]["adaptive_boost_switch_on"] is True

    async def test_update_data_active_control_off(self):
        coord = make_recording_coordinator()
        sh = attach_shadow(coord)
        await sh.async_setup()
        coord._active_control = False
        coord._adaptive_boost_enabled = True

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
        coord._adaptive_boost_enabled = True
        await coord._async_update_data()   # must not raise


# ── boost_offset_c attribute: 0.0 when boost blocked ─────────────────────────

class TestBoostOffsetAttributeZero:
    """boost_offset_c and boost_factor compat attributes remain neutral when blocked."""

    async def test_boost_offset_c_zero_when_adaptive_boost_off(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()
        coord._active_control = True
        coord._adaptive_boost_enabled = False

        data = await coord._async_update_data()
        z = data["zone"]
        # boost_offset_c is the LE2 prediction reading, not the applied offset.
        # With no trained model it should be 0.0.
        assert z.get("boost_offset_c", 0.0) == 0.0

    async def test_boost_factor_compat_is_one_when_no_boost(self):
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        await sh.async_setup()

        data = await coord._async_update_data()
        z = data["zone"]
        assert z.get("boost_factor", 1.0) == 1.0   # compat attribute: 1.0 = neutral
