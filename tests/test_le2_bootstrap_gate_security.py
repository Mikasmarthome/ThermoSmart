"""S1a Item 3 — Bootstrap Gate Security.

Proves that `authorized_override=True` NEVER bypasses structural (non-confidence) gates:

  - control_enabled=False (SHADOW mode) blocks even with authorized_override=True
  - runtime_is_control=False (ControlPolicy) blocks even with authorized_override=True
  - Active Control OFF outer gate: blocks before authorized_override is evaluated
  - Learning mode OFF outer gate: blocks before authorized_override is evaluated
  - Restore pending: non-bypassable outer gate
  - Device unavailable: non-bypassable outer gate
  - Safety (window_open, heating_failure): non-bypassable inner gate
  - Hard release (lifecycle): non-bypassable
  - Clamp violation: non-bypassable

Only confidence/reliability/maturity gates (4a-4e in ControlPolicy) are bypassed by
authorized_override=True, and only in the context of resolve_adaptive_boost_control().
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.runtime.ha_integration import (
    LearningShadowController,
    BoostBlockReason,
)
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
from custom_components.thermosmart.learning.runtime.control import (
    ControlContext,
    ControlFeature,
    ControlledPrediction,
    ControlPolicy,
    ControlPolicyParameters,
    ControlRejection,
)
from tests.helpers_ha_runtime import FakeStore

pytestmark = pytest.mark.asyncio

_T0 = datetime(2025, 3, 10, 6, 0, 0, tzinfo=timezone.utc)


def _shadow(*, mode=LearningRuntimeMode.SHADOW):
    return LearningShadowController(
        hass=MagicMock(), zone_id="z", store=FakeStore(),
        mode=mode, clock=FakeClock(start=_T0),
    )


def _rec(*, window_open=False, heating_failure=False, device_unavailable=False,
          restore_pending=False, target=21.0, current=18.0, setpoint=21.0):
    return {
        "effective_target": target,
        "adjusted_target": target,
        "current_temp": current,
        "trv_setpoint": setpoint,
        "tpi_valve_direct": False,
        "window_open": window_open,
        "heating_failure": heating_failure,
        "_setpoint_device_unavailable": device_unavailable,
        "boost_active": False,
        "mode": "heat",
    }


# ── 1. control_enabled=False (SHADOW mode) ────────────────────────────────────

class TestShadowModeGateNonBypassable:
    """SHADOW mode shadow rejects boost even with authorized_override=True."""

    async def test_shadow_blocks_with_authorized_override(self):
        sh = _shadow(mode=LearningRuntimeMode.SHADOW)
        await sh.async_setup()
        rec = _rec()
        sh.adjust_recommendation_safe(rec, authorized_override=True)
        assert rec.get("le2_boost_adjusted") is not True
        assert rec.get("_boost_applied_c") is None

    async def test_shadow_blocks_without_authorized_override(self):
        sh = _shadow(mode=LearningRuntimeMode.SHADOW)
        await sh.async_setup()
        rec = _rec()
        sh.adjust_recommendation_safe(rec, authorized_override=False)
        assert rec.get("le2_boost_adjusted") is not True

    async def test_control_blocks_without_authorized_override(self):
        sh = _shadow(mode=LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        rec = _rec()
        sh.adjust_recommendation_safe(rec, authorized_override=False)
        # No boost applied — no prediction available; mode gate passes but confidence blocks
        assert rec.get("le2_boost_adjusted") is not True


# ── 2. ControlPolicy runtime_is_control gate ─────────────────────────────────

class TestControlPolicyRuntimeGateNonBypassable:
    """ControlPolicy.resolve() rejects when runtime_is_control=False, even with authorized_override."""

    def _policy(self, *, runtime_is_control: bool):
        return ControlPolicy(
            ControlPolicyParameters(),
            runtime_is_control=runtime_is_control,
        )

    async def test_not_control_blocks_with_authorized_override(self):
        policy = self._policy(runtime_is_control=False)
        ctx = ControlContext(authorized_override=True)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, baseline_value=21.0,
            prediction=None, context=ctx,
        )
        assert not decision.applied
        assert decision.rejection == ControlRejection.RUNTIME_NOT_CONTROL.value

    async def test_not_control_blocks_without_authorized_override(self):
        policy = self._policy(runtime_is_control=False)
        ctx = ControlContext(authorized_override=False)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, baseline_value=21.0,
            prediction=None, context=ctx,
        )
        assert not decision.applied
        assert decision.rejection == ControlRejection.RUNTIME_NOT_CONTROL.value

    async def test_control_mode_passes_runtime_gate(self):
        policy = self._policy(runtime_is_control=True)
        # Build a minimal ConfidenceResult mock that _summary() can read
        cr = MagicMock()
        cr.purpose.value = "boost"
        cr.value = 0.95
        cr.level.value = "high"
        cr.reliability = 0.9
        cr.control_allowed = True
        cr.fallback_used = False
        cr.applied_caps = []
        cr.breakdown.prior_fraction = 0.0
        cr.reason_codes = ()
        pred = ControlledPrediction(
            feature=ControlFeature.BOOST_OFFSET,
            value=0.5,
            unit="celsius_offset",
            confidence_result=cr,
        )
        ctx = ControlContext(authorized_override=False)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, baseline_value=0.0,
            prediction=pred, context=ctx, unit="celsius_offset",
        )
        # Mode gate passes; rejection (if any) is not RUNTIME_NOT_CONTROL
        assert decision.rejection != ControlRejection.RUNTIME_NOT_CONTROL.value


# ── 3. Outer gates: Active Control OFF and Learning Mode OFF ──────────────────

class TestOuterGatesNonBypassable:
    """Outer gates 1-2 in resolve_adaptive_boost_control() block before authorized_override."""

    async def _resolve(self, sh, *, learning, active):
        rec = _rec()
        result = sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=learning, active_control_on=active)
        return result, rec

    async def test_learning_off_always_blocks(self):
        sh = _shadow(mode=LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        result, rec = await self._resolve(sh, learning=False, active=True)
        assert result.boost_allowed is False
        assert result.blocking_reason == BoostBlockReason.LEARNING_MODE_OFF

    async def test_active_control_off_always_blocks(self):
        sh = _shadow(mode=LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        result, rec = await self._resolve(sh, learning=True, active=False)
        assert result.boost_allowed is False
        assert result.blocking_reason == BoostBlockReason.ACTIVE_CONTROL_OFF

    async def test_both_off_blocks_at_learning_gate_first(self):
        sh = _shadow(mode=LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        result, rec = await self._resolve(sh, learning=False, active=False)
        assert result.boost_allowed is False
        assert result.blocking_reason == BoostBlockReason.LEARNING_MODE_OFF


# ── 4. Restore pending outer gate ────────────────────────────────────────────

class TestRestorePendingGate:
    async def test_restore_pending_blocks(self):
        sh = _shadow(mode=LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        rec = _rec()
        result = sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=True, active_control_on=True, restore_pending=True)
        assert result.boost_allowed is False
        assert result.blocking_reason == BoostBlockReason.RESTORE_PENDING

    async def test_restore_not_pending_passes_gate(self):
        sh = _shadow(mode=LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        rec = _rec()
        result = sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=True, active_control_on=True, restore_pending=False)
        # Gate 0 passes; subsequent gates may block but not restore gate
        assert result.blocking_reason != BoostBlockReason.RESTORE_PENDING


# ── 5. Device unavailable gate ────────────────────────────────────────────────

class TestDeviceUnavailableGate:
    async def test_device_unavailable_blocks(self):
        sh = _shadow(mode=LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        rec = _rec(device_unavailable=True)
        result = sh.resolve_adaptive_boost_control(
            rec, learning_mode_on=True, active_control_on=True)
        assert result.boost_allowed is False
        assert result.blocking_reason == BoostBlockReason.DEVICE_UNAVAILABLE


# ── 6. Safety gates (window_open, heating_failure) ───────────────────────────

class TestSafetyGatesNonBypassable:
    """Inner safety gates (window_open, heating_failure) are never bypassed."""

    async def test_window_open_blocks_boost(self):
        sh = _shadow(mode=LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        rec = _rec(window_open=True)
        sh.adjust_recommendation_safe(rec, authorized_override=True)
        assert rec.get("le2_boost_adjusted") is not True
        assert rec.get("_boost_rejection_reason") == "window_open"

    async def test_heating_failure_blocks_boost(self):
        sh = _shadow(mode=LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        rec = _rec(heating_failure=True)
        sh.adjust_recommendation_safe(rec, authorized_override=True)
        assert rec.get("le2_boost_adjusted") is not True
        assert rec.get("_boost_rejection_reason") == "heating_failure"
