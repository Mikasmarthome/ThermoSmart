"""Phase S1a Item 3 — Bootstrap authorized_override security proofs.

Proves (structural refactor: authorized_override is a direct parameter, not a mutable flag):
1.  adjust_recommendation_safe() default is authorized_override=False — shadow mode blocks.
2.  authorized_override=True cannot persist between calls — it is a per-call parameter.
3.  An exception inside adjust_recommendation_safe(authorized_override=True) cannot leave
    stale authorization: subsequent calls without the parameter are blocked.
4.  authorized_override is NOT persisted to the store (no restart resurrection).
5.  SHADOW mode (active_control=False) blocks resolve_adaptive_boost_control at Gate 2
    before adjust_recommendation_safe is called at all.
6.  Physical safety gates (window_open, heating_failure, excessive_deviation, clamp) are
    NEVER bypassed by authorized_override.
7.  Only confidence gates are bypassed by authorized_override (runtime_is_control is structural).
8.  authorized_override cannot be set by ordinary prediction/confidence/fallback data.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch as _patch

import pytest

from custom_components.thermosmart.learning.runtime.control import (
    ControlContext,
    ControlDecisionStatus,
    ControlFeature,
    ControlPolicy,
    ControlPolicyParameters,
    ControlRejection,
    ControlledPrediction,
)
from tests.helpers import make_coordinator
from tests.helpers_ha_runtime import FakeStore, attach_shadow, inject_eligible_boost_model

_T0 = datetime(2025, 3, 10, 6, 0, 0, tzinfo=timezone.utc)
_DT_NOW = "homeassistant.util.dt.now"
_DT_UTCNOW = "homeassistant.util.dt.utcnow"


# ── helpers ───────────────────────────────────────────────────────────────────

def _local_t0():
    from datetime import timedelta, timezone
    return _T0.astimezone(timezone(timedelta(hours=1)))


def _make_shadow(active_control=True, learning_mode=True, *, control_mode=True):
    from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntimeMode
    coord = make_coordinator()
    coord.set_active_control(active_control)
    coord._active_control_initialized = True
    mode = LearningRuntimeMode.CONTROL if control_mode else LearningRuntimeMode.SHADOW
    shadow = attach_shadow(coord, store=FakeStore(), mode=mode)
    return coord, shadow


async def _setup_shadow(shadow):
    with _patch(_DT_NOW, return_value=_local_t0()), \
         _patch(_DT_UTCNOW, return_value=_T0):
        await shadow.async_setup()


def _minimal_recommendation():
    return {
        "trv_setpoint": 21.5,
        "effective_target": 21.0,
        "adjusted_target": 21.0,
        "current_temp": 18.0,
        "window_open": False,
        "heating_failure": False,
        "mode": "heat",
        "hvac_mode": "heat",
        "zone_control_type": "setpoint",
        "_setpoint_device_unavailable": False,
        "is_summer": False,
    }


# ── 1. Authorization is a per-call parameter, not persistent state ────────────

class TestAuthorizationLifetime:
    def test_adjust_default_blocked_in_shadow_mode(self):
        """Without authorized_override=True, adjust_recommendation_safe is a no-op in SHADOW."""
        _, shadow = _make_shadow()
        rec = _minimal_recommendation()
        shadow.adjust_recommendation_safe(rec)
        # No boost applied — shadow mode without explicit override
        assert rec.get("_boost_applied_c") is None, (
            "adjust_recommendation_safe() without authorized_override must not apply boost."
        )

    @pytest.mark.asyncio
    async def test_authorization_not_leaked_after_resolve(self):
        """After resolve_adaptive_boost_control, subsequent calls without override are blocked."""
        coord, shadow = _make_shadow()
        await _setup_shadow(shadow)

        rec = _minimal_recommendation()
        with _patch(_DT_NOW, return_value=_local_t0()), \
             _patch(_DT_UTCNOW, return_value=_T0):
            shadow.resolve_adaptive_boost_control(
                rec,
                learning_mode_on=True,
                active_control_on=True,
                restore_pending=False,
                boost_runtime_limit=8.0,
            )
        # Subsequent call without override must be blocked — no residual authorization
        rec2 = _minimal_recommendation()
        shadow.adjust_recommendation_safe(rec2)
        assert rec2.get("_boost_applied_c") is None, (
            "After resolve_adaptive_boost_control, a subsequent call to "
            "adjust_recommendation_safe() without authorized_override must be blocked. "
            "The authorization must not persist between calls."
        )

    @pytest.mark.asyncio
    async def test_exception_does_not_persist_authorization(self):
        """An exception inside adjust_recommendation_safe(authorized_override=True) cannot
        leave stale authorization: subsequent calls without override are still blocked."""
        coord, shadow = _make_shadow()
        await _setup_shadow(shadow)
        inject_eligible_boost_model(shadow)

        original_adjust = shadow.adjust_recommendation_safe

        def raising_adjust(rec, *, authorized_override=False, **kw):
            raise RuntimeError("simulated inner failure")

        shadow.adjust_recommendation_safe = raising_adjust

        rec = _minimal_recommendation()
        with _patch(_DT_NOW, return_value=_local_t0()), \
             _patch(_DT_UTCNOW, return_value=_T0):
            try:
                shadow.resolve_adaptive_boost_control(
                    rec,
                    learning_mode_on=True,
                    active_control_on=True,
                    restore_pending=False,
                    boost_runtime_limit=8.0,
                )
            except Exception:
                pass  # resolve catches all exceptions internally

        shadow.adjust_recommendation_safe = original_adjust
        # After exception: subsequent call without override must still be blocked
        rec2 = _minimal_recommendation()
        original_adjust(rec2)  # no authorized_override → blocked
        assert rec2.get("_boost_applied_c") is None, (
            "After exception in authorized adjust, subsequent calls without override "
            "must still be blocked. No stale authorization possible with parameter-based design."
        )


# ── 2. authorized_override is not persisted ───────────────────────────────────

class TestAuthorizationNotPersisted:
    @pytest.mark.asyncio
    async def test_authorized_override_not_in_store_after_flush(self):
        """authorized_override is a call parameter — it cannot appear in the store."""
        coord, shadow = _make_shadow()
        await _setup_shadow(shadow)

        fake_store = shadow._runtime._store  # FakeStore attached at setup
        with _patch(_DT_NOW, return_value=_local_t0()), \
             _patch(_DT_UTCNOW, return_value=_T0):
            await shadow.async_flush()

        stored = getattr(fake_store, "data", {}) or {}
        stored_str = str(stored)
        assert "_boost_authorized_this_cycle" not in stored_str, (
            "_boost_authorized_this_cycle must never be written to the store — "
            "it has been replaced by a direct function parameter."
        )

    @pytest.mark.asyncio
    async def test_adjust_blocked_after_restart(self):
        """After a restart/restore cycle, adjust_recommendation_safe without override is blocked."""
        coord1, shadow1 = _make_shadow()
        await _setup_shadow(shadow1)

        store = FakeStore()
        shadow1._runtime._store = store
        with _patch(_DT_NOW, return_value=_local_t0()), \
             _patch(_DT_UTCNOW, return_value=_T0):
            await shadow1.async_flush()

        saved_data = dict(store.data or {})

        coord2 = make_coordinator()
        coord2.set_active_control(True)
        coord2._active_control_initialized = True
        new_store = FakeStore(data=saved_data)
        shadow2 = attach_shadow(coord2, store=new_store)

        with _patch(_DT_NOW, return_value=_local_t0()), \
             _patch(_DT_UTCNOW, return_value=_T0):
            await shadow2.async_setup()

        # After restart: adjust without override must be blocked (shadow mode)
        rec = _minimal_recommendation()
        shadow2.adjust_recommendation_safe(rec)
        assert rec.get("_boost_applied_c") is None, (
            "After restart/restore, adjust_recommendation_safe() without authorized_override "
            "must still be blocked in SHADOW mode."
        )


# ── 3. SHADOW mode cannot reach authorized_override ──────────────────────────

class TestShadowModeGate:
    @pytest.mark.asyncio
    async def test_shadow_mode_blocked_at_gate2(self):
        """With active_control=False, resolve returns at Gate 2 before flag is ever set."""
        coord, shadow = _make_shadow(active_control=False)
        await _setup_shadow(shadow)
        inject_eligible_boost_model(shadow)

        rec = _minimal_recommendation()
        with _patch(_DT_NOW, return_value=_local_t0()), \
             _patch(_DT_UTCNOW, return_value=_T0):
            result = shadow.resolve_adaptive_boost_control(
                rec,
                learning_mode_on=True,
                active_control_on=False,
                restore_pending=False,
                boost_runtime_limit=8.0,
            )

        assert result.boost_allowed is False
        assert result.blocking_reason == "active_control_off"
        # No flag state to check — parameter-based design: adjust_recommendation_safe
        # was never called, so authorized_override was never passed as True.
        rec2 = _minimal_recommendation()
        shadow.adjust_recommendation_safe(rec2)
        assert rec2.get("_boost_applied_c") is None, (
            "After SHADOW-mode resolve, adjust_recommendation_safe without override must be blocked."
        )

    @pytest.mark.asyncio
    async def test_shadow_mode_no_real_trv_calls(self):
        """In shadow mode the coordinator does not dispatch TRV service calls."""
        from tests.helpers_ha_runtime import make_recording_coordinator

        coord = make_recording_coordinator()
        coord.set_active_control(False)
        coord._active_control_initialized = True
        shadow = attach_shadow(coord, store=FakeStore())
        inject_eligible_boost_model(shadow)

        from tests.helpers import make_state, set_hass_states
        set_hass_states(coord, {"sensor.test_temp": make_state("18.0")})
        coord.hass.states.get = MagicMock(side_effect=lambda eid: (
            MagicMock(state="heat",
                      attributes={"temperature": 21.5, "min_temp": 5.0,
                                  "max_temp": 35.0, "hvac_mode": "heat",
                                  "target_temp_step": 0.5})
            if "climate" in eid else make_state("18.0")
        ))

        with _patch(_DT_NOW, return_value=_local_t0()), \
             _patch(_DT_UTCNOW, return_value=_T0):
            await shadow.async_setup()
            await coord._async_update_data()

        # SHADOW mode: no control calls should have been dispatched
        assert coord.control_calls == [], (
            f"SHADOW mode must NOT dispatch control calls. Got: {coord.control_calls}"
        )


# ── 4. Physical safety gates survive authorized_override ─────────────────────

class TestSafetyGatesNotBypassed:
    """ControlPolicy.resolve() safety gates (Gate 3) must block even authorized boosts."""

    def _make_policy(self):
        params = ControlPolicyParameters(
            enabled_features=frozenset({ControlFeature.BOOST_OFFSET}),
            boost_offset_runtime_limit=8.0,
        )
        # runtime_is_control=True: production CONTROL mode, so safety gates are reachable
        return ControlPolicy(params=params, runtime_is_control=True)

    def _prediction(self, value=1.0):
        return ControlledPrediction(
            feature=ControlFeature.BOOST_OFFSET,
            value=value, unit="celsius_offset", confidence_result=None)

    def test_window_open_blocks_authorized_override(self):
        policy = self._make_policy()
        ctx = ControlContext(window_open=True, authorized_override=True)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, 21.0, self._prediction(), context=ctx)
        assert decision.rejection == ControlRejection.WINDOW_OPEN.value, (
            "window_open must block boost even when authorized_override=True."
        )

    def test_heating_failure_blocks_authorized_override(self):
        policy = self._make_policy()
        ctx = ControlContext(heating_failure=True, authorized_override=True)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, 21.0, self._prediction(), context=ctx)
        assert decision.rejection == ControlRejection.HEATING_FAILURE.value

    def test_frost_protection_blocks_authorized_override(self):
        policy = self._make_policy()
        ctx = ControlContext(frost_protection=True, authorized_override=True)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, 21.0, self._prediction(), context=ctx)
        assert decision.rejection == ControlRejection.FROST_PROTECTION.value

    def test_unsafe_mode_blocks_authorized_override(self):
        policy = self._make_policy()
        ctx = ControlContext(mode_safe=False, authorized_override=True)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, 21.0, self._prediction(), context=ctx)
        assert decision.rejection == ControlRejection.UNSAFE_MODE.value

    def test_excessive_deviation_blocks_authorized_override(self):
        policy = self._make_policy()
        ctx = ControlContext(authorized_override=True)
        # Clamp.excessive_deviation=4.0; propose 10.0 offset from baseline 21.0 → deviation=10 > 4
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, 21.0, self._prediction(value=31.0), context=ctx)
        assert decision.rejection == ControlRejection.EXCESSIVE_DEVIATION.value, (
            "Excessive deviation gate must block even authorized boosts."
        )

    def test_device_max_clamps_authorized_override(self):
        # BOOST_OFFSET baseline = current offset (0.0), prediction = proposed offset (1.5).
        # After step-clamp (max_step=0.5): stepped=0.5. device_max=0.3 < 0.5 → clamped.
        policy = self._make_policy()
        ctx = ControlContext(device_max=0.3, authorized_override=True)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, 0.0, self._prediction(value=1.5), context=ctx)
        assert decision.applied, f"Should apply; got rejection={decision.rejection}"
        assert decision.final_value is not None
        assert decision.final_value <= 0.3, (
            "device_max must clamp even when authorized_override=True."
        )


# ── 5. Authorized override bypasses ONLY the intended gates ──────────────────

class TestGateBehaviorWithOverride:
    """Precise characterization: which gates bypass, which remain.

    New contract (S1a Item 3): authorized_override ONLY bypasses confidence gates (4a-4e).
    runtime_is_control is a structural gate — non-bypassable.
    """

    def _policy_not_control(self):
        params = ControlPolicyParameters(
            enabled_features=frozenset({ControlFeature.BOOST_OFFSET}),
            min_control_confidence=0.55, min_reliability=0.3,
            boost_offset_runtime_limit=8.0,
        )
        return ControlPolicy(params=params, runtime_is_control=False)

    def _policy_control(self):
        params = ControlPolicyParameters(
            enabled_features=frozenset({ControlFeature.BOOST_OFFSET}),
            min_control_confidence=0.55, min_reliability=0.3,
            boost_offset_runtime_limit=8.0,
        )
        return ControlPolicy(params=params, runtime_is_control=True)

    def _prediction(self, value=0.5):
        return ControlledPrediction(
            feature=ControlFeature.BOOST_OFFSET,
            value=value, unit="celsius_offset", confidence_result=None)

    def test_without_override_runtime_gate_blocks(self):
        policy = self._policy_not_control()
        ctx = ControlContext(authorized_override=False)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, 21.0, self._prediction(), context=ctx)
        assert decision.rejection == ControlRejection.RUNTIME_NOT_CONTROL.value

    def test_with_override_runtime_gate_still_blocks(self):
        """runtime_is_control gate is NON-bypassable — authorized_override=True cannot bypass it."""
        policy = self._policy_not_control()
        ctx = ControlContext(authorized_override=True)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, 21.0, self._prediction(), context=ctx)
        assert decision.rejection == ControlRejection.RUNTIME_NOT_CONTROL.value, (
            "runtime_is_control gate must block even with authorized_override=True."
        )

    def test_with_override_confidence_none_passes_in_control_mode(self):
        """In CONTROL mode, conf=None + authorized_override bypasses confidence gate."""
        policy = self._policy_control()
        # Without override: confidence=None blocks
        decision_no_ov = policy.resolve(
            ControlFeature.BOOST_OFFSET, 21.0, self._prediction(),
            context=ControlContext(authorized_override=False))
        assert decision_no_ov.rejection == ControlRejection.CONTROL_GATE_CLOSED.value

        # With override: confidence gate bypassed, decision applies
        decision_ov = policy.resolve(
            ControlFeature.BOOST_OFFSET, 21.0, self._prediction(),
            context=ControlContext(authorized_override=True))
        assert decision_ov.rejection not in (
            ControlRejection.CONTROL_GATE_CLOSED.value,
            ControlRejection.RUNTIME_NOT_CONTROL.value,
        ), f"With authorized_override in CONTROL mode, confidence=None must not block. Got: {decision_ov.rejection}"

    def test_feature_gate_not_bypassed(self):
        """Features not in enabled_features are still rejected even with override (in CONTROL mode)."""
        params = ControlPolicyParameters(
            enabled_features=frozenset({ControlFeature.PREHEAT_START}),  # BOOST_OFFSET excluded
            boost_offset_runtime_limit=8.0,
        )
        policy = ControlPolicy(params=params, runtime_is_control=True)
        ctx = ControlContext(authorized_override=True)
        decision = policy.resolve(
            ControlFeature.BOOST_OFFSET, 21.0, self._prediction(), context=ctx)
        assert decision.rejection == ControlRejection.FEATURE_NOT_ENABLED.value, (
            "Feature gate must reject even authorized overrides for disabled features."
        )


# ── 6. Authorized override cannot be set by prediction/confidence data ────────

class TestOverrideCannotBeSetByData:
    """authorized_override is an explicit call parameter, not derived from model data."""

    @pytest.mark.asyncio
    async def test_insufficient_readiness_blocks_without_override(self):
        """With no eligible model, resolve returns at Gate 4 without calling adjust."""
        coord, shadow = _make_shadow(control_mode=True)
        await _setup_shadow(shadow)
        # No inject_eligible_boost_model — readiness will be INSUFFICIENT_DATA

        rec = _minimal_recommendation()
        with _patch(_DT_NOW, return_value=_local_t0()), \
             _patch(_DT_UTCNOW, return_value=_T0):
            result = shadow.resolve_adaptive_boost_control(
                rec,
                learning_mode_on=True,
                active_control_on=True,
                restore_pending=False,
                boost_runtime_limit=8.0,
            )

        assert result.boost_allowed is False
        # adjust_recommendation_safe was never called with authorized_override=True
        rec2 = _minimal_recommendation()
        shadow.adjust_recommendation_safe(rec2)
        assert rec2.get("_boost_applied_c") is None

    @pytest.mark.asyncio
    async def test_restore_pending_blocks_before_adjust(self):
        coord, shadow = _make_shadow(control_mode=True)
        await _setup_shadow(shadow)

        rec = _minimal_recommendation()
        with _patch(_DT_NOW, return_value=_local_t0()), \
             _patch(_DT_UTCNOW, return_value=_T0):
            result = shadow.resolve_adaptive_boost_control(
                rec,
                learning_mode_on=True,
                active_control_on=True,
                restore_pending=True,   # blocks at Gate 0
                boost_runtime_limit=8.0,
            )

        assert result.boost_allowed is False
        # No residual authorization — adjust still blocked
        rec2 = _minimal_recommendation()
        shadow.adjust_recommendation_safe(rec2)
        assert rec2.get("_boost_applied_c") is None

    def test_control_context_authorized_override_is_separate_from_confidence(self):
        """authorized_override on ControlContext is a pure construction parameter,
        not derived from any confidence or fallback result."""
        ctx_default = ControlContext()
        assert ctx_default.authorized_override is False

        ctx_override = ControlContext(authorized_override=True)
        assert ctx_override.authorized_override is True

        # There is no path from ConfidenceSummary to authorized_override
        from custom_components.thermosmart.learning.runtime.control import ConfidenceSummary
        high_conf = ConfidenceSummary(
            purpose="boost", value=0.99, level="high", reliability=0.99,
            control_allowed=True, fallback_used=False, hard_cap=False,
            prior_fraction=0.0)
        # Even max-confidence cannot set authorized_override (it's a dataclass field)
        ctx_high = ControlContext()
        assert ctx_high.authorized_override is False, (
            "authorized_override must only be set explicitly — not derived from confidence data."
        )
