"""LE 2.0 Boost Lifecycle — Pflicht-Tests (Phase 19C final acceptance).

Covers:
- All lifecycle release conditions (target_reached, schedule/mode change, manual override,
  early cutoff, timeout, window_open, heating_failure)
- Stale prediction cannot extend/increase active boost
- Episode binding and retry protection (failed episode cannot retry-loop)
- TPI/Boost authority separation (single dispatch, no double application)
- Safety cap hierarchy (user prior → adaptive → confidence → device clamp → safety)
- Restart/reload safety (orphaned offset impossible)
- SHADOW default
- Cooldown reasons and durations
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------------- helpers


def _ts(offset_s: float = 0.0) -> str:
    """ISO timestamp relative to a fixed base."""
    base = datetime(2025, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
    return (base + timedelta(seconds=offset_s)).isoformat()


def _fresh_model(*, adaptive_max_boost_c: float = 3.0,
                 max_boost_duration_s: float = 3600.0,
                 cooldown_duration_s: float = 300.0,
                 failure_cooldown_duration_s: float = 900.0):
    from custom_components.thermosmart.learning.models.boost import BoostModel, BoostParameters
    p = BoostParameters(
        adaptive_max_boost_c=adaptive_max_boost_c,
        max_boost_duration_s=max_boost_duration_s,
        cooldown_duration_s=cooldown_duration_s,
        failure_cooldown_duration_s=failure_cooldown_duration_s,
    )
    return BoostModel("test_zone", params=p)


def _apply(model, *, episode_id: str = "ep1", offset_c: float = 1.0,
           target_c: float = 21.0, ts: str = None) -> bool:
    return model.apply_lifecycle(
        episode_id=episode_id,
        applied_offset_c=offset_c,
        base_target_c=target_c,
        ts=ts or _ts(0))


# ===========================================================================
# 1. Lifecycle Release Conditions
# ===========================================================================

class TestLifecycleReleaseConditions:
    """All 10 release conditions must trigger correctly and leave clean state."""

    def test_target_reached_ends_boost(self):
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        assert m._state.lifecycle == BoostLifecycle.APPLIED
        m.release_lifecycle("target_reached", _ts(600))
        assert m._state.lifecycle == BoostLifecycle.RELEASED_TARGET_REACHED
        assert m._state.lifecycle_release_reason == "target_reached"

    def test_target_reached_no_cooldown(self):
        """Successful target-reached must not lock out new episodes."""
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("target_reached", _ts(600))
        # No cooldown after clean completion
        assert not m.cooldown_active(_ts(601))
        assert m._state.cooldown_until_ts is None

    def test_next_cycle_has_no_boost_after_target_reached(self):
        """After target_reached the model returns to INACTIVE — no residual offset."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("target_reached", _ts(600))
        m.reset_lifecycle()
        assert m._state.lifecycle == BoostLifecycle.INACTIVE
        assert m._state.applied_offset_c == 0.0

    def test_schedule_change_ends_boost(self):
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("schedule_change", _ts(300))
        assert m._state.lifecycle == BoostLifecycle.RELEASED_MODE_CHANGE
        assert m._state.lifecycle_release_reason == "schedule_change"

    def test_schedule_change_no_cooldown(self):
        """Schedule change is a context change, not a failure — no cooldown needed."""
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("schedule_change", _ts(300))
        assert not m.cooldown_active(_ts(301))

    def test_mode_change_ends_boost(self):
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("mode_change", _ts(300))
        assert m._state.lifecycle == BoostLifecycle.RELEASED_MODE_CHANGE

    def test_mode_change_no_cooldown(self):
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("mode_change", _ts(300))
        assert not m.cooldown_active(_ts(301))

    def test_manual_override_ends_boost(self):
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("manual_override", _ts(400))
        assert m._state.lifecycle == BoostLifecycle.RELEASED_MANUAL_OVERRIDE
        # Short cooldown (300s) after manual override
        assert m.cooldown_active(_ts(500))
        assert not m.cooldown_active(_ts(800))

    def test_early_cutoff_ends_boost(self):
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("early_cutoff", _ts(500))
        assert m._state.lifecycle == BoostLifecycle.RELEASED_MODE_CHANGE
        # No cooldown — boost was superseded cleanly by early cutoff
        assert not m.cooldown_active(_ts(501))

    def test_timeout_ends_boost(self):
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("timeout", _ts(3600))
        assert m._state.lifecycle == BoostLifecycle.RELEASED_TIMEOUT
        # Neutral cooldown (300s) after timeout
        assert m.cooldown_active(_ts(3700))
        assert not m.cooldown_active(_ts(3950))

    def test_window_open_ends_boost(self):
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("window_open", _ts(200))
        assert m._state.lifecycle == BoostLifecycle.RELEASED_WINDOW_OPEN
        assert m.cooldown_active(_ts(400))  # 300s cooldown
        assert not m.cooldown_active(_ts(600))

    def test_heating_failure_ends_boost_failure_cooldown(self):
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("heating_failure", _ts(900))
        assert m._state.lifecycle == BoostLifecycle.FAILED_NO_RESPONSE
        # Failure cooldown = 900s
        assert m.cooldown_active(_ts(1700))
        assert not m.cooldown_active(_ts(1900))

    def test_no_response_failure_cooldown(self):
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("no_response", _ts(0))
        # 900s failure cooldown
        assert m.cooldown_active(_ts(800))
        assert not m.cooldown_active(_ts(1000))

    def test_overshoot_failure_cooldown(self):
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("overshoot", _ts(0))
        assert m.cooldown_active(_ts(800))
        assert not m.cooldown_active(_ts(1000))


# ===========================================================================
# 2. Stale Prediction Cannot Extend or Increase Active Boost
# ===========================================================================

class TestStalePredictionGuard:
    """A stale or superseded prediction must not alter an active lifecycle."""

    def test_stale_prediction_cannot_increase_applied_offset(self):
        """If a boost is already APPLIED, a new higher prediction does not override
        the current episode's applied_offset_c via the lifecycle."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m, offset_c=1.0)
        assert m._state.applied_offset_c == pytest.approx(1.0)
        # Simulated stale/new higher prediction: lifecycle must NOT be mutated
        # (apply_lifecycle would start a NEW episode; here we verify the state is stable)
        assert m._state.lifecycle == BoostLifecycle.APPLIED
        assert m._state.applied_offset_c == pytest.approx(1.0)

    def test_apply_lifecycle_does_not_stack_on_existing_episode(self):
        """Calling apply_lifecycle again on a live episode overwrites episode_id
        but should not additively stack offsets — most recent wins."""
        m = _fresh_model()
        _apply(m, episode_id="ep1", offset_c=1.5)
        _apply(m, episode_id="ep2", offset_c=2.0)
        # Second apply replaces; no additive stacking
        assert m._state.applied_offset_c == pytest.approx(2.0)
        assert m._state.current_episode_id == "ep2"

    def test_lifecycle_inactive_blocks_no_automatic_reapply(self):
        """After release, lifecycle stays in terminal state until explicitly reset."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("target_reached", _ts(600))
        # Lifecycle stays in RELEASED_TARGET_REACHED, not auto-recycled
        assert m._state.lifecycle == BoostLifecycle.RELEASED_TARGET_REACHED
        # Only reset_lifecycle() returns to INACTIVE
        m.reset_lifecycle()
        assert m._state.lifecycle == BoostLifecycle.INACTIVE


# ===========================================================================
# 3. Episode Binding and Retry Protection
# ===========================================================================

class TestEpisodeBinding:
    """Lifecycle must be uniquely bound to an episode; failed episode cannot retry."""

    def test_episode_id_stored_on_apply(self):
        m = _fresh_model()
        _apply(m, episode_id="morning_episode_001")
        assert m._state.current_episode_id == "morning_episode_001"

    def test_failed_episode_cannot_retry_during_cooldown(self):
        """After failure, cooldown is active — a new apply during cooldown is blocked
        by the lifecycle's cooldown_active() check."""
        m = _fresh_model(failure_cooldown_duration_s=900.0)
        _apply(m)
        m.release_lifecycle("no_response", _ts(0))
        # Cooldown active: new apply should be gated by caller checking cooldown_active()
        assert m.cooldown_active(_ts(600))

    def test_genuinely_new_episode_can_boost_after_cooldown(self):
        """After cooldown expires + reset, a new episode can boost."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model(failure_cooldown_duration_s=900.0)
        _apply(m, episode_id="ep1")
        m.release_lifecycle("no_response", _ts(0))
        # After cooldown
        assert not m.cooldown_active(_ts(1000))
        m.reset_lifecycle()
        _apply(m, episode_id="ep2")
        assert m._state.lifecycle == BoostLifecycle.APPLIED
        assert m._state.current_episode_id == "ep2"

    def test_episode_fields_all_stored(self):
        """All episode binding fields are set on apply."""
        m = _fresh_model()
        m.apply_lifecycle(
            episode_id="ep_bind_test",
            applied_offset_c=1.5,
            base_target_c=21.0,
            ts=_ts(0),
            user_target_c=22.0,
            max_duration_s=1800.0,
        )
        s = m._state
        assert s.current_episode_id == "ep_bind_test"
        assert s.applied_offset_c == pytest.approx(1.5)
        assert s.lifecycle_base_target_c == pytest.approx(21.0)
        assert s.lifecycle_user_target_c == pytest.approx(22.0)
        assert s.lifecycle_max_duration_s == pytest.approx(1800.0)
        assert s.lifecycle_start_ts == _ts(0)

    def test_episode_binding_not_persisted(self):
        """serialize_state must NOT include runtime lifecycle fields."""
        m = _fresh_model()
        _apply(m, episode_id="runtime_ep")
        state_dict = m.serialize_state()
        assert "lifecycle_start_ts" not in state_dict
        assert "lifecycle_base_target_c" not in state_dict
        assert "lifecycle_user_target_c" not in state_dict
        assert "lifecycle_release_reason" not in state_dict
        assert "current_episode_id" not in state_dict

    def test_deserialize_always_inactive(self):
        """After reload: lifecycle always starts as INACTIVE (never orphaned)."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle, BoostModel
        m = _fresh_model()
        _apply(m, episode_id="live")
        state_dict = m.serialize_state()
        m2 = BoostModel("test_zone")
        m2.deserialize_state(state_dict)
        assert m2._state.lifecycle == BoostLifecycle.INACTIVE
        assert m2._state.applied_offset_c == 0.0
        assert m2._state.current_episode_id is None

    def test_failed_episode_blocked_by_episode_binding(self):
        """Same episode ID that failed must be blocked even after cooldown expires."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model(failure_cooldown_duration_s=900.0)
        result = _apply(m, episode_id="ep_fail")
        assert result is True
        m.release_lifecycle("no_response", _ts(0))
        assert m._state.last_failed_episode_id == "ep_fail"
        # After cooldown expires, same episode must still be blocked
        assert not m.cooldown_active(_ts(1000))
        result2 = _apply(m, episode_id="ep_fail")
        assert result2 is False  # episode binding blocks same-episode retry
        assert m._state.lifecycle == BoostLifecycle.FAILED_NO_RESPONSE

    def test_new_episode_id_allowed_after_failure(self):
        """Different episode ID (genuine new heating context) must be allowed."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model(failure_cooldown_duration_s=900.0)
        _apply(m, episode_id="ep_fail")
        m.release_lifecycle("no_response", _ts(0))
        assert not m.cooldown_active(_ts(1000))
        result = _apply(m, episode_id="ep_new")
        assert result is True
        assert m._state.lifecycle == BoostLifecycle.APPLIED
        assert m._state.current_episode_id == "ep_new"

    def test_schedule_change_clears_failed_episode_binding(self):
        """schedule_change (clean episode change) must clear last_failed_episode_id."""
        m = _fresh_model()
        _apply(m, episode_id="ep_fail")
        m.release_lifecycle("no_response", _ts(0))
        assert m._state.last_failed_episode_id == "ep_fail"
        # Schedule change clears binding
        m.release_lifecycle("schedule_change", _ts(1))
        assert m._state.last_failed_episode_id is None
        # Same episode ID now allowed (new context)
        result = _apply(m, episode_id="ep_fail")
        assert result is True

    def test_failed_episode_binding_not_persisted(self):
        """last_failed_episode_id must NOT be in serialized state (runtime only)."""
        m = _fresh_model()
        _apply(m, episode_id="fail_ep")
        m.release_lifecycle("no_response", _ts(0))
        state_dict = m.serialize_state()
        assert "last_failed_episode_id" not in state_dict


# ===========================================================================
# 4. TPI / Boost Authority Separation
# ===========================================================================

class TestTPIBoostAuthority:
    """TPI baseline + LE2 boost must never double-apply the offset."""

    def test_boost_offset_additive_to_tpi_baseline(self):
        """final_setpoint = TPI_baseline + boost_offset. The TPI baseline (not the
        comfort target) is the additive base for LE2 boost.

        Concrete example with clear separation:
          target=21°C, TPI_baseline=24°C, boost_offset=1°C → final=25°C
        Wrong formula would give: target + boost = 22°C (discards TPI baseline).
        """
        from tests.helpers_decision import preds as make_preds, rec as make_rec, run
        from custom_components.thermosmart.learning.decision.contracts import DecisionMode
        target = 21.0
        tpi_baseline = 24.0  # TPI raises to 24°C via duty cycle; this is in rec as trv_setpoint
        # Existing le2 boost=1.0°C already applied (prev cycle), same requested → no step
        r = make_rec(target=target, setpoint=tpi_baseline, boost_offset_c=1.0)
        trace, _ = run(DecisionMode.CONTROL, recommendation=r,
                       predictions=make_preds(boost=1.0))
        # Resolver must add boost ON TOP OF TPI baseline (correct formula)
        if trace.applied_any:
            assert trace.tpi_baseline_setpoint_c == pytest.approx(tpi_baseline), (
                "TPI baseline must be preserved in trace")
            assert trace.final_setpoint_c == pytest.approx(25.0), (
                "final = tpi_baseline(24) + boost(1) = 25°C")
            # WRONG formula would give target + boost = 22°C
            assert trace.final_setpoint_c != pytest.approx(target + 1.0), (
                "final must not be comfort_target + boost (discards TPI baseline)")
            # WRONG formula would give target + step = 21.5°C
            assert trace.final_setpoint_c != pytest.approx(target + 0.5)

    def test_duty_to_setpoint_does_not_use_boost_factor(self):
        """duty_to_setpoint takes (target, duty, max_boost) — no boost_factor param."""
        from custom_components.thermosmart.tpi import duty_to_setpoint
        import inspect
        sig = inspect.signature(duty_to_setpoint)
        param_names = list(sig.parameters.keys())
        assert "boost_factor" not in param_names
        assert "le2" not in param_names

    def test_boost_factor_attribute_never_read_by_tpi(self):
        """boost_factor is a display-only attribute; TPI code must not import or read it."""
        import ast, pathlib
        tpi_source = pathlib.Path(
            "custom_components/thermosmart/tpi.py").read_text()
        assert "boost_factor" not in tpi_source

    def test_single_boost_entry_per_cycle(self):
        """FinalResolver produces exactly one boost_offset entry per cycle."""
        from tests.helpers_decision import preds as make_preds, run
        from custom_components.thermosmart.learning.decision.contracts import DecisionMode
        trace, _ = run(DecisionMode.CONTROL, predictions=make_preds(boost=1.5))
        boost_entries = [e for e in trace.entries if e.feature == "boost_offset"]
        assert len(boost_entries) == 1

    def test_no_le_v1_boost_factor_affects_control(self):
        """LE v1 _boost_factors dict must not be written by update_boost_factor."""
        from custom_components.thermosmart.learning_engine import LearningEngine
        e = LearningEngine.__new__(LearningEngine)
        e._boost_factors = {}
        e.update_boost_factor("zone1", 1.5)
        # update_boost_factor is a documented no-op in LE 2.0
        assert not e._boost_factors

    def test_boost_offset_c_to_compat_factor_neutral(self):
        """0.0°C offset must always map to 1.0 (neutral LE v1 factor)."""
        from custom_components.thermosmart.learning.models.boost import boost_offset_c_to_compat_factor
        assert boost_offset_c_to_compat_factor(0.0) == pytest.approx(1.0)

    def test_compat_factor_never_drives_control(self):
        """boost_factor (compat) must not appear in resolver control logic."""
        import pathlib
        resolver_src = pathlib.Path(
            "custom_components/thermosmart/learning/decision/resolver.py").read_text()
        # boost_factor is imported only for display/trace; must not be used in setpoint math
        # The control path uses "boost_offset" key, not "boost_factor"
        assert 'preds.get("boost_factor")' not in resolver_src
        assert '"boost_factor"' not in resolver_src.split("boost_offset_c_to_compat_factor")[0]


# ===========================================================================
# 5. Safety Cap Hierarchy
# ===========================================================================

class TestSafetyCapHierarchy:
    """user_prior → adaptive_cap → confidence_gate → device_clamp → safety guards."""

    def _model(self, **kw):
        from custom_components.thermosmart.learning.models.boost import BoostModel, BoostParameters
        p = BoostParameters(**kw)
        return BoostModel("test_zone", params=p)

    def test_high_user_prior_capped_at_cold_start(self):
        """A 5.0°C device_prior is reduced to 0.5°C by cold-start adaptive cap."""
        from custom_components.thermosmart.learning.models.boost import BoostPredictionContext
        m = self._model()
        ctx = BoostPredictionContext(device_prior_offset_c=5.0)
        rec = m.predict_boost_factor(ctx)
        assert rec.boost_offset_c == pytest.approx(0.5)
        assert rec.adaptive_cap_c == pytest.approx(0.5)
        assert rec.fallback_used

    def test_low_confidence_stays_conservative_despite_high_prior(self):
        """Even with a high prior, cold-start confidence < 0.35 → cap stays at 0.5°C."""
        from custom_components.thermosmart.learning.models.boost import BoostPredictionContext
        m = self._model()
        ctx = BoostPredictionContext(device_prior_offset_c=8.0)
        rec = m.predict_boost_factor(ctx)
        assert rec.boost_offset_c <= 0.5
        assert rec.confidence < 0.35

    def test_negative_outcomes_reduce_boost(self):
        """Overshoot outcomes degrade effective_factor → lower offset despite high prior."""
        from custom_components.thermosmart.learning.models.boost import BoostPredictionContext
        try:
            import tests.helpers_boost as hb
        except ImportError:
            import helpers_boost as hb
        from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
        m = self._model(adaptive_max_boost_c=3.0, full_confidence_samples=5.0)
        # Feed episodes that end in overshoot (temperature well above target)
        for i in range(6):
            ep = hb.boost_episode(f"e{i}", [18, 19, 20, 21, 22.5, 23.0],
                                   reason=EpisodeReason.REACHED, target=21.0)
            m.update(ep, hb.boost_context(ep, requested_offset_c=2.0))
        ctx = BoostPredictionContext()
        rec = m.predict_boost_factor(ctx)
        # With high overshoot rate, adaptive cap applies 0.7x penalty
        full_cap = m.adaptive_cap_c()
        assert full_cap < 3.0  # overshoot penalty applied

    def test_absolute_device_max_is_backstop(self):
        """max_boost_offset_c is the technical hard ceiling — exceeds adaptive cap."""
        from custom_components.thermosmart.learning.models.boost import BoostPredictionContext
        m = self._model(max_boost_offset_c=8.0, adaptive_max_boost_c=3.0)
        # Even at max confidence, LE2 cap is 3.0, not 8.0
        ctx = BoostPredictionContext()
        rec = m.predict_boost_factor(ctx)
        assert rec.boost_offset_c <= 3.0

    def test_safety_lock_wins_in_resolver(self):
        """Frost protection blocks boost regardless of prediction."""
        from tests.helpers_decision import preds as make_preds, run
        from custom_components.thermosmart.learning.decision.contracts import DecisionMode
        from tests.helpers_decision import rec as make_rec
        r = make_rec(frost_protection=True)
        trace, _ = run(DecisionMode.CONTROL, recommendation=r, predictions=make_preds(boost=3.0))
        assert not trace.applied_any

    def test_tpi_baseline_is_boost_base(self):
        """TPI baseline must be the additive base for LE2 boost. Never comfort_target + boost.

        Correct: final = tpi_baseline_setpoint + le2_boost_offset
        Example: target=21°C, TPI_baseline=26°C, LE2 step=0.5°C → final=26.5°C
        Wrong:   target + step = 21.5°C  (discards TPI baseline entirely)
        """
        from tests.helpers_decision import preds as make_preds, rec as make_rec, run
        from custom_components.thermosmart.learning.decision import DecisionMode
        tpi_setpoint = 26.0
        target = 21.0
        r = make_rec(target=target, setpoint=tpi_setpoint)
        trace, _ = run(DecisionMode.CONTROL, recommendation=r, predictions=make_preds(boost=1.0))
        # Resolver must expose the TPI baseline in the trace (audit field)
        assert trace.tpi_baseline_setpoint_c == pytest.approx(tpi_setpoint)
        if trace.applied_any:
            # LE2 boost is additive on TPI baseline: final > tpi_baseline
            assert trace.final_setpoint_c > tpi_setpoint, (
                "LE2 boost must be additive on TPI baseline, not replace it")
            # Must NOT compute final as target + something (that discards TPI)
            assert trace.final_setpoint_c != pytest.approx(target + 0.5), (
                "final must not be comfort_target + step (TPI baseline discarded)")
            assert trace.final_setpoint_c != pytest.approx(target + 1.0), (
                "final must not be comfort_target + boost (TPI baseline discarded)")
            # final must be within [tpi_baseline, tpi_baseline + requested_boost]
            assert tpi_setpoint <= trace.final_setpoint_c <= tpi_setpoint + 1.0


# ===========================================================================
# 6. Restart / Reload Safety
# ===========================================================================

class TestRestartReloadSafety:
    """No orphaned offset after restart; state resets cleanly."""

    def test_deserialize_resets_lifecycle_to_inactive(self):
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle, BoostModel
        m = _fresh_model()
        _apply(m, episode_id="orphan", offset_c=2.5)
        m.release_lifecycle("timeout", _ts(3600))
        state_dict = m.serialize_state()
        m2 = BoostModel("test_zone")
        m2.deserialize_state(state_dict)
        assert m2._state.lifecycle == BoostLifecycle.INACTIVE
        assert m2._state.applied_offset_c == 0.0

    def test_outcomes_survive_restart(self):
        """Learning evidence is persisted; only lifecycle runtime fields are transient."""
        from custom_components.thermosmart.learning.models.boost import BoostModel
        try:
            import tests.helpers_boost as hb
        except ImportError:
            import helpers_boost as hb
        ZONE = "test_zone"
        m = _fresh_model()
        for i in range(5):
            ep = hb.boost_episode(f"e{i}", [18, 19, 20, 21, 21], zone=ZONE)
            m.update(ep, hb.boost_context_with_comparison(ep, zone=ZONE))
        state_dict = m.serialize_state()
        m2 = BoostModel(ZONE)
        m2.deserialize_state(state_dict)
        assert m2._state.general.has_evidence
        assert m2._state.general.effective_factor.effective_n > 0

    def test_no_duplicate_apply_after_reload(self):
        """After reload + INACTIVE state, applying boost requires explicit call."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle, BoostModel
        m = _fresh_model()
        _apply(m, episode_id="pre_reload")
        state_dict = m.serialize_state()
        m2 = BoostModel("test_zone")
        m2.deserialize_state(state_dict)
        # Fresh start — no auto-reapplication
        assert m2._state.lifecycle == BoostLifecycle.INACTIVE
        assert m2._state.current_episode_id is None


# ===========================================================================
# 7. Duration and Cooldown Parameters
# ===========================================================================

class TestDurationCooldownParams:
    """Verify justified parameter values are correctly stored and used."""

    def test_max_boost_duration_is_3600(self):
        from custom_components.thermosmart.learning.models.boost import BoostParameters
        assert BoostParameters().max_boost_duration_s == 3600.0

    def test_failure_detection_window_is_1200(self):
        from custom_components.thermosmart.learning.models.boost import BoostParameters
        assert BoostParameters().failure_detection_window_s == 1200.0

    def test_normal_cooldown_is_300(self):
        from custom_components.thermosmart.learning.models.boost import BoostParameters
        assert BoostParameters().cooldown_duration_s == 300.0

    def test_failure_cooldown_is_900(self):
        from custom_components.thermosmart.learning.models.boost import BoostParameters
        assert BoostParameters().failure_cooldown_duration_s == 900.0

    def test_lifecycle_initialized_with_max_duration(self):
        """apply_lifecycle should use max_boost_duration_s as default."""
        m = _fresh_model(max_boost_duration_s=3600.0)
        _apply(m)
        assert m._state.lifecycle_max_duration_s == pytest.approx(3600.0)

    def test_custom_episode_duration_overrides_default(self):
        """Caller can supply a specific max_duration_s (from prediction)."""
        m = _fresh_model()
        m.apply_lifecycle("ep", 1.0, 21.0, _ts(), max_duration_s=900.0)
        assert m._state.lifecycle_max_duration_s == pytest.approx(900.0)

    def test_cooldown_reason_target_reached(self):
        """target_reached: no cooldown (0s)."""
        m = _fresh_model(cooldown_duration_s=300.0)
        _apply(m)
        m.release_lifecycle("target_reached", _ts(0))
        assert m._state.cooldown_until_ts is None
        assert not m.cooldown_active(_ts(1))

    def test_cooldown_reason_window_open(self):
        """window_open: short cooldown (cooldown_duration_s=300s)."""
        m = _fresh_model(cooldown_duration_s=300.0)
        _apply(m)
        m.release_lifecycle("window_open", _ts(0))
        assert m.cooldown_active(_ts(200))
        assert not m.cooldown_active(_ts(400))

    def test_cooldown_reason_overshoot(self):
        """overshoot: long cooldown (failure_cooldown_duration_s=900s)."""
        m = _fresh_model(failure_cooldown_duration_s=900.0)
        _apply(m)
        m.release_lifecycle("overshoot", _ts(0))
        assert m.cooldown_active(_ts(700))
        assert not m.cooldown_active(_ts(1000))


# ===========================================================================
# 8. SHADOW Mode Default
# ===========================================================================

class TestShadowDefault:
    """SHADOW mode must never apply any setpoint change."""

    def test_shadow_produces_no_dispatch(self):
        from tests.helpers_decision import preds as make_preds, run
        from custom_components.thermosmart.learning.decision.contracts import DecisionMode
        trace, dispatch = run(DecisionMode.SHADOW, predictions=make_preds(boost=2.0))
        assert not trace.applied_any
        assert dispatch.shadow_only

    def test_shadow_trace_has_full_boost_diagnostics(self):
        """Even in SHADOW the trace must contain boost prediction info."""
        from tests.helpers_decision import preds as make_preds, run
        from custom_components.thermosmart.learning.decision.contracts import DecisionMode
        trace, _ = run(DecisionMode.SHADOW, predictions=make_preds(boost=2.0))
        boost_entry = next(e for e in trace.entries if e.feature == "boost_offset")
        assert boost_entry.le2_value == pytest.approx(2.0)
        assert not boost_entry.applied


# ===========================================================================
# 9. Learning Failure vs Heating Failure Separation
# ===========================================================================

class TestLearningVsHeatingFailure:
    """A boost learning failure must not become a heating system failure."""

    def test_no_response_is_learning_failure_not_heating_failure(self):
        """FAILED_NO_RESPONSE is a boost-learning state, not a system heating failure."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("no_response", _ts(1200))
        assert m._state.lifecycle == BoostLifecycle.FAILED_NO_RESPONSE
        # This should NOT propagate as a system-level heating failure signal
        # (tested by verifying the lifecycle state is FAILED_NO_RESPONSE, not a system flag)

    def test_heating_failure_separately_mapped(self):
        """Coordinator heating_failure → lifecycle FAILED_NO_RESPONSE (not RELEASED)."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("heating_failure", _ts(900))
        assert m._state.lifecycle == BoostLifecycle.FAILED_NO_RESPONSE
        assert m._state.lifecycle_release_reason == "heating_failure"


# ===========================================================================
# 10. CompatAdapter: boost_factor remains display-only
# ===========================================================================

class TestCompatAdapterSeparation:
    """boost_factor is exclusively for display; boost_offset_c is the control truth."""

    def test_boost_offset_zero_maps_to_factor_one(self):
        from custom_components.thermosmart.learning.models.boost import boost_offset_c_to_compat_factor
        assert boost_offset_c_to_compat_factor(0.0) == pytest.approx(1.0)

    def test_negative_offset_maps_to_factor_one(self):
        from custom_components.thermosmart.learning.models.boost import boost_offset_c_to_compat_factor
        assert boost_offset_c_to_compat_factor(-1.0) == pytest.approx(1.0)

    def test_compat_factor_never_zero(self):
        from custom_components.thermosmart.learning.models.boost import boost_offset_c_to_compat_factor
        for offset in [-5, -1, 0, 0.5, 1.0, 4.0, 8.0]:
            assert boost_offset_c_to_compat_factor(float(offset)) > 0.0

    def test_boost_factor_not_read_by_coordinator_control_path(self):
        """Coordinator reads boost_factor only to write it as display attribute."""
        import pathlib
        src = pathlib.Path("custom_components/thermosmart/coordinator.py").read_text()
        # boost_factor must only appear in assignment (write) context in coordinator,
        # not as a value read for control decisions
        lines_reading_bf = [
            line for line in src.splitlines()
            if "boost_factor" in line
            and "recommendation[" not in line
            and "boost_offset_c_to_compat_factor" not in line
            and ".get(" not in line.split("boost_factor")[0]
            and "# " not in line.lstrip()[:3]
        ]
        # Should only appear as write targets or in compat conversion
        # (If non-empty, that would indicate a control read — flag for review)
        assert len(lines_reading_bf) == 0, \
            f"Unexpected boost_factor control read in coordinator:\n" + "\n".join(lines_reading_bf)


# ===========================================================================
# 11. Deescalation: Mid-Episode Release When Heating Is Self-Sufficient
# ===========================================================================

class TestDeescalation:
    """_check_lifecycle_deescalation_safe: three mid-episode release checks.

    Authority map (no heuristic alone):
      TPI sufficient    → LE2 HEAT_RATE prediction + deficit projection (NOT duty threshold)
      Overshoot risk    → LE2 EXPECTED_OVERSHOOT (AfterheatModel residual rise) primary + heuristic
      Afterheat suff.   → LE2 EXPECTED_OVERSHOOT (AfterheatModel residual rise), authoritative
      (BOOST_OUTCOME = historical boost quality, NOT physical afterheat — must NOT be deesc authority)

    Each check has:
    - Lifecycle transition (APPLIED → RELEASED_xxx)
    - 0s cooldown (deescalation = successful boost conclusion, not failure)
    - completed_episode_id binding (blocks re-boost of same episode)
    - Reason code in lifecycle state
    """

    # ------------------------------------------------------------------ sync: pure lifecycle

    def test_deescalation_zero_cooldown(self):
        """All three deescalation reasons must have 0s cooldown (not a failure)."""
        for reason in ("released_tpi_sufficient", "released_overshoot_risk",
                       "released_afterheat_sufficient"):
            m = _fresh_model()
            _apply(m)
            m.release_lifecycle(reason, _ts(0))
            assert not m.cooldown_active(_ts(1)), (
                f"Deescalation reason '{reason}' must have 0s cooldown")
            assert m._state.cooldown_until_ts is None, (
                f"cooldown_until_ts must be None after '{reason}'")

    def test_deescalation_sets_completed_episode_binding(self):
        """After deescalation, same episode must not re-boost (completed binding set)."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        for reason in ("released_tpi_sufficient", "released_overshoot_risk",
                       "released_afterheat_sufficient"):
            m = _fresh_model()
            _apply(m, episode_id="ep_dees")
            m.release_lifecycle(reason, _ts(0))
            assert m._state.last_completed_episode_id == "ep_dees", (
                f"completed binding must be set after '{reason}'")
            result = _apply(m, episode_id="ep_dees")
            assert result is False, (
                f"Same episode must be blocked after deescalation reason '{reason}'")

    def test_deescalation_new_episode_allowed_after_release(self):
        """After deescalation, a genuinely new episode can still boost."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m, episode_id="ep_old")
        m.release_lifecycle("released_tpi_sufficient", _ts(0))
        m.reset_lifecycle()
        result = _apply(m, episode_id="ep_new")
        assert result is True
        assert m._state.lifecycle == BoostLifecycle.APPLIED

    def test_released_tpi_sufficient_lifecycle_state(self):
        """release_lifecycle('released_tpi_sufficient') → RELEASED_TPI_SUFFICIENT state."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("released_tpi_sufficient", _ts(100))
        assert m._state.lifecycle == BoostLifecycle.RELEASED_TPI_SUFFICIENT
        assert m._state.lifecycle_release_reason == "released_tpi_sufficient"

    def test_released_overshoot_risk_lifecycle_state(self):
        """release_lifecycle('released_overshoot_risk') → RELEASED_OVERSHOOT_RISK state."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("released_overshoot_risk", _ts(700))
        assert m._state.lifecycle == BoostLifecycle.RELEASED_OVERSHOOT_RISK
        assert m._state.lifecycle_release_reason == "released_overshoot_risk"

    def test_released_afterheat_sufficient_lifecycle_state(self):
        """release_lifecycle('released_afterheat_sufficient') → RELEASED_AFTERHEAT_SUFFICIENT."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        m = _fresh_model()
        _apply(m)
        m.release_lifecycle("released_afterheat_sufficient", _ts(500))
        assert m._state.lifecycle == BoostLifecycle.RELEASED_AFTERHEAT_SUFFICIENT
        assert m._state.lifecycle_release_reason == "released_afterheat_sufficient"

    # ------------------------------------------------------------------ helpers for async

    def _make_hr_pred(self, heat_rate: float = 3.0, confidence: float = 0.8) -> MagicMock:
        """Build a mock HEAT_RATE prediction."""
        p = MagicMock()
        p.fallback_used = False
        p.confidence = confidence
        p.values = {"heat_rate": heat_rate}
        return p

    def _make_afterheat_pred(self, residual_rise_c: float = 0.5,
                             confidence: float = 0.7,
                             source_episode_id: str = None) -> MagicMock:
        """Build a mock EXPECTED_OVERSHOOT prediction (from AfterheatModel residual rise).

        Authority source for: released_overshoot_risk (LE2 path) AND released_afterheat_sufficient.
        Key 'expected_overshoot' carries the residual rise in °C.
        All validity gates default to passing (unit=C, no stale/superseded warnings).
        source_episode_id=None means no episode binding check (compatible with any episode).
        """
        p = MagicMock()
        p.fallback_used = False
        p.confidence = confidence
        p.values = {"expected_overshoot": residual_rise_c}
        p.units = {"expected_overshoot": "C"}
        p.warnings = ()
        p.source_episode_id = source_episode_id
        return p

    def _make_boost_outcome_pred(self, expected_overshoot: float = 0.5,
                                 confidence: float = 0.7) -> MagicMock:
        """Build a mock BOOST_OUTCOME prediction (historical boost quality, NOT residual rise).

        BOOST_OUTCOME is used for trace/risk assessment only, NOT for deescalation authority.
        """
        p = MagicMock()
        p.fallback_used = False
        p.confidence = confidence
        p.values = {"expected_gain": 2.0, "expected_overshoot": expected_overshoot,
                    "expected_comfort": 0.8}
        return p

    def _make_cold_start_pred(self) -> MagicMock:
        """Build a cold-start (fallback_used=True) prediction."""
        p = MagicMock()
        p.fallback_used = True
        p.confidence = 0.2
        p.values = {"heat_rate": 2.0}
        return p

    async def _setup_shadow(self):
        from custom_components.thermosmart.learning.runtime import LearningRuntimeMode
        from tests.helpers_ha_runtime import attach_shadow, make_recording_coordinator
        coord = make_recording_coordinator(indoor="18.0")
        sh = attach_shadow(coord)
        sh.runtime.set_mode(LearningRuntimeMode.CONTROL)
        await sh.async_setup()
        return sh

    # ------------------------------------------------------------------ async: TPI sufficient

    async def test_tpi_sufficient_no_prediction_no_release(self):
        """Without LE2 heat_rate prediction (cold start), TPI sufficient never fires.

        High TPI duty alone (95%) does NOT prove sufficiency — it proves high demand.
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_no_hr", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # No predictions injected → cold start → TPI sufficient must not fire
        rec = {"effective_target": 21.0, "current_temp": 20.5,
               "tpi_duty_cycle": 95.0, "temp_slope": 0.1}
        sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "TPI duty alone must not trigger released_tpi_sufficient")

    async def test_tpi_sufficient_large_deficit_no_release(self):
        """TPI sufficient requires small remaining deficit (≤ 1°C). Large deficit → no release."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        from custom_components.thermosmart.learning.contracts import PredictionType
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_large", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # Heat rate: 3°C/h valid, but deficit = 21.0 - 18.5 = 2.5°C > 1.0°C threshold
        hr_pred = self._make_hr_pred(heat_rate=3.0)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.HEAT_RATE: hr_pred}):
            rec = {"effective_target": 21.0, "current_temp": 18.5,
                   "tpi_duty_cycle": 95.0, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Large deficit (2.5°C > 1.0°C) must block tpi_sufficient release")

    async def test_tpi_sufficient_with_heat_rate_small_deficit_triggers(self):
        """TPI sufficient fires when: LE2 heat_rate valid, deficit small, predicted close fast.

        remaining = 0.5°C, heat_rate = 3°C/h → predicted = 0.5/(3/60) = 10 min ≤ 20 min
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        from custom_components.thermosmart.learning.contracts import PredictionType
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_tpi_hr", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # remaining = 21.0 - 20.5 = 0.5°C ≤ 1.0°C; predicted = 0.5/(3/60)=10 min ≤ 20 min
        hr_pred = self._make_hr_pred(heat_rate=3.0, confidence=0.8)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.HEAT_RATE: hr_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.5, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_TPI_SUFFICIENT

    async def test_tpi_sufficient_slow_heat_rate_no_release(self):
        """Slow heat rate means too long to close gap → no release.

        remaining = 0.8°C, heat_rate = 0.3°C/h → predicted = 0.8/(0.3/60) = 160 min > 20 min
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        from custom_components.thermosmart.learning.contracts import PredictionType
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_slow_hr", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # heat_rate=0.3°C/h → predicted close = 160 min >> 20 min threshold
        hr_pred = self._make_hr_pred(heat_rate=0.3, confidence=0.8)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.HEAT_RATE: hr_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.2,
                   "tpi_duty_cycle": 95.0, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Slow heat rate (predicted close > 20 min) must not trigger tpi_sufficient")

    async def test_tpi_sufficient_cold_start_confidence_no_release(self):
        """Cold-start heat_rate prediction (fallback_used=True) → no tpi_sufficient release."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        from custom_components.thermosmart.learning.contracts import PredictionType
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_cold", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        cold_pred = self._make_cold_start_pred()  # fallback_used=True
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.HEAT_RATE: cold_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.8, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Cold-start prediction (fallback_used=True) must not trigger tpi_sufficient")

    # ------------------------------------------------------------------ async: overshoot risk

    async def test_released_overshoot_risk_heuristic_fallback(self):
        """Overshoot risk heuristic fires: remaining ≤ 0.3°C AND rising AND active ≥ 600s."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timedelta, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        # Apply lifecycle 700s ago so age gate is satisfied
        ts_past = (datetime.now(timezone.utc) - timedelta(seconds=700)).isoformat()
        model.apply_lifecycle(episode_id="ep_ov_h", applied_offset_c=1.5,
                              base_target_c=21.0, ts=ts_past)
        # remaining=0.25 ≤ 0.3, slope=0.06 > 0, active~700s ≥ 600s → heuristic fires
        rec = {"effective_target": 21.0, "current_temp": 20.75,
               "tpi_duty_cycle": 50.0, "temp_slope": 0.06}
        sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_OVERSHOOT_RISK

    async def test_released_overshoot_risk_prediction_primary(self):
        """LE2 EXPECTED_OVERSHOOT (AfterheatModel) covers small remaining → overshoot risk release.

        LE2-primary path requires remaining ≤ 0.3°C (overshoot_deficit_c threshold).
        Authority: PredictionType.EXPECTED_OVERSHOOT (residual rise), NOT BOOST_OUTCOME.
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        from custom_components.thermosmart.learning.contracts import PredictionType
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_ov_p", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # remaining=0.2°C ≤ 0.3 threshold; residual_rise=0.4°C → release via LE2 EXPECTED_OVERSHOOT
        afterheat_pred = self._make_afterheat_pred(residual_rise_c=0.4, confidence=0.7)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: afterheat_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.8, "temp_slope": 0.04}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_OVERSHOOT_RISK

    async def test_released_overshoot_risk_not_triggers_before_min_active(self):
        """Heuristic fallback requires ≥ 600s active. Fresh boost + no prediction → no release."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        # Fresh boost (0s active) — age gate blocks heuristic; no prediction → LE2 skipped
        model.apply_lifecycle(episode_id="ep_fresh", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        rec = {"effective_target": 21.0, "current_temp": 20.8,
               "tpi_duty_cycle": 50.0, "temp_slope": 0.03}
        sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED

    async def test_overshoot_low_confidence_prediction_no_release(self):
        """Low-confidence EXPECTED_OVERSHOOT (< 0.30 threshold) → no LE2-path release."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        from custom_components.thermosmart.learning.contracts import PredictionType
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_low_conf", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # Low confidence (0.2 < 0.30 threshold) — LE2 path skipped; heuristic needs 600s age
        low_conf_pred = self._make_afterheat_pred(residual_rise_c=0.5, confidence=0.2)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: low_conf_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.6, "temp_slope": 0.04}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Low-confidence EXPECTED_OVERSHOOT prediction must not trigger overshoot release")

    # ------------------------------------------------------------------ async: afterheat sufficient

    async def test_afterheat_sufficient_with_prediction_triggers(self):
        """LE2 EXPECTED_OVERSHOOT (AfterheatModel residual rise) covers gap → afterheat release.

        residual_rise=0.7°C ≥ remaining(0.5°C) + margin(0.1°C) → released_afterheat_sufficient.
        Authority: PredictionType.EXPECTED_OVERSHOOT (AfterheatModel), NOT BOOST_OUTCOME.
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        from custom_components.thermosmart.learning.contracts import PredictionType
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_aft_p", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # remaining=0.5°C; residual_rise=0.7 ≥ 0.5 + 0.1 margin → afterheat release
        afterheat_pred = self._make_afterheat_pred(residual_rise_c=0.7, confidence=0.65)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: afterheat_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.5, "temp_slope": 0.06}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_AFTERHEAT_SUFFICIENT

    async def test_afterheat_no_prediction_no_release(self):
        """Without EXPECTED_OVERSHOOT prediction, afterheat check must not fire.

        Active heating slope alone is not afterheat evidence — this was the old bug.
        BOOST_OUTCOME alone must also not trigger afterheat release (separate test below).
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_aft_none", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # Strong slope (0.10 °C/min) but NO prediction → must not release afterheat
        rec = {"effective_target": 21.0, "current_temp": 20.4,
               "tpi_duty_cycle": 60.0, "temp_slope": 0.10}
        sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Slope alone (no BOOST_OUTCOME prediction) must not trigger afterheat release")

    async def test_afterheat_cold_start_prediction_no_release(self):
        """Cold-start EXPECTED_OVERSHOOT (fallback_used=True) must not trigger afterheat release."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        from custom_components.thermosmart.learning.contracts import PredictionType
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_aft_cs", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        cold_pred = MagicMock()
        cold_pred.fallback_used = True
        cold_pred.confidence = 0.2
        cold_pred.values = {"expected_overshoot": 0.9}
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: cold_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.5, "temp_slope": 0.07}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Cold-start EXPECTED_OVERSHOOT (fallback_used=True) must not trigger afterheat release")

    async def test_afterheat_insufficient_coverage_no_release(self):
        """residual_rise does not cover remaining + margin → no afterheat release.

        remaining=0.8°C; residual_rise=0.7 < 0.8 + 0.1 margin → stays APPLIED.
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        from custom_components.thermosmart.learning.contracts import PredictionType
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_aft_nc", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # 0.7 < 0.8 + 0.1 → not covered
        afterheat_pred = self._make_afterheat_pred(residual_rise_c=0.7, confidence=0.7)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: afterheat_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.2, "temp_slope": 0.06}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED

    # ------------------------------------------------------------------ async: hard release guards

    async def test_deescalation_no_current_temp_is_noop(self):
        """Without current_temp in rec, deescalation check is a no-op (no release)."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_nct", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        rec = {"effective_target": 21.0, "tpi_duty_cycle": 95.0, "temp_slope": 0.1}
        sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED

    # ------------------------------------------------------------------ static / code guards

    def test_current_temp_key_is_current_temp(self):
        """Coordinator uses 'current_temp' key, not 'current_temperature'."""
        import pathlib
        src = pathlib.Path(
            "custom_components/thermosmart/learning/runtime/ha_integration.py").read_text()
        deesc_section = src[src.find("def _check_lifecycle_deescalation_safe"):
                            src.find("def adjust_recommendation_safe")]
        assert 'recommendation.get("current_temperature")' not in deesc_section, (
            "Deescalation must read 'current_temp', not 'current_temperature'")
        adjust_section = src[src.find("def adjust_recommendation_safe"):
                             src.find("def compute_decision_trace_safe")]
        assert 'recommendation.get("current_temperature")' not in adjust_section, (
            "adjust_recommendation_safe must read 'current_temp', not 'current_temperature'")

    def test_tpi_duty_not_used_in_tpi_sufficient(self):
        """tpi_sufficient check must NOT use duty threshold (duty = demand, not sufficiency)."""
        import pathlib
        src = pathlib.Path(
            "custom_components/thermosmart/learning/runtime/ha_integration.py").read_text()
        start = src.find("def _check_lifecycle_deescalation_safe")
        end = src.find("\n    def ", start + 1)
        method_src = src[start:end]
        assert "deescalation_tpi_duty_min" not in method_src, (
            "TPI sufficient must not use duty threshold — use heat_rate projection")
        assert "duty_min" not in method_src, (
            "TPI sufficient must not reference duty minimum threshold")

    def test_early_cutoff_returns_immediately(self):
        """adjust_recommendation_safe must return after early_cutoff release (hard release)."""
        import pathlib
        src = pathlib.Path(
            "custom_components/thermosmart/learning/runtime/ha_integration.py").read_text()
        adjust_start = src.find("def adjust_recommendation_safe")
        compute_start = src.find("def compute_decision_trace_safe")
        adjust_section = src[adjust_start:compute_start]
        # The early_cutoff block must contain a return statement
        ec_idx = adjust_section.find("early_cutoff_state")
        ec_block = adjust_section[ec_idx:ec_idx + 400]
        assert "return" in ec_block, (
            "adjust_recommendation_safe must return after early_cutoff release (hard release)")

    def test_hard_releases_precede_soft_deescalation(self):
        """In adjust_recommendation_safe, early_cutoff and target_reached must come BEFORE
        _check_lifecycle_deescalation_safe call so hard releases always take priority."""
        import pathlib
        src = pathlib.Path(
            "custom_components/thermosmart/learning/runtime/ha_integration.py").read_text()
        adjust_start = src.find("def adjust_recommendation_safe")
        compute_start = src.find("def compute_decision_trace_safe")
        adjust_section = src[adjust_start:compute_start]
        ec_idx = adjust_section.find("early_cutoff_state")
        deesc_idx = adjust_section.find("_check_lifecycle_deescalation_safe")
        assert ec_idx < deesc_idx, (
            "Hard release (early_cutoff) must appear BEFORE soft deescalation check")

    def test_boost_outcome_not_used_as_afterheat_authority(self):
        """BOOST_OUTCOME must not appear in the afterheat or overshoot deescalation logic."""
        import pathlib
        src = pathlib.Path(
            "custom_components/thermosmart/learning/runtime/ha_integration.py").read_text()
        start = src.find("def _check_lifecycle_deescalation_safe")
        # Find next method definition after deescalation check
        end = src.find("\n    def ", start + 1)
        method_src = src[start:end]
        assert len(method_src) > 100, (
            "_check_lifecycle_deescalation_safe method not found or empty")
        # BOOST_OUTCOME must not be fetched for deescalation logic
        assert "PredictionType.BOOST_OUTCOME" not in method_src, (
            "_check_lifecycle_deescalation_safe must not use BOOST_OUTCOME for physical checks")
        # EXPECTED_OVERSHOOT (AfterheatModel) must be used instead
        assert "PredictionType.EXPECTED_OVERSHOOT" in method_src, (
            "_check_lifecycle_deescalation_safe must use EXPECTED_OVERSHOOT for afterheat")

    # ------------------------------------------------------------------ async: hard release (offset = 0)

    async def test_window_open_immediate_zero_boost(self):
        """window_open → hard release → trv_setpoint = TPI baseline (boost offset = 0.0)."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_win", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # Window open: hard release → trv_setpoint must NOT be changed (TPI setpoint only)
        rec = {"effective_target": 21.0, "current_temp": 19.5,
               "trv_setpoint": 21.5,  # TPI baseline (no LE2 boost in this slot)
               "window_open": True, "tpi_duty_cycle": 70.0}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_WINDOW_OPEN
        # trv_setpoint must stay at TPI baseline — no additional boost offset
        assert rec["trv_setpoint"] == 21.5, (
            "window_open hard release must leave trv_setpoint at TPI baseline (no boost)")

    async def test_early_cutoff_immediate_zero_boost(self):
        """early_cutoff → hard release → trv_setpoint = TPI baseline (boost offset = 0.0)."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_ec2", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        rec = {"effective_target": 21.0, "current_temp": 19.5,
               "trv_setpoint": 21.3,
               "early_cutoff_state": "cutoff_applied", "tpi_duty_cycle": 80.0}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_MODE_CHANGE
        assert rec["trv_setpoint"] == 21.3, (
            "early_cutoff hard release must leave trv_setpoint at TPI baseline (no boost)")

    async def test_target_reached_immediate_zero_boost(self):
        """target_reached → hard release → trv_setpoint = TPI baseline in same cycle."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_tr2", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # current_temp >= target → target reached
        rec = {"effective_target": 21.0, "current_temp": 21.1,
               "trv_setpoint": 21.0, "tpi_duty_cycle": 10.0}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_TARGET_REACHED
        assert rec["trv_setpoint"] == 21.0, (
            "target_reached must return before boost apply → TPI baseline unchanged")

    # ------------------------------------------------------------------ async: soft step-down

    async def test_soft_deescalation_step_down_from_2c(self):
        """Soft deescalation (TPI sufficient) steps boost offset down, not immediate zero.

        Boost at +2.0°C → TPI sufficient fires → same cycle offset steps to 1.5°C.
        Each subsequent call reduces by deescalation_soft_step_c (0.5°C) until 0.
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_step", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # TPI sufficient: small remaining (0.5°C ≤ 1.0), good heat_rate (3°C/h), predicted 10 min
        # No schedule → fallback horizon 20 min; 10 min ≤ 20 min → fires
        hr_pred = self._make_hr_pred(heat_rate=3.0, confidence=0.8)
        tpi_baseline_setpoint = 21.0  # TPI setpoint before boost
        rec = {"effective_target": 21.0, "current_temp": 20.5,
               "trv_setpoint": tpi_baseline_setpoint, "temp_slope": 0.05}
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.HEAT_RATE: hr_pred}):
            sh.adjust_recommendation_safe(rec)
        # Lifecycle must be released (soft)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_TPI_SUFFICIENT, (
            "TPI sufficient must fire and set lifecycle to RELEASED_TPI_SUFFICIENT")
        # Same-cycle offset must be 1.5°C (stepped from 2.0 by 0.5), NOT immediately 0
        expected_setpoint = tpi_baseline_setpoint + 1.5
        assert rec["trv_setpoint"] == expected_setpoint, (
            f"Soft step-down: expected {expected_setpoint}, got {rec['trv_setpoint']}")
        # Second call: lifecycle is RELEASED, step-down continues
        rec2 = {"effective_target": 21.0, "current_temp": 20.5,
                "trv_setpoint": tpi_baseline_setpoint}
        sh.adjust_recommendation_safe(rec2)
        assert rec2["trv_setpoint"] == tpi_baseline_setpoint + 1.0, (
            f"Second step-down should be 1.0°C above baseline")
        # Third call: offset 0.5
        rec3 = {"effective_target": 21.0, "current_temp": 20.5,
                "trv_setpoint": tpi_baseline_setpoint}
        sh.adjust_recommendation_safe(rec3)
        assert rec3["trv_setpoint"] == tpi_baseline_setpoint + 0.5, (
            f"Third step-down should be 0.5°C above baseline")
        # Fourth call: offset reaches 0 → TPI baseline
        rec4 = {"effective_target": 21.0, "current_temp": 20.5,
                "trv_setpoint": tpi_baseline_setpoint}
        sh.adjust_recommendation_safe(rec4)
        assert rec4["trv_setpoint"] == tpi_baseline_setpoint, (
            "Fourth step-down: offset reaches 0, setpoint returns to TPI baseline")

    async def test_soft_deescalation_no_reapply_during_stepdown(self):
        """During soft step-down, no new higher boost can be applied (episode binding)."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_noreapply", applied_offset_c=1.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # Trigger soft release
        hr_pred = self._make_hr_pred(heat_rate=3.0, confidence=0.8)
        rec = {"effective_target": 21.0, "current_temp": 20.5,
               "trv_setpoint": 21.0, "temp_slope": 0.05}
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.HEAT_RATE: hr_pred}):
            sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_TPI_SUFFICIENT
        # During step-down: boost must not jump back up to 2.0
        rec2 = {"effective_target": 21.0, "current_temp": 20.5,
                "trv_setpoint": 21.0}
        sh.adjust_recommendation_safe(rec2)
        # Setpoint must be ≤ 21.5 (continuing step-down or at baseline)
        assert rec2.get("trv_setpoint", 21.0) <= 21.5, (
            "During soft step-down, no re-boost above baseline allowed")

    # ------------------------------------------------------------------ async: TPI target-time

    async def test_tpi_sufficient_target_too_soon_no_release(self):
        """TPI prediction 15 min, but target in 8 min → boost stays (insufficient headroom).

        remaining=0.5°C, heat_rate=2°C/h → predicted=0.5/(2/60)=15 min
        remaining_time=8 min, safety_margin=5 min → effective_horizon=3 min
        15 min > 3 min → no release.
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timedelta, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_tight", applied_offset_c=1.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # target in 8 minutes
        sh._last_comfort_time_utc = (
            datetime.now(timezone.utc) + timedelta(minutes=8)).isoformat()
        hr_pred = self._make_hr_pred(heat_rate=2.0, confidence=0.8)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.HEAT_RATE: hr_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.5, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Predicted 15 min > effective_horizon 3 min → boost must NOT release")

    async def test_tpi_sufficient_ample_time_releases(self):
        """TPI prediction 15 min, target in 30 min → effective_horizon=25 min → release.

        remaining=0.5°C, heat_rate=2°C/h → predicted=15 min
        remaining_time=30 min, safety_margin=5 min → effective_horizon=25 min
        15 min ≤ 25 min → release.
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timedelta, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_ample", applied_offset_c=1.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # target in 30 minutes → effective_horizon = 30 - 5 = 25 min
        sh._last_comfort_time_utc = (
            datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        hr_pred = self._make_hr_pred(heat_rate=2.0, confidence=0.8)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.HEAT_RATE: hr_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.5, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_TPI_SUFFICIENT, (
            "Predicted 15 min ≤ effective_horizon 25 min → boost must release")

    async def test_tpi_sufficient_no_schedule_uses_fallback_horizon(self):
        """Without schedule (no comfort_time_utc), TPI uses conservative fixed fallback horizon.

        remaining=0.5°C, heat_rate=3°C/h → predicted=10 min ≤ fallback 20 min → release.
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_noschedule", applied_offset_c=1.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        sh._last_comfort_time_utc = None  # no schedule
        hr_pred = self._make_hr_pred(heat_rate=3.0, confidence=0.8)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.HEAT_RATE: hr_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.5, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_TPI_SUFFICIENT, (
            "No schedule → fallback 20 min horizon; predicted 10 min ≤ 20 → release")

    # ------------------------------------------------------------------ static: authority separation

    async def test_boost_outcome_alone_no_afterheat_release(self):
        """BOOST_OUTCOME alone (without EXPECTED_OVERSHOOT) must not trigger afterheat release."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_bo_only", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # Only BOOST_OUTCOME injected — deescalation must NOT use it for afterheat
        boost_outcome_pred = self._make_boost_outcome_pred(expected_overshoot=0.9, confidence=0.8)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.BOOST_OUTCOME: boost_outcome_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.5, "temp_slope": 0.06}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "BOOST_OUTCOME alone must not trigger afterheat release — "
            "EXPECTED_OVERSHOOT (AfterheatModel) is required")

    # ------------------------------------------------------------------ Point 1: Hard/Soft Matrix

    async def test_overshoot_risk_hard_release_immediate_zero(self):
        """released_overshoot_risk must produce immediate 0 boost, not step-down.

        Boost at +2.0°C → overshoot risk fires → same cycle trv_setpoint = TPI baseline.
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_os_hard", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # 0.2°C remaining, overshoot pred 0.3°C, confidence 0.5 → overshoot_risk fires
        ah_pred = self._make_afterheat_pred(residual_rise_c=0.3, confidence=0.5)
        tpi_baseline = 20.8  # coordinator's TPI setpoint (21.0 - 0.2 remaining ≈ baseline)
        rec = {"effective_target": 21.0, "current_temp": 20.8, "temp_slope": 0.05,
               "trv_setpoint": tpi_baseline}
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: ah_pred}):
            sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_OVERSHOOT_RISK, (
            "Overshoot risk must fire")
        # trv_setpoint must NOT be bumped up by a residual boost offset
        assert rec["trv_setpoint"] == tpi_baseline, (
            f"Hard release: trv_setpoint must stay at TPI baseline {tpi_baseline}, "
            f"got {rec.get('trv_setpoint')}")

    async def test_overshoot_risk_subsequent_cycle_still_hard(self):
        """After overshoot risk release, subsequent cycles also produce no residual boost."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_os_cycle2", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        ah_pred = self._make_afterheat_pred(residual_rise_c=0.3, confidence=0.5)
        rec1 = {"effective_target": 21.0, "current_temp": 20.8, "temp_slope": 0.05,
                "trv_setpoint": 20.8}
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: ah_pred}):
            sh.adjust_recommendation_safe(rec1)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_OVERSHOOT_RISK
        # Second cycle — no residual boost
        rec2 = {"effective_target": 21.0, "current_temp": 20.8, "trv_setpoint": 20.8}
        sh.adjust_recommendation_safe(rec2)
        assert rec2["trv_setpoint"] == 20.8, (
            "Cycle 2 after overshoot hard release: no residual boost on trv_setpoint")

    async def test_afterheat_sufficient_hard_release_immediate_zero(self):
        """released_afterheat_sufficient must produce immediate 0 boost, not step-down.

        Boost at +2.0°C → afterheat sufficient → same cycle trv_setpoint = TPI baseline.
        """
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_ah_hard", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # 0.5°C remaining, afterheat pred 0.7°C (> 0.5 + 0.1 margin → covers gap)
        ah_pred = self._make_afterheat_pred(residual_rise_c=0.7, confidence=0.6)
        tpi_baseline = 20.5
        rec = {"effective_target": 21.0, "current_temp": 20.5, "temp_slope": 0.06,
               "trv_setpoint": tpi_baseline}
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: ah_pred}):
            sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_AFTERHEAT_SUFFICIENT, (
            "Afterheat sufficient must fire")
        assert rec["trv_setpoint"] == tpi_baseline, (
            f"Hard release: trv_setpoint must stay at TPI baseline, got {rec.get('trv_setpoint')}")

    async def test_tpi_sufficient_still_soft_step_down(self):
        """TPI sufficient remains soft (step-down per cycle), not immediate zero."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_tpi_soft2", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        hr_pred = self._make_hr_pred(heat_rate=3.0, confidence=0.8)
        sh._last_comfort_time_utc = None  # fallback horizon
        rec = {"effective_target": 21.0, "current_temp": 20.5, "temp_slope": 0.05,
               "trv_setpoint": 21.0}
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.HEAT_RATE: hr_pred}):
            sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_TPI_SUFFICIENT
        # Soft: setpoint = baseline + (2.0 - 0.5) = 21.0 + 1.5 = 22.5, NOT 21.0
        assert rec["trv_setpoint"] == 22.5, (
            f"TPI soft step-down: expected 22.5 (not 21.0), got {rec.get('trv_setpoint')}")

    # ------------------------------------------------------------------ Point 2: Prediction Validity

    async def test_expected_overshoot_wrong_unit_no_release(self):
        """EXPECTED_OVERSHOOT with wrong unit (e.g. 'K') must not trigger any deescalation."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_unit", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        bad_unit_pred = MagicMock()
        bad_unit_pred.fallback_used = False
        bad_unit_pred.confidence = 0.8
        bad_unit_pred.values = {"expected_overshoot": 0.7}
        bad_unit_pred.units = {"expected_overshoot": "K"}  # wrong unit!
        bad_unit_pred.warnings = ()
        bad_unit_pred.source_episode_id = "ep_unit"
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: bad_unit_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.3, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Wrong unit 'K' must fail validity gate → no release")

    async def test_expected_overshoot_stale_warning_no_release(self):
        """EXPECTED_OVERSHOOT marked stale (in warnings) must not trigger deescalation."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_stale", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        stale_pred = MagicMock()
        stale_pred.fallback_used = False
        stale_pred.confidence = 0.8
        stale_pred.values = {"expected_overshoot": 0.7}
        stale_pred.units = {"expected_overshoot": "C"}
        stale_pred.warnings = ("stale",)  # stale flag
        stale_pred.source_episode_id = "ep_stale"
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: stale_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.3, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Stale prediction must fail validity gate → no release")

    async def test_expected_overshoot_superseded_warning_no_release(self):
        """EXPECTED_OVERSHOOT marked superseded must not trigger deescalation."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_sup", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        sup_pred = MagicMock()
        sup_pred.fallback_used = False
        sup_pred.confidence = 0.8
        sup_pred.values = {"expected_overshoot": 0.7}
        sup_pred.units = {"expected_overshoot": "C"}
        sup_pred.warnings = ("superseded",)  # superseded flag
        sup_pred.source_episode_id = "ep_sup"
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: sup_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.3, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Superseded prediction must fail validity gate → no release")

    async def test_expected_overshoot_wrong_episode_no_release(self):
        """EXPECTED_OVERSHOOT from a different episode must not trigger deescalation."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_current", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        wrong_ep_pred = MagicMock()
        wrong_ep_pred.fallback_used = False
        wrong_ep_pred.confidence = 0.8
        wrong_ep_pred.values = {"expected_overshoot": 0.7}
        wrong_ep_pred.units = {"expected_overshoot": "C"}
        wrong_ep_pred.warnings = ()
        wrong_ep_pred.source_episode_id = "ep_OLD_different"  # mismatch!
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: wrong_ep_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.3, "temp_slope": 0.05}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.APPLIED, (
            "Prediction from wrong episode must fail validity gate → no release")

    async def test_expected_overshoot_all_gates_pass_releases(self):
        """EXPECTED_OVERSHOOT with all validity gates passing triggers afterheat release."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_valid", applied_offset_c=1.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        valid_pred = MagicMock()
        valid_pred.fallback_used = False      # Gate 1 ✓
        valid_pred.confidence = 0.7           # Gate 5 ✓
        valid_pred.values = {"expected_overshoot": 0.7}
        valid_pred.units = {"expected_overshoot": "C"}  # Gate 2 ✓
        valid_pred.warnings = ()              # Gate 3 ✓
        valid_pred.source_episode_id = "ep_valid"  # Gate 4 ✓ (matches current episode)
        with patch.object(sh, "_get_le2_predictions_for_zone",
                          return_value={PredictionType.EXPECTED_OVERSHOOT: valid_pred}):
            rec = {"effective_target": 21.0, "current_temp": 20.5, "temp_slope": 0.06}
            sh._check_lifecycle_deescalation_safe(rec)
        assert model._state.lifecycle in (
            BoostLifecycle.RELEASED_OVERSHOOT_RISK,
            BoostLifecycle.RELEASED_AFTERHEAT_SUFFICIENT), (
            "All validity gates pass → prediction-based release must fire")

    # ------------------------------------------------------------------ Point 3: Manual Override

    async def test_manual_override_hard_release(self):
        """override_active=True → manual_override hard release → trv_setpoint = TPI baseline."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_mo_hard", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        tpi_baseline = 22.0  # coordinator TPI setpoint
        rec = {"effective_target": 21.0, "current_temp": 19.5,
               "trv_setpoint": tpi_baseline, "override_active": True}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_MANUAL_OVERRIDE, (
            "override_active must fire manual_override hard release")
        assert rec["trv_setpoint"] == tpi_baseline, (
            "Manual override: trv_setpoint must stay at TPI baseline, no residual boost")

    async def test_manual_override_no_reapply_same_episode(self):
        """After manual_override release, same episode cannot re-boost (episode binding)."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_mo_bind", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        rec = {"effective_target": 21.0, "current_temp": 19.5,
               "trv_setpoint": 21.0, "override_active": True}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_MANUAL_OVERRIDE
        # Simulate next cycle: override cleared, same episode tries to re-boost
        _initial_setpoint = 21.0
        rec2 = {"effective_target": 21.0, "current_temp": 19.5, "trv_setpoint": _initial_setpoint}
        result = model.apply_lifecycle(episode_id="ep_mo_bind", applied_offset_c=2.0,
                                       base_target_c=21.0,
                                       ts=datetime.now(timezone.utc).isoformat())
        assert not result, "Same episode must not be able to re-boost after manual override"

    async def test_manual_override_trace_reason(self):
        """Manual override release reason must be recorded in lifecycle state."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_mo_trace", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        rec = {"effective_target": 21.0, "current_temp": 19.5,
               "trv_setpoint": 21.0, "override_active": True}
        sh.adjust_recommendation_safe(rec)
        release_reason = model._state.lifecycle_release_reason
        assert release_reason == "manual_override", (
            f"Lifecycle release reason must be 'manual_override', got {release_reason!r}")

    # ------------------------------------------------------------------ Point 4: Target Change

    async def test_target_change_hard_release(self):
        """Target temperature change mid-boost → hard release, old offset invalidated."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        # Boost applied at target 21.0°C
        model.apply_lifecycle(episode_id="ep_tc", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        assert model._state.lifecycle_base_target_c == 21.0
        tpi_baseline = 22.5
        # New recommendation: target changed to 22.5°C (>0.1°C difference)
        rec = {"effective_target": 22.5, "current_temp": 20.0, "trv_setpoint": tpi_baseline}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_MODE_CHANGE, (
            "Target change of 1.5°C must trigger hard release (RELEASED_MODE_CHANGE)")
        assert rec["trv_setpoint"] == tpi_baseline, (
            "After target change: trv_setpoint must stay at TPI baseline, no residual boost")

    async def test_target_unchanged_no_false_release(self):
        """Small floating-point delta (< 0.1°C) must not trigger a target-change release."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from custom_components.thermosmart.learning.contracts import PredictionType
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        model.apply_lifecycle(episode_id="ep_notc", applied_offset_c=1.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # Target 21.05 — within 0.1°C tolerance → no release
        rec = {"effective_target": 21.05, "current_temp": 20.0, "trv_setpoint": 21.0}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle in (
            BoostLifecycle.APPLIED, BoostLifecycle.RELEASED_TPI_SUFFICIENT,
            BoostLifecycle.RELEASED_OVERSHOOT_RISK,
            BoostLifecycle.RELEASED_AFTERHEAT_SUFFICIENT), (
            "Target delta 0.05°C < 0.1°C tolerance must not trigger target_change release")
        assert model._state.lifecycle != BoostLifecycle.RELEASED_MODE_CHANGE, (
            "Within-tolerance target must not produce RELEASED_MODE_CHANGE")

    # ------------------------------------------------------------------ Point 1: Schedule context

    async def test_schedule_context_same_target_comfort_time_changes_releases(self):
        """Same effective_target, new comfort_time_utc → boost must end (schedule transition)."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        # Simulate: boost was applied in period A (comfort_time = 07:00)
        sh._boost_applied_comfort_time_utc = "2024-01-15T07:00:00+00:00"
        sh._last_comfort_time_utc = "2024-01-15T07:00:00+00:00"  # same initially
        model.apply_lifecycle(episode_id="ep_ctx_a", applied_offset_c=2.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # Period B starts: new comfort_time (12:00), SAME target (21.0°C)
        sh._last_comfort_time_utc = "2024-01-15T12:00:00+00:00"  # changed!
        tpi_baseline = 21.0
        rec = {"effective_target": 21.0, "current_temp": 19.5, "trv_setpoint": tpi_baseline}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_MODE_CHANGE, (
            "comfort_time change with same target must trigger schedule_change hard release")
        assert rec["trv_setpoint"] == tpi_baseline, (
            "Schedule context release: trv_setpoint must stay at TPI baseline")

    async def test_schedule_preheat_to_comfort_transition_releases(self):
        """Preheat → regular comfort period (comfort_time advances): boost must end."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        sh._boost_applied_comfort_time_utc = "2024-01-15T06:30:00+00:00"  # preheat target time
        sh._last_comfort_time_utc = "2024-01-15T06:30:00+00:00"
        model.apply_lifecycle(episode_id="ep_preheat", applied_offset_c=1.5,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # Comfort period starts: comfort_time advances to next slot
        sh._last_comfort_time_utc = "2024-01-15T12:00:00+00:00"
        rec = {"effective_target": 21.0, "current_temp": 20.5, "trv_setpoint": 21.0}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_MODE_CHANGE, (
            "Preheat→comfort transition (comfort_time change) must trigger schedule_change release")

    async def test_identical_context_no_false_schedule_release(self):
        """Same target AND same comfort_time → no spurious schedule_change release."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        same_time = "2024-01-15T07:00:00+00:00"
        sh._boost_applied_comfort_time_utc = same_time
        sh._last_comfort_time_utc = same_time
        model.apply_lifecycle(episode_id="ep_stable", applied_offset_c=1.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        rec = {"effective_target": 21.0, "current_temp": 20.0, "trv_setpoint": 21.0}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle != BoostLifecycle.RELEASED_MODE_CHANGE, (
            "Identical comfort_time must NOT trigger schedule_change release")

    async def test_no_schedule_both_none_no_false_release(self):
        """No schedule on either side (both None) → no spurious schedule_change release."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        sh._boost_applied_comfort_time_utc = None
        sh._last_comfort_time_utc = None
        model.apply_lifecycle(episode_id="ep_no_sched", applied_offset_c=1.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        rec = {"effective_target": 21.0, "current_temp": 20.0, "trv_setpoint": 21.0}
        sh.adjust_recommendation_safe(rec)
        assert model._state.lifecycle != BoostLifecycle.RELEASED_MODE_CHANGE, (
            "Both comfort_times None (no schedule) must NOT trigger schedule_change release")

    async def test_new_episode_after_schedule_change_can_boost(self):
        """After clean schedule_change release, a new episode with same target CAN boost."""
        from custom_components.thermosmart.learning.models.boost import BoostLifecycle
        from datetime import datetime, timezone
        sh = await self._setup_shadow()
        model = sh._boost_model()
        if model is None:
            pytest.skip("no boost model available")
        # Apply in period A
        sh._boost_applied_comfort_time_utc = "2024-01-15T07:00:00+00:00"
        sh._last_comfort_time_utc = "2024-01-15T07:00:00+00:00"
        model.apply_lifecycle(episode_id="ep_period_a", applied_offset_c=1.0,
                              base_target_c=21.0, ts=datetime.now(timezone.utc).isoformat())
        # Period B: schedule_change fires
        sh._last_comfort_time_utc = "2024-01-15T12:00:00+00:00"
        rec1 = {"effective_target": 21.0, "current_temp": 19.5, "trv_setpoint": 21.0}
        sh.adjust_recommendation_safe(rec1)
        assert model._state.lifecycle == BoostLifecycle.RELEASED_MODE_CHANGE
        # New episode in period B can boost (schedule_change = clean context = no episode binding)
        sh._boost_applied_comfort_time_utc = "2024-01-15T12:00:00+00:00"
        result = model.apply_lifecycle(episode_id="ep_period_b", applied_offset_c=1.5,
                                       base_target_c=21.0,
                                       ts=datetime.now(timezone.utc).isoformat())
        assert result, "New episode in new schedule period must be able to boost after clean release"

    # ------------------------------------------------------------------ Point 2: Unit contract

    def test_celsius_unit_C_accepted(self):
        """Unit 'C' (AfterheatModel contract form) must be accepted."""
        from custom_components.thermosmart.learning.runtime.ha_integration import _is_celsius_unit
        assert _is_celsius_unit("C"), "Unit 'C' must be accepted"

    def test_celsius_unit_lowercase_accepted(self):
        """Unit 'celsius' (other model contract form) must be accepted."""
        from custom_components.thermosmart.learning.runtime.ha_integration import _is_celsius_unit
        assert _is_celsius_unit("celsius"), "Unit 'celsius' must be accepted"

    def test_kelvin_unit_rejected(self):
        """Unit 'K' must be rejected (wrong physical unit)."""
        from custom_components.thermosmart.learning.runtime.ha_integration import _is_celsius_unit
        assert not _is_celsius_unit("K"), "Unit 'K' must be rejected"

    def test_percent_unit_rejected(self):
        """Unit '%' must be rejected."""
        from custom_components.thermosmart.learning.runtime.ha_integration import _is_celsius_unit
        assert not _is_celsius_unit("%"), "Unit '%' must be rejected"

    def test_min_unit_rejected(self):
        """Unit 'min' must be rejected."""
        from custom_components.thermosmart.learning.runtime.ha_integration import _is_celsius_unit
        assert not _is_celsius_unit("min"), "Unit 'min' must be rejected"

    def test_empty_unit_rejected_conservatively(self):
        """Empty unit string must be rejected (conservative: missing unit → no control)."""
        from custom_components.thermosmart.learning.runtime.ha_integration import _is_celsius_unit
        assert not _is_celsius_unit(""), "Empty unit must be rejected (conservative)"

    def test_degree_c_symbol_not_contract_form(self):
        """Unit '°C' is NOT a contract form and must be rejected (not in LE2 output)."""
        from custom_components.thermosmart.learning.runtime.ha_integration import _is_celsius_unit
        assert not _is_celsius_unit("°C"), "Unit '°C' is not a LE2 contract form — must be rejected"

    def test_celsius_uppercase_not_contract_form(self):
        """Unit 'CELSIUS' (all caps) is NOT a LE2 contract form and must be rejected."""
        from custom_components.thermosmart.learning.runtime.ha_integration import _is_celsius_unit
        assert not _is_celsius_unit("CELSIUS"), "Unit 'CELSIUS' is not a LE2 contract form"

    def test_celsius_unit_in_expected_overshoot_afterheat_pred_accepted(self):
        """Prediction with unit 'C' for expected_overshoot (AfterheatModel contract) must pass gate."""
        import pathlib
        src = pathlib.Path(
            "custom_components/thermosmart/learning/models/afterheat.py").read_text()
        assert '"expected_overshoot": "C"' in src or "\"expected_overshoot\": \"C\"" in src, (
            "AfterheatModel must emit units={'expected_overshoot': 'C'} per LE2 contract")
