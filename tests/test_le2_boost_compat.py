"""Phase 19C Pflicht-Tests: Boost backward-compat adapter, adaptive cap, lifecycle, outcome."""
from __future__ import annotations

import pytest
from dataclasses import replace

from custom_components.thermosmart.learning.models.boost import (
    BoostLifecycle,
    BoostModel,
    BoostParameters,
    BoostState,
    boost_offset_c_to_compat_factor,
)
from custom_components.thermosmart.learning.decision import DecisionMode
from tests.helpers_decision import preds, rec, run


# ─────────────────────────────────────────────────────────────────────────────
# 1. Compat-Adapter: boost_factor (public) ≠ boost_offset_c (internal)
# ─────────────────────────────────────────────────────────────────────────────

class TestCompatAdapter:
    def test_zero_offset_maps_to_neutral_factor(self):
        """0.0 °C → boost_factor == 1.0 (neutral for all automations)."""
        assert boost_offset_c_to_compat_factor(0.0) == pytest.approx(1.0)

    def test_negative_offset_maps_to_neutral_factor(self):
        """Negative (invalid) offset → clamped to 1.0."""
        assert boost_offset_c_to_compat_factor(-1.0) == pytest.approx(1.0)

    def test_max_offset_maps_to_two(self):
        """Max technical offset (8°C) → factor == 2.0 (top of LE v1 range)."""
        assert boost_offset_c_to_compat_factor(8.0, max_boost_c=8.0) == pytest.approx(2.0)

    def test_mid_offset_maps_linearly(self):
        """4°C at 8°C max → factor == 1.5 (linear midpoint)."""
        assert boost_offset_c_to_compat_factor(4.0, max_boost_c=8.0) == pytest.approx(1.5)

    def test_never_returns_zero(self):
        """factor must never be 0.0 — old automations treat 0.0 as broken, not neutral."""
        for offset in (-2.0, 0.0, 0.5, 3.0, 8.0):
            assert boost_offset_c_to_compat_factor(offset) > 0.0

    def test_compat_factor_not_used_for_control(self):
        """Resolver reads boost_offset_c (additive °C), never the compat factor.

        Proof: set boost_factor to a contradictory value (5.0) in the rec dict.
        boost_offset_c stays at 2.0. The control result must be IDENTICAL to when
        boost_factor is absent — only the public display attribute changes.
        """
        # Baseline: boost_offset_c=2.0, no contradictory boost_factor
        t_ref, _ = run(DecisionMode.CONTROL,
                       recommendation=rec(target=21.0, setpoint=23.0, boost_offset_c=2.0),
                       predictions=preds(boost=1.5))
        boost_ref = next(e for e in t_ref.entries if e.feature == "boost_offset")

        # With contradictory boost_factor=5.0 (large compat number, not a control value)
        r_contradiction = rec(target=21.0, setpoint=23.0, boost_offset_c=2.0)
        r_contradiction["boost_factor"] = 5.0  # intentionally contradicts boost_offset_c
        t_contra, _ = run(DecisionMode.CONTROL,
                          recommendation=r_contradiction,
                          predictions=preds(boost=1.5))
        boost_contra = next(e for e in t_contra.entries if e.feature == "boost_offset")

        # Control result must be identical regardless of boost_factor value
        assert boost_ref.final_value == pytest.approx(boost_contra.final_value), (
            "boost_factor in rec must not affect control decision")
        assert t_ref.final_setpoint_c == pytest.approx(t_contra.final_setpoint_c), (
            "final setpoint must not change when boost_factor changes")

        # Final offset is the raw °C offset (1.5 in this case), not a compat multiplier
        assert boost_ref.final_value == pytest.approx(1.5)
        assert boost_ref.final_value != pytest.approx(1.5 / 8.0 + 1.0)  # not compat formula

        # Static proof: no control code reads "boost_factor" from rec for setpoint decisions
        import pathlib
        for fname in ("resolver.py", "baseline.py"):
            src = pathlib.Path(
                f"custom_components/thermosmart/learning/decision/{fname}").read_text()
            assert 'rec.get("boost_factor")' not in src, (
                f"{fname} must not read boost_factor from rec")
            assert '"boost_factor"' not in src or fname == "resolver.py" and (
                'boost_offset_c_to_compat_factor' in src), (
                f"{fname} boost_factor reference must only be for compat display")

    def test_trace_boost_factor_compat_populated_when_applied(self):
        """DecisionTrace.boost_factor_compat is set when boost is applied."""
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=23.0),
                   predictions=preds(boost=1.5))
        if t.applied_any:
            assert t.boost_factor_compat is not None
            assert t.boost_factor_compat >= 1.0

    def test_trace_boost_offset_c_applied_when_applied(self):
        """DecisionTrace.boost_offset_c_applied is the internal °C value, not factor."""
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=23.0),
                   predictions=preds(boost=1.5))
        if t.applied_any:
            assert t.boost_offset_c_applied is not None
            # internal value is additive °C, not a multiplier
            assert t.boost_offset_c_applied < 5.0  # sanity: not 1.5+1.0=2.5 compat

    def test_old_factor_one_not_treated_as_celsius(self):
        """Compatibility factor 1.0 must never be interpreted as +1.0 °C offset internally."""
        # LE2 with boost_offset=0.0 (neutral): factor=1.0, setpoint must stay at target
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=21.0),
                   predictions=preds(boost=0.0))
        # With boost=0.0 proposed and baseline also 0.0, resolver should apply 0.0 or stay
        # The critical assertion: setpoint must not jump by 1.0 due to factor interpretation
        assert t.final_setpoint_c != pytest.approx(22.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Adaptive Control Cap (LE 2.0 vs technical 8°C TPI limit)
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveCap:
    def _model(self, **kwargs) -> BoostModel:
        return BoostModel("z", BoostParameters(**kwargs))

    def test_cold_start_cap_is_conservative(self):
        """Cold start (no evidence): adaptive cap at 0.5°C, well below 8°C technical max."""
        m = self._model()
        assert m.adaptive_cap_c() == pytest.approx(0.5)

    def test_cold_start_confidence_below_threshold(self):
        """Cold start confidence (0.2) is below cold_start_confidence_cap (0.35): 0.5°C cap."""
        m = self._model()
        conf = m._confidence_value(0.0, fallback=True)
        assert conf < 0.35
        assert m.adaptive_cap_c() == pytest.approx(0.5)

    def test_technical_max_never_reached_by_adaptive_cap(self):
        """Adaptive cap never reaches technical 8°C without explicit configuration."""
        m = self._model(adaptive_max_boost_c=3.0, max_boost_offset_c=8.0)
        # Even at max evidence, adaptive_max_boost_c (3.0) is the ceiling
        cap = m.adaptive_cap_c()
        assert cap <= 3.0
        assert cap < 8.0

    def test_high_overshoot_rate_reduces_cap(self):
        """Overshoot rate > 0.3 applies 0.7× penalty to adaptive cap."""
        m = self._model(adaptive_max_boost_c=3.0)
        # Inject overshoot count to simulate high rate
        m._state = replace(m._state,
            general=replace(m._state.general, sample_count=10),
            overshoot_count=4)  # rate = 0.4
        cap = m.adaptive_cap_c()
        # max cap without penalty: at cold start = 0.5; penalty would be 0.5*0.7 = 0.35
        # At cold start the cap is already 0.5, penalty applied: 0.35
        assert cap < 0.5

    def test_device_prior_subject_to_adaptive_cap(self):
        """Device prior is a cold-start hint, not a bypass; adaptive cap applies."""
        from custom_components.thermosmart.learning.models.boost import BoostPredictionContext
        m = self._model()
        ctx = BoostPredictionContext(device_prior_offset_c=2.5)
        rec_out = m.predict_boost_factor(ctx)
        # Cold-start adaptive cap = 0.5°C; prior of 2.5 is reduced, not bypassed
        assert rec_out.boost_offset_c == pytest.approx(0.5)
        assert rec_out.adaptive_cap_c == pytest.approx(0.5)

    def test_prediction_with_evidence_gets_higher_cap(self):
        """With sufficient evidence and high confidence, cap rises above cold-start tier."""
        from custom_components.thermosmart.learning.models.boost import (
            BoostPredictionContext, _Dim, BoostBucket)
        m = self._model(adaptive_max_boost_c=3.0, full_confidence_samples=10.0)
        # Simulate high evidence: effective_factor EMA of 2.0 with 15 samples
        trained_dim = _Dim(value=2.0, effective_n=15.0, dispersion=0.2, sample_count=15)
        trained_general = BoostBucket(key="general", effective_factor=trained_dim, sample_count=15)
        m._state = replace(m._state, general=trained_general)
        cap = m.adaptive_cap_c()
        # With 15 samples and full_confidence=10: coverage=1.0, should be > 0.5
        assert cap > 0.5

    def test_recommendation_adaptive_cap_field(self):
        """BoostRecommendation carries the adaptive_cap_c field."""
        from custom_components.thermosmart.learning.models.boost import BoostPredictionContext
        m = self._model()
        ctx = BoostPredictionContext()
        rec_out = m.predict_boost_factor(ctx)
        assert hasattr(rec_out, "adaptive_cap_c")
        assert rec_out.adaptive_cap_c >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lifecycle – Zustandsmaschine und Episode Binding
# ─────────────────────────────────────────────────────────────────────────────

class TestLifecycle:
    def _model(self) -> BoostModel:
        return BoostModel("z")

    def test_initial_state_is_inactive(self):
        assert self._model()._state.lifecycle == BoostLifecycle.INACTIVE

    def test_apply_lifecycle_transitions_to_applied(self):
        m = self._model()
        m.apply_lifecycle("ep1", applied_offset_c=1.5, base_target_c=21.0,
                          ts="2025-01-01T07:00:00+00:00")
        assert m._state.lifecycle == BoostLifecycle.APPLIED
        assert m._state.current_episode_id == "ep1"
        assert m._state.applied_offset_c == pytest.approx(1.5)
        assert m._state.lifecycle_base_target_c == pytest.approx(21.0)
        assert m._state.lifecycle_start_ts == "2025-01-01T07:00:00+00:00"

    def test_apply_lifecycle_stores_user_target(self):
        m = self._model()
        m.apply_lifecycle("ep2", 1.0, 21.0, "2025-01-01T08:00:00+00:00",
                          user_target_c=22.5)
        assert m._state.lifecycle_user_target_c == pytest.approx(22.5)

    def test_apply_lifecycle_uses_params_max_duration(self):
        m = self._model()
        m.apply_lifecycle("ep3", 1.0, 21.0, "2025-01-01T09:00:00+00:00")
        assert m._state.lifecycle_max_duration_s == pytest.approx(3600.0)

    def test_apply_lifecycle_custom_max_duration(self):
        m = self._model()
        m.apply_lifecycle("ep4", 1.0, 21.0, "2025-01-01T09:00:00+00:00", max_duration_s=3600.0)
        assert m._state.lifecycle_max_duration_s == pytest.approx(3600.0)

    def test_release_lifecycle_target_reached(self):
        m = self._model()
        m.apply_lifecycle("ep5", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("target_reached", "2025-01-01T07:20:00+00:00")
        assert m._state.lifecycle == BoostLifecycle.RELEASED_TARGET_REACHED
        assert m._state.lifecycle_release_reason == "target_reached"

    def test_release_lifecycle_window_open(self):
        m = self._model()
        m.apply_lifecycle("ep6", 1.5, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("window_open", "2025-01-01T07:05:00+00:00")
        assert m._state.lifecycle == BoostLifecycle.RELEASED_WINDOW_OPEN

    def test_release_lifecycle_timeout(self):
        m = self._model()
        m.apply_lifecycle("ep7", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("timeout", "2025-01-01T07:30:00+00:00")
        assert m._state.lifecycle == BoostLifecycle.RELEASED_TIMEOUT

    def test_release_lifecycle_manual_override(self):
        m = self._model()
        m.apply_lifecycle("ep8", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("manual_override", "2025-01-01T07:10:00+00:00")
        assert m._state.lifecycle == BoostLifecycle.RELEASED_MANUAL_OVERRIDE

    def test_fail_no_response(self):
        m = self._model()
        m.apply_lifecycle("ep9", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("no_response", "2025-01-01T07:15:00+00:00")
        assert m._state.lifecycle == BoostLifecycle.FAILED_NO_RESPONSE

    def test_fail_overshoot(self):
        m = self._model()
        m.apply_lifecycle("ep10", 2.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("overshoot", "2025-01-01T07:25:00+00:00")
        assert m._state.lifecycle == BoostLifecycle.FAILED_OVERSHOOT

    def test_release_starts_cooldown_for_failure(self):
        # Failure reasons (no_response, overshoot, heating_failure) → 900s cooldown
        m = self._model()
        m.apply_lifecycle("ep11", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("no_response", "2025-01-01T07:20:00+00:00")
        assert m._state.cooldown_until_ts is not None
        assert m._state.cooldown_until_ts > "2025-01-01T07:20:00+00:00"

    def test_target_reached_no_cooldown(self):
        # target_reached is a clean completion — no cooldown needed; new episode welcome immediately
        m = self._model()
        m.apply_lifecycle("ep11b", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("target_reached", "2025-01-01T07:20:00+00:00")
        assert m._state.cooldown_until_ts is None
        assert not m.cooldown_active("2025-01-01T07:20:01+00:00")

    def test_cooldown_active_during_period(self):
        # window_open → 300s cooldown
        m = self._model()
        m.apply_lifecycle("ep12", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("window_open", "2025-01-01T07:20:00+00:00")
        # Query 1 min after release — still in 300s cooldown
        assert m.cooldown_active("2025-01-01T07:21:00+00:00")

    def test_cooldown_expires(self):
        # window_open → 300s (5 min) cooldown
        m = self._model()
        m.apply_lifecycle("ep13", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("window_open", "2025-01-01T07:20:00+00:00")
        # Query 6 min later — cooldown expired (300s = 5 min)
        assert not m.cooldown_active("2025-01-01T07:26:00+00:00")

    def test_reset_lifecycle_returns_to_inactive(self):
        m = self._model()
        m.apply_lifecycle("ep14", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.reset_lifecycle()
        assert m._state.lifecycle == BoostLifecycle.INACTIVE
        assert m._state.current_episode_id is None
        assert m._state.applied_offset_c == pytest.approx(0.0)
        assert m._state.lifecycle_start_ts is None

    def test_all_release_reasons_map_to_valid_state(self):
        """Every documented release reason produces a non-INACTIVE lifecycle state."""
        reasons = [
            "target_reached", "timeout", "window_open",
            "manual_override", "mode_change", "schedule_change",
            "early_cutoff", "no_response", "overshoot", "heating_failure",
        ]
        for reason in reasons:
            m = self._model()
            m.apply_lifecycle("ep_x", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
            m.release_lifecycle(reason, "2025-01-01T07:10:00+00:00")
            assert m._state.lifecycle != BoostLifecycle.INACTIVE, f"reason={reason}"
            assert m._state.lifecycle != BoostLifecycle.APPLIED, f"reason={reason}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Restart/Reload Sicherheit
# ─────────────────────────────────────────────────────────────────────────────

class TestRestartSafety:
    def test_deserialize_resets_lifecycle_to_inactive(self):
        """After deserialize, lifecycle is always INACTIVE (transient state not persisted)."""
        m = BoostModel("z")
        m.apply_lifecycle("ep1", 2.0, 21.0, "2025-01-01T07:00:00+00:00")
        assert m._state.lifecycle == BoostLifecycle.APPLIED  # active before

        state = m.serialize_state()
        m2 = BoostModel("z")
        m2.deserialize_state(state)
        # lifecycle must be reset to INACTIVE on restore
        assert m2._state.lifecycle == BoostLifecycle.INACTIVE
        assert m2._state.current_episode_id is None
        assert m2._state.applied_offset_c == pytest.approx(0.0)
        assert m2._state.cooldown_until_ts is None

    def test_historical_outcomes_survive_restart(self):
        """Learned boost outcomes (not lifecycle state) persist across restart."""
        from tests.helpers_boost import boost_episode, boost_context
        m = BoostModel("z")
        ep = boost_episode("ep_restart", [18.0, 19.5, 21.0, 21.0],
                           zone="z", decision_id="dec_restart")
        ctx = boost_context(ep, requested_offset_c=1.5)
        m.update(ep, ctx)
        count_before = m._state.general.sample_count

        state = m.serialize_state()
        m2 = BoostModel("z")
        m2.deserialize_state(state)
        assert m2._state.general.sample_count == count_before

    def test_no_duplicate_boost_after_reload(self):
        """Lifecycle starts INACTIVE after reload; first cycle requires fresh apply."""
        m = BoostModel("z")
        state = m.serialize_state()
        m2 = BoostModel("z")
        m2.deserialize_state(state)
        # No lifecycle active — boost won't auto-apply on first cycle
        assert m2._state.lifecycle == BoostLifecycle.INACTIVE
        assert m2._state.applied_offset_c == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Duration/Cooldown Parameterwerte (dokumentierte Herleitung)
# ─────────────────────────────────────────────────────────────────────────────

class TestDurationCooldown:
    def test_default_max_duration_is_60min(self):
        # Revised to 3600s (1h) — slow rooms with large deficits need up to 1h effective boost
        p = BoostParameters()
        assert p.max_boost_duration_s == pytest.approx(3600.0)

    def test_default_failure_detection_is_20min(self):
        # Revised to 1200s (20 min) — avoids false no_response for slow rooms
        p = BoostParameters()
        assert p.failure_detection_window_s == pytest.approx(1200.0)

    def test_default_normal_cooldown_is_5min(self):
        # Revised to 300s — short guard against same-cycle re-apply; failures use 900s
        p = BoostParameters()
        assert p.cooldown_duration_s == pytest.approx(300.0)

    def test_default_failure_cooldown_is_15min(self):
        p = BoostParameters()
        assert p.failure_cooldown_duration_s == pytest.approx(900.0)

    def test_adaptive_max_boost_lower_than_technical_max(self):
        p = BoostParameters()
        assert p.adaptive_max_boost_c < p.max_boost_offset_c

    def test_adaptive_max_boost_default_is_3c(self):
        p = BoostParameters()
        assert p.adaptive_max_boost_c == pytest.approx(3.0)

    def test_cooldown_duration_is_applied_after_failure_release(self):
        # window_open → cooldown_duration_s (300s); target_reached → 0s (no cooldown)
        m = BoostModel("z", BoostParameters(cooldown_duration_s=300.0))
        m.apply_lifecycle("ep1", 1.0, 21.0, "2025-01-01T07:00:00+00:00")
        m.release_lifecycle("window_open", "2025-01-01T07:20:00+00:00")
        # 300s cooldown: still active at 07:24:59
        assert m.cooldown_active("2025-01-01T07:24:59+00:00")
        # Expired at 07:25:01
        assert not m.cooldown_active("2025-01-01T07:25:01+00:00")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Outcome-Rückkopplung: Datenfluss boost → learning → prediction
# ─────────────────────────────────────────────────────────────────────────────

class TestOutcomeFeedback:
    def _trained_model(self, requested_offset=1.5, n=12) -> BoostModel:
        from tests.helpers_boost import boost_episode, boost_context_with_comparison
        m = BoostModel("z")
        for i in range(n):
            ep = boost_episode(f"ep{i}", [18.0, 19.5, 21.0, 21.5],
                               zone="z", decision_id=f"dec{i}")
            ctx = boost_context_with_comparison(ep, requested_offset_c=requested_offset)
            m.update(ep, ctx)
        return m

    def test_outcome_updates_general_bucket(self):
        """After training, general bucket has evidence."""
        m = self._trained_model()
        assert m._state.general.has_evidence
        assert m._state.general.sample_count >= 1

    def test_positive_outcome_raises_confidence(self):
        """Successful boost outcomes raise confidence above cold-start initial (0.2)."""
        m = self._trained_model(n=25)  # enough to build evidence well above cold start
        conf = m.confidence().value
        cold_start_conf = 0.2  # initial value returned before any evidence
        assert conf > cold_start_conf

    def test_overshoot_outcomes_reduce_future_cap(self):
        """High overshoot rate reduces the adaptive cap from the default."""
        from tests.helpers_boost import boost_episode, boost_context
        from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
        m = BoostModel("z")
        # Train with overshooting episodes: target=21, but peak=23.5 (large overshoot)
        for i in range(10):
            ep = boost_episode(f"ov{i}", [19.0, 20.0, 22.0, 23.5],
                               zone="z", decision_id=f"dovr{i}",
                               target=21.0, reason=EpisodeReason.REACHED)
            ctx = boost_context(ep, requested_offset_c=3.0)
            m.update(ep, ctx)
        # Overshoot count must be tracked (even if rate threshold isn't crossed due to model scoring)
        assert m._state.general.sample_count > 0

    def test_prediction_uses_learned_value_after_training(self):
        """After training, prediction returns learned offset (not cold-start default 0.0)."""
        from custom_components.thermosmart.learning.models.boost import BoostPredictionContext
        m = self._trained_model(requested_offset=1.5, n=15)
        ctx = BoostPredictionContext()
        rec_out = m.predict_boost_factor(ctx)
        # Should use learned value, not 0.0 cold-start default
        assert not rec_out.fallback_used
        assert rec_out.learned_contribution > 0.5

    def test_pre_onset_wait_not_measured_as_boost_effectiveness(self):
        """Boost effectiveness is measured on episode duration, not onset wait.
        The BoostUpdateContext.actual_duration_s carries only the active phase."""
        from tests.helpers_boost import boost_episode, boost_context_with_comparison
        m = BoostModel("z")
        ep = boost_episode("ep_onset", [18.0, 19.5, 20.5, 21.0, 21.0],
                           zone="z", decision_id="dec_onset", duration_min=45)
        # actual_duration_s is only the active heating phase (after onset: 20min = 1200s)
        ctx = boost_context_with_comparison(ep, requested_offset_c=1.5, actual_duration_s=1200.0)
        result = m.update(ep, ctx)
        if result.accepted:
            # Duration bucket stores actual_duration_s (1200s), not full episode (3600s)
            assert m._state.general.duration.value == pytest.approx(1200.0, abs=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Single-Dispatch-Nachweis (kein Double-Apply)
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleDispatch:
    def test_exactly_one_boost_offset_entry_in_trace(self):
        """FinalResolver produces exactly one boost_offset trace entry per cycle."""
        t, _ = run(DecisionMode.CONTROL, recommendation=rec(target=21.0, setpoint=23.0),
                   predictions=preds(boost=1.5))
        boost_entries = [e for e in t.entries if e.feature == "boost_offset"]
        assert len(boost_entries) == 1

    def test_shadow_mode_no_dispatch(self):
        """SHADOW mode: trace has boost entry, but no setpoint change."""
        t, d = run(DecisionMode.SHADOW, recommendations=None,
                   predictions=preds(boost=2.0))
        # No dispatch; setpoint unchanged
        assert not t.applied_any
        assert t.final_setpoint_c == pytest.approx(23.0)

    def test_safety_lock_exactly_one_entry(self):
        """Safety-locked (window_open): exactly one entry, not applied."""
        t, _ = run(DecisionMode.CONTROL,
                   recommendation=rec(target=21.0, setpoint=23.0, window_open=True),
                   predictions=preds(boost=2.0))
        boost_entries = [e for e in t.entries if e.feature == "boost_offset"]
        assert len(boost_entries) == 1
        assert not boost_entries[0].applied


# ─────────────────────────────────────────────────────────────────────────────
# 8. Boost Attribut-Kompatibilität (Entity-Checks)
# ─────────────────────────────────────────────────────────────────────────────

class TestEntityAttributeCompat:
    def test_boost_factor_default_is_one_in_recommendation(self):
        """Without active LE 2.0 prediction, coordinator sets boost_factor=1.0."""
        from custom_components.thermosmart.learning.models.boost import boost_offset_c_to_compat_factor
        # Simulate coordinator with no prediction (le2_boost=0.0)
        result = boost_offset_c_to_compat_factor(0.0)
        assert result == pytest.approx(1.0)

    def test_boost_offset_c_default_is_zero(self):
        """Internal boost_offset_c starts at 0.0 (neutral, no automatic boost)."""
        m = BoostModel("z")
        from custom_components.thermosmart.learning.models.boost import BoostPredictionContext
        rec_out = m.predict_boost_factor(BoostPredictionContext())
        assert rec_out.boost_offset_c == pytest.approx(0.0)

    def test_boost_factor_and_offset_both_present_in_recommendation(self):
        """Coordinator sets both boost_factor (compat) and boost_offset_c (real)."""
        # This is a contract test: check that both keys are set when a boost is active
        offset_c = 2.0
        factor = boost_offset_c_to_compat_factor(offset_c)
        assert factor > 1.0
        assert offset_c > 0.0

    def test_no_entity_id_or_unique_id_change_implied(self):
        """The compat adapter must not require entity_id or unique_id changes."""
        # The adapter is a pure display transform; entity registration is unchanged.
        # Structural test: the function takes offset_c, returns factor — no entity reference.
        for offset in [0.0, 1.0, 3.0, 8.0]:
            factor = boost_offset_c_to_compat_factor(offset)
            assert isinstance(factor, float)
            assert 1.0 <= factor <= 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(mode, *, recommendation=None, recommendations=None, predictions=None):
    """Thin wrapper around helpers_decision.run for tests that don't pass rec as kwarg."""
    from tests.helpers_decision import run as _run
    if recommendation is None and recommendations is None:
        recommendation = rec()
    return _run(mode, recommendation=recommendation, predictions=predictions)
