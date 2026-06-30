"""B2b-5: degradation detection, asymmetric evidence, bounded rollback, cooldown, rehab."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.models import (
    BoostDegradationState, BoostLearningReadiness as R, DegradationParams, KnownGoodState,
    maybe_capture_known_good, observe_degradation, resolve_cooldown)

T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
P = DegradationParams()


def _ts(minutes):
    return (T0 + timedelta(minutes=minutes)).isoformat()


def obs(state, cls, *, overshoot=None, rel=0.9, attr=0.9, minutes=0, adaptive=True,
        known=True, factor=1.5):
    return observe_degradation(
        state, outcome_class=cls, overshoot_c=overshoot, reliability=rel, attribution=attr,
        adaptive=adaptive, reliability_known=known, now_ts=_ts(minutes), current_factor=factor,
        current_dim={"value": factor, "effective_n": 12.0, "dispersion": 0.1},
        confidence=0.7, params=P)


# ── degradation detection ────────────────────────────────────────────────────
class TestDegradation:
    def test_single_no_effect_no_rollback(self):
        s, rb = obs(BoostDegradationState(), "boost_no_effect", minutes=0)
        assert not rb.triggered and s.readiness != R.ROLLED_BACK

    def test_repeated_no_effect_degrades(self):
        s = BoostDegradationState()
        for i in range(6):
            s, rb = obs(s, "boost_no_effect", minutes=i * 30)
        assert not rb.triggered
        assert s.readiness in (R.DEGRADED,)        # consecutive negatives -> DEGRADED

    def test_single_moderate_overshoot_no_rollback(self):
        s, rb = obs(BoostDegradationState(), "boost_overshoot", overshoot=0.6, minutes=0)
        assert not rb.triggered
        assert s.overshoot_evidence > 0.0

    def test_two_reliable_overshoots_rollback(self):
        s = BoostDegradationState()
        s, rb1 = obs(s, "boost_overshoot", overshoot=0.7, minutes=0)
        s, rb2 = obs(s, "boost_overshoot", overshoot=0.7, minutes=60)
        assert rb2.triggered and rb2.fast_safety
        assert s.readiness == R.ROLLED_BACK

    def test_extreme_overshoot_fast_trigger(self):
        s, rb = obs(BoostDegradationState(), "boost_overshoot", overshoot=2.0, minutes=0)
        assert rb.triggered and rb.fast_safety and rb.reason == "fast_safety_overshoot"

    def test_neutral_classes_no_evidence(self):
        s = BoostDegradationState()
        for cls in ("boost_confounded", "boost_interrupted", "boost_partial_dispatch",
                    "boost_failed_dispatch", "boost_not_attempted", "insufficient_comparison"):
            s, rb = obs(s, cls, overshoot=2.0, minutes=0)
            assert not rb.triggered
        assert s.degradation_score == 0.0 and s.overshoot_evidence == 0.0

    def test_weak_reliability_no_strong_degradation(self):
        s = BoostDegradationState()
        # two LOW-reliability overshoots must NOT force a fast rollback
        s, rb1 = obs(s, "boost_overshoot", overshoot=2.0, rel=0.2, minutes=0)
        s, rb2 = obs(s, "boost_overshoot", overshoot=2.0, rel=0.2, minutes=60)
        assert not rb1.triggered and not rb2.triggered

    def test_non_adaptive_outcome_ignored(self):
        s, rb = obs(BoostDegradationState(), "boost_overshoot", overshoot=2.0, adaptive=False)
        assert not rb.triggered and s.overshoot_evidence == 0.0


# ── asymmetry ────────────────────────────────────────────────────────────────
class TestAsymmetry:
    def test_success_raises_slowly_overshoot_drops_fast(self):
        # one overshoot weight vs one success relief (scaled) -> overshoot dominates
        s_o, _ = obs(BoostDegradationState(), "boost_overshoot", overshoot=0.6, minutes=0)
        s_s, _ = obs(BoostDegradationState(), "boost_success", minutes=0)
        assert s_o.degradation_score > s_s.weighted_positive * P.positive_scale

    def test_alternating_success_overshoot_does_not_clear_safety(self):
        s = BoostDegradationState()
        for i in range(4):
            s, _ = obs(s, "boost_success", minutes=i * 120)
            s, rb = obs(s, "boost_overshoot", overshoot=0.7, minutes=i * 120 + 60)
            if rb.triggered:
                break
        # safety evidence accumulates despite interleaved successes -> eventual rollback
        assert rb.triggered or s.overshoot_evidence > 0.5

    def test_success_never_zeroes_recent_overshoot(self):
        s, _ = obs(BoostDegradationState(), "boost_overshoot", overshoot=0.7, minutes=0)
        s2, _ = obs(s, "boost_success", minutes=10)
        assert s2.overshoot_evidence == pytest.approx(s.overshoot_evidence, rel=0.05)


# ── known good ───────────────────────────────────────────────────────────────
class TestKnownGood:
    def _eligible_state(self):
        s = BoostDegradationState(readiness=R.ELIGIBLE, adaptive_outcomes=8)
        return s

    def test_capture_from_eligible_stable(self):
        s = maybe_capture_known_good(
            self._eligible_state(), factor=1.4, effective_n=12.0, dispersion=0.1,
            confidence=0.7, sample_weight=0.9, now_ts=_ts(0), model_version=1,
            provenance={}, params=P)
        assert s.known_good is not None and s.known_good.factor == 1.4

    def test_no_capture_when_not_eligible(self):
        s = maybe_capture_known_good(
            BoostDegradationState(readiness=R.SHADOW_LEARNING, adaptive_outcomes=8),
            factor=1.4, effective_n=12.0, dispersion=0.1, confidence=0.7, sample_weight=0.9,
            now_ts=_ts(0), model_version=1, provenance={}, params=P)
        assert s.known_good is None

    def test_no_capture_with_safety_evidence(self):
        s = BoostDegradationState(readiness=R.ELIGIBLE, adaptive_outcomes=8,
                                  overshoot_evidence=1.0)
        s = maybe_capture_known_good(
            s, factor=1.4, effective_n=12.0, dispersion=0.1, confidence=0.7, sample_weight=0.9,
            now_ts=_ts(0), model_version=1, provenance={}, params=P)
        assert s.known_good is None

    def test_rollback_restores_known_good(self):
        kg = KnownGoodState(factor=0.8, effective_n=10.0, dispersion=0.1, confidence=0.65,
                            sample_weight=0.9, ts=_ts(0), model_version=1)
        s = BoostDegradationState(known_good=kg)
        s, rb = obs(s, "boost_overshoot", overshoot=2.0, minutes=0)
        assert rb.triggered and rb.target_factor == 0.8 and rb.target_known_good is kg

    def test_rollback_without_known_good_is_zero_prior(self):
        s, rb = obs(BoostDegradationState(), "boost_overshoot", overshoot=2.0, minutes=0)
        assert rb.triggered and rb.target_factor is None     # -> zero-adaptive safe prior


# ── cooldown + rehabilitation ────────────────────────────────────────────────
class TestCooldown:
    def _rolled(self):
        s, rb = obs(BoostDegradationState(), "boost_overshoot", overshoot=2.0, minutes=0)
        return s, rb

    def test_rollback_starts_cooldown(self):
        s, rb = self._rolled()
        assert rb.cooldown_until_ts is not None and s.cooldown_until_ts == rb.cooldown_until_ts
        assert s.readiness == R.ROLLED_BACK

    def test_not_eligible_before_cooldown_elapsed(self):
        s, _ = self._rolled()
        r = resolve_cooldown(s, _ts(10), P)           # well within cooldown
        assert r.readiness == R.COOLDOWN

    def test_elapsed_without_rehab_not_eligible(self):
        s, _ = self._rolled()
        far = (T0 + timedelta(seconds=P.cooldown_max_s + 10)).isoformat()
        r = resolve_cooldown(s, far, P)
        assert r.readiness == R.COOLDOWN              # time alone is not enough

    def test_elapsed_with_rehab_returns_shadow(self):
        s, _ = self._rolled()
        # accrue clean rehabilitation evidence AFTER cooldown elapses
        t = P.cooldown_max_s / 60.0 + 10
        for i in range(P.rehab_required_clean):
            s, _ = obs(s, "boost_success", minutes=t + i * 60)
        r = resolve_cooldown(s, _ts(t + P.rehab_required_clean * 60 + 1), P)
        assert r.readiness == R.SHADOW_LEARNING       # never ROLLED_BACK -> ELIGIBLE directly

    def test_repeated_rollbacks_increase_cooldown_bounded(self):
        s, rb1 = self._rolled()
        d1 = (datetime.fromisoformat(rb1.cooldown_until_ts) - T0).total_seconds()
        # second rollback after the first (simulate later)
        s2 = BoostDegradationState(rollback_count=3)
        s2, rb2 = obs(s2, "boost_overshoot", overshoot=2.0, minutes=0)
        d2 = (datetime.fromisoformat(rb2.cooldown_until_ts) - T0).total_seconds()
        assert d2 >= d1 and d2 <= P.cooldown_max_s     # bounded by max
