"""B2b-5a: bucket-scoped rollback, general escalation, per-scope rehab, deterministic
re-eligibility with hysteresis + factor stability, bounds."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.models import (
    BoostDegradationState, BoostLearningReadiness as R, BoostScopedDegradation,
    DegradationParams, is_scope_eligible, observe_scoped, recompute_readiness,
    resolve_scoped_cooldown)

T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
P = DegradationParams()


def _ts(minutes):
    return (T0 + timedelta(minutes=minutes)).isoformat()


def obs(scoped, bucket, cls, *, overshoot=None, rel=0.9, attr=0.9, minutes=0, factor=1.5):
    return observe_scoped(
        scoped, bucket_key=bucket, outcome_class=cls, overshoot_c=overshoot, reliability=rel,
        attribution=attr, adaptive=True, reliability_known=True, now_ts=_ts(minutes),
        bucket_factor=factor, bucket_confidence=0.7, general_factor=factor,
        general_confidence=0.7, params=P)


# ── local scope isolation ────────────────────────────────────────────────────
class TestLocalScope:
    def test_one_bucket_rolls_back_only_itself(self):
        s = BoostScopedDegradation()
        # bucket A gets an extreme overshoot -> A rolled back; B untouched
        r = obs(s, "A", "boost_overshoot", overshoot=2.0, minutes=0)
        assert r.bucket_action.triggered and r.scoped.buckets["A"].readiness == R.ROLLED_BACK
        # general not rolled back by a single local fault
        assert not r.general_action.triggered
        assert "B" not in r.scoped.buckets

    def test_other_bucket_stays_eligible(self):
        s = BoostScopedDegradation()
        # make B eligible+stable first
        for i in range(6):
            r = obs(s, "B", "boost_success", minutes=i * 30); s = r.scoped
        b_before = s.buckets["B"]
        # A degrades
        r = obs(s, "A", "boost_overshoot", overshoot=2.0, minutes=300); s = r.scoped
        assert s.buckets["A"].readiness == R.ROLLED_BACK
        assert s.buckets["B"].readiness == b_before.readiness         # B unchanged
        assert s.buckets["B"].known_good == b_before.known_good

    def test_local_rollback_uses_local_known_good_target(self):
        s = BoostScopedDegradation()
        for i in range(6):
            r = obs(s, "A", "boost_success", minutes=i * 30); s = r.scoped
        kg = s.buckets["A"].known_good
        assert kg is not None
        r = obs(s, "A", "boost_overshoot", overshoot=2.0, minutes=300); s = r.scoped
        assert r.bucket_action.triggered and r.bucket_action.target_known_good is not None
        assert r.bucket_action.target_factor == kg.factor

    def test_local_rollback_without_known_good_is_zero(self):
        s = BoostScopedDegradation()
        r = obs(s, "A", "boost_overshoot", overshoot=2.0, minutes=0)
        assert r.bucket_action.triggered and r.bucket_action.target_factor is None


# ── general escalation ───────────────────────────────────────────────────────
class TestEscalation:
    def test_single_local_fault_no_general_rollback(self):
        s = BoostScopedDegradation()
        r = obs(s, "A", "boost_overshoot", overshoot=2.0, minutes=0)
        assert not r.general_action.triggered
        assert r.scoped.general.readiness != R.ROLLED_BACK

    def test_two_distinct_degraded_buckets_escalate(self):
        s = BoostScopedDegradation()
        r = obs(s, "A", "boost_overshoot", overshoot=2.0, minutes=0); s = r.scoped
        r = obs(s, "B", "boost_overshoot", overshoot=2.0, minutes=30)
        assert r.general_action.triggered and r.general_action.reason == "escalation_multi_bucket"
        assert r.scoped.general.readiness == R.ROLLED_BACK

    def test_general_not_degraded_by_single_bucket(self):
        # a single local fault must NOT degrade/roll back general (keeps general fallback valid)
        s = BoostScopedDegradation()
        r = obs(s, "A", "boost_overshoot", overshoot=2.0, minutes=0)
        assert r.scoped.general.readiness not in (R.ROLLED_BACK, R.DEGRADED)


# ── per-scope rehabilitation ─────────────────────────────────────────────────
class TestScopedRehab:
    def test_other_bucket_success_does_not_rehab_local(self):
        s = BoostScopedDegradation()
        r = obs(s, "A", "boost_overshoot", overshoot=2.0, minutes=0); s = r.scoped
        # clean successes in B must NOT rehabilitate A
        for i in range(5):
            r = obs(s, "B", "boost_success", minutes=10 + i * 10); s = r.scoped
        assert s.buckets["A"].rehab_clean_count == 0

    def test_local_clean_evidence_rehabilitates_local(self):
        s = BoostScopedDegradation()
        r = obs(s, "A", "boost_overshoot", overshoot=2.0, minutes=0); s = r.scoped
        far = P.cooldown_max_s / 60.0 + 10
        for i in range(P.rehab_required_clean):
            r = obs(s, "A", "boost_success", minutes=far + i * 30); s = r.scoped
        s = resolve_scoped_cooldown(s, _ts(far + P.rehab_required_clean * 30 + 1), P)
        assert s.buckets["A"].readiness == R.SHADOW_LEARNING   # not directly ELIGIBLE


# ── deterministic re-eligibility / hysteresis / stability ────────────────────
class TestReEligibility:
    def test_hysteresis_no_flapping(self):
        # at a score in the band [recovery, degraded) a DEGRADED scope stays DEGRADED
        st = BoostDegradationState(readiness=R.DEGRADED, degradation_score=1.0,
                                   adaptive_outcomes=10, factor_history=(1.5, 1.5, 1.5))
        out = recompute_readiness(st, confidence=0.7, params=P)
        assert out.readiness == R.DEGRADED               # 0.75 <= 1.0 < 1.5 -> stays

    def test_recovers_below_recovery_threshold(self):
        st = BoostDegradationState(readiness=R.DEGRADED, degradation_score=0.5,
                                   adaptive_outcomes=10, factor_history=(1.5, 1.5, 1.5, 1.5))
        out = recompute_readiness(st, confidence=0.7, params=P)
        assert out.readiness == R.ELIGIBLE

    def test_unstable_factor_blocks_eligible(self):
        st = BoostDegradationState(readiness=R.SHADOW_LEARNING, degradation_score=0.0,
                                   adaptive_outcomes=10, factor_history=(0.5, 2.5, 0.5))
        out = recompute_readiness(st, confidence=0.7, params=P)
        assert out.readiness == R.SHADOW_LEARNING        # spread 2.0 > 0.5 -> not stable

    def test_low_confidence_blocks_eligible(self):
        st = BoostDegradationState(readiness=R.SHADOW_LEARNING, degradation_score=0.0,
                                   adaptive_outcomes=10, factor_history=(1.5, 1.5, 1.5))
        out = recompute_readiness(st, confidence=0.2, params=P)
        assert out.readiness == R.SHADOW_LEARNING

    def test_is_scope_eligible_gate(self):
        ok = BoostDegradationState(readiness=R.ELIGIBLE, degradation_score=0.0,
                                   overshoot_evidence=0.0)
        assert is_scope_eligible(ok, _ts(0), P)
        cd = BoostDegradationState(readiness=R.ELIGIBLE, cooldown_until_ts=_ts(100))
        assert not is_scope_eligible(cd, _ts(0), P)       # in cooldown -> not eligible


# ── persistence + bounds ─────────────────────────────────────────────────────
class TestPersistenceBounds:
    def test_scoped_round_trip(self):
        s = BoostScopedDegradation()
        r = obs(s, "A", "boost_overshoot", overshoot=2.0, minutes=0); s = r.scoped
        s2 = BoostScopedDegradation.from_dict(s.to_dict())
        assert s2.buckets["A"].readiness == s.buckets["A"].readiness
        assert s2.general.readiness == s.general.readiness

    def test_legacy_single_dict_becomes_general(self):
        s = BoostScopedDegradation.from_dict({"readiness": "degraded", "component_version": 1})
        assert s.general.readiness == R.DEGRADED and not s.buckets

    def test_unknown_version_safe(self):
        s = BoostScopedDegradation.from_dict({"scoped": True, "component_version": 999})
        assert s.general.readiness == R.INSUFFICIENT_DATA

    def test_corrupt_bucket_isolated(self):
        d = BoostScopedDegradation().to_dict()
        d["buckets"] = {"A": {"component_version": 999}, "B": BoostDegradationState(
            readiness="degraded").to_dict()}
        s = BoostScopedDegradation.from_dict(d)
        assert "A" not in s.buckets and "B" in s.buckets

    def test_bucket_count_cap(self):
        p = DegradationParams(bucket_state_cap=3)
        s = BoostScopedDegradation()
        for i in range(8):
            s = observe_scoped(s, bucket_key=f"b{i}", outcome_class="boost_no_effect",
                               overshoot_c=None, reliability=0.9, attribution=0.9, adaptive=True,
                               reliability_known=True, now_ts=_ts(i * 10), bucket_factor=1.0,
                               bucket_confidence=0.7, general_factor=1.0, general_confidence=0.7,
                               params=p).scoped
        assert len(s.buckets) <= 3
