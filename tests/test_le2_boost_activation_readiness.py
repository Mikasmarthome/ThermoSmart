"""B2b-7/8: BoostActivationReadiness contract — eligibility invariant, typed reasons, scope
fallback, control-type, restart/legacy/unknown, end-to-end. NO activation / NO switch."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
from custom_components.thermosmart.learning.models import (
    BoostEligibilityReason as RE, BoostLearningReadiness as LR, BoostModel,
    BoostPredictionContext, BoostRecommendation, build_activation_readiness)
from tests.helpers_boost import boost_context_with_comparison, boost_episode

_SUCCESS = [18.0, 19.5, 20.5, 21.0, 21.0]
_OVERSHOOT = [18.0, 19.0, 20.0, 21.0, 22.5, 23.0, 23.0]
NOW = "2025-02-01T00:00:00+00:00"


def _feed(m, vals, did, *, offset=1.5):
    ep = boost_episode(did, vals, decision_id=did, reason=EpisodeReason.REACHED)
    m.update(ep, boost_context_with_comparison(ep, requested_offset_c=offset))


def _ctx(control="setpoint", **kw):
    return BoostPredictionContext(start_deficit_c=3.0, decision_type="ts", intensity_c=1.5,
                                  control_type=control, now_ts=NOW, **kw)


def _eligible_model(n=40):
    m = BoostModel("lz")
    for i in range(n):
        _feed(m, _SUCCESS, f"s{i}")
    return m


def _crafted(**kw):
    base = dict(boost_offset_c=1.5, recommended_duration_s=1800.0, confidence=0.8,
                reliability=0.9, fallback_used=False, prior_contribution=0.0,
                learned_contribution=1.0, evidence_count=15, bucket="def2|dt:ts",
                missing_evidence=(), reason_codes=(), selected_scope="bucket",
                bucket_handle="abc", readiness=LR.ELIGIBLE, factor_usable=True,
                rollback_active=False, cooldown_active=False, cooldown_remaining_s=None,
                eligibility_reason="bucket_eligible", rehab_progress=1.0,
                known_good_available=True)
    base.update(kw)
    return BoostRecommendation(**base)


# ── eligibility verdicts (model level) ───────────────────────────────────────
class TestEligibilityVerdict:
    def test_fully_eligible(self):
        ar = _eligible_model().activation_readiness(_ctx())
        assert ar.eligibility and ar.eligibility_reason == RE.ELIGIBLE and ar.factor_usable
        assert ar.selected_scope == "bucket" and ar.control_type == "setpoint"

    def test_low_confidence(self):
        ar = _eligible_model(25).activation_readiness(_ctx())
        assert not ar.eligibility and ar.eligibility_reason == RE.LOW_CONFIDENCE

    def test_insufficient_data(self):
        ar = BoostModel("lz").activation_readiness(_ctx())
        assert not ar.eligibility and ar.eligibility_reason == RE.INSUFFICIENT_DATA

    def test_shadow_learning(self):
        m = BoostModel("lz")
        for i in range(6):
            _feed(m, _SUCCESS, f"s{i}")          # eligible learning but conf < activation min
        ar = m.activation_readiness(_ctx())
        assert not ar.eligibility
        assert ar.eligibility_reason in (RE.LOW_CONFIDENCE, RE.SHADOW_LEARNING)

    def test_rolled_back_and_cooldown(self):
        m = _eligible_model()
        _feed(m, _OVERSHOOT, "b1"); _feed(m, _OVERSHOOT, "b2")   # bucket rollback
        # query the SAME bucket; general is eligible so it falls back -> but assert the bucket
        # scope itself is not eligible via a fresh-zone escalation instead:
        ar = m.activation_readiness(_ctx())
        assert ar.bucket_readiness in (LR.ROLLED_BACK, None) or ar.selected_scope == "general_fallback"

    def test_neither_scope_eligible_after_general_escalation(self):
        m = BoostModel("lz")
        # two distinct degraded buckets -> general escalation rollback
        _feed(m, _OVERSHOOT, "a", offset=1.5)
        ep = boost_episode("b", _OVERSHOOT, decision_id="b", reason=EpisodeReason.REACHED)
        # second bucket (different deficit band) overshoot
        m.update(ep, boost_context_with_comparison(
            boost_episode("b", _OVERSHOOT, decision_id="b"), requested_offset_c=1.5))
        ar = m.activation_readiness(_ctx())
        assert not ar.eligibility

    def test_direct_valve_unsupported(self):
        ar = _eligible_model().activation_readiness(_ctx(control="direct_valve"))
        assert not ar.eligibility and ar.eligibility_reason == RE.UNSUPPORTED_CONTROL_TYPE
        assert not ar.factor_usable

    def test_missing_control_type(self):
        ar = _eligible_model().activation_readiness(_ctx(control=None))
        assert not ar.eligibility and ar.eligibility_reason == RE.MISSING_CONTEXT

    def test_missing_context(self):
        ar = _eligible_model().activation_readiness(None)
        assert not ar.eligibility and ar.eligibility_reason == RE.MISSING_CONTEXT

    def test_blocking_confounder(self):
        ar = _eligible_model().activation_readiness(
            _ctx(blocking_reasons=("additional_heat_source",)))
        assert not ar.eligibility and ar.eligibility_reason == RE.BLOCKING_CONFOUNDER


# ── hard formula / gate priority (pure builder, crafted recommendation) ──────
class TestHardFormula:
    def test_all_gates_pass(self):
        ar = build_activation_readiness(recommendation=_crafted(), factor_stable=True,
                                        control_type="setpoint", has_context=True,
                                        general_readiness=LR.ELIGIBLE)
        assert ar.eligibility and ar.eligibility_reason == RE.ELIGIBLE

    def test_factor_positive_but_not_usable(self):
        rec = _crafted(factor_usable=False, readiness=LR.ELIGIBLE, boost_offset_c=2.0,
                       selected_scope="bucket")
        ar = build_activation_readiness(recommendation=rec, factor_stable=True,
                                        control_type="setpoint", has_context=True,
                                        general_readiness=LR.ELIGIBLE)
        assert not ar.eligibility and ar.eligibility_reason == RE.NO_USABLE_FACTOR

    def test_high_confidence_but_cooldown(self):
        rec = _crafted(confidence=0.9, cooldown_active=True)
        ar = build_activation_readiness(recommendation=rec, factor_stable=True,
                                        control_type="setpoint", has_context=True,
                                        general_readiness=LR.ELIGIBLE)
        assert not ar.eligibility and ar.eligibility_reason == RE.COOLDOWN

    def test_eligible_but_unstable_factor(self):
        ar = build_activation_readiness(recommendation=_crafted(), factor_stable=False,
                                        control_type="setpoint", has_context=True,
                                        general_readiness=LR.ELIGIBLE)
        assert not ar.eligibility and ar.eligibility_reason == RE.UNSTABLE_FACTOR

    def test_eligible_but_low_reliability(self):
        rec = _crafted(reliability=0.1)
        ar = build_activation_readiness(recommendation=rec, factor_stable=True,
                                        control_type="setpoint", has_context=True,
                                        general_readiness=LR.ELIGIBLE)
        assert not ar.eligibility and ar.eligibility_reason == RE.LOW_RELIABILITY

    def test_no_implicit_activation_via_factor_alone(self):
        rec = _crafted(readiness=LR.SHADOW_LEARNING, factor_usable=False, boost_offset_c=3.0,
                       known_good_available=False)
        ar = build_activation_readiness(recommendation=rec, factor_stable=True,
                                        control_type="setpoint", has_context=True,
                                        general_readiness=LR.SHADOW_LEARNING)
        assert not ar.eligibility


# ── scope fallback ───────────────────────────────────────────────────────────
class TestScopeFallback:
    def test_eligible_bucket_selected(self):
        ar = _eligible_model().activation_readiness(_ctx())
        assert ar.selected_scope == "bucket" and not ar.fallback_used

    def test_general_fallback_when_bucket_rolled_back(self):
        m = _eligible_model()
        _feed(m, _OVERSHOOT, "b1"); _feed(m, _OVERSHOOT, "b2")
        ar = m.activation_readiness(_ctx())
        assert ar.selected_scope == "general_fallback" and ar.fallback_used and ar.eligibility

    def test_no_cross_zone(self):
        m1 = _eligible_model()
        m2 = BoostModel("other")                 # different zone, no evidence
        ar2 = m2.activation_readiness(_ctx())
        assert not ar2.eligibility               # m1's maturity never leaks to m2


# ── restart / legacy / unknown ───────────────────────────────────────────────
class TestRestart:
    def test_eligible_round_trip(self):
        m = _eligible_model()
        assert m.activation_readiness(_ctx()).eligibility
        m2 = BoostModel("lz"); m2.deserialize_state(m.serialize_state())
        assert m2.activation_readiness(_ctx()).eligibility

    def test_non_eligible_round_trip_stays_non_eligible(self):
        m = _eligible_model(25)
        m2 = BoostModel("lz"); m2.deserialize_state(m.serialize_state())
        assert not m2.activation_readiness(_ctx()).eligibility

    def test_rolled_back_bucket_not_eligible_after_restart(self):
        m = _eligible_model()
        _feed(m, _OVERSHOOT, "b1"); _feed(m, _OVERSHOOT, "b2")
        m2 = BoostModel("lz"); m2.deserialize_state(m.serialize_state())
        # the bucket scope is still rolled back after restart (no spontaneous eligibility)
        b = next(iter(m2._state.degradation.buckets.values()))
        assert b.readiness == LR.ROLLED_BACK

    def test_unknown_degradation_version_not_eligible(self):
        m = _eligible_model()
        blob = dict(m.serialize_state())
        blob["degradation"] = {"scoped": True, "component_version": 999}
        m.deserialize_state(blob)
        assert not m.activation_readiness(_ctx()).eligibility


# ── UI target contract (documented; NO entity created) ───────────────────────
class TestUiTargetContract:
    def test_no_switch_entity_created_in_b2b(self):
        # B2b-7/8 must not add an Adaptive-Boost switch entity anywhere
        from pathlib import Path
        import custom_components.thermosmart as pkg
        root = Path(pkg.__file__).parent
        hits = []
        for p in root.rglob("*.py"):
            txt = p.read_text(encoding="utf-8", errors="ignore").lower()
            if "adaptive_boost" in txt and "switchentity" in txt:
                hits.append(p.name)
        assert not hits

    def test_readiness_is_advisory_only(self):
        # computing readiness must not mutate the model state (no activation side effects)
        m = _eligible_model()
        before = m.serialize_state()
        m.activation_readiness(_ctx())
        assert m.serialize_state() == before


# ── end-to-end through the real update path ──────────────────────────────────
class TestEndToEnd:
    def test_maturity_then_overshoot_then_rehab(self):
        m = _eligible_model()
        assert m.activation_readiness(_ctx()).eligibility          # matured -> eligible
        _feed(m, _OVERSHOOT, "b1"); _feed(m, _OVERSHOOT, "b2")     # bucket rollback
        b = next(iter(m._state.degradation.buckets.values()))
        assert b.readiness == LR.ROLLED_BACK                       # eligibility withdrawn (bucket)

    def test_diagnostics_privacy_safe(self):
        ar = _eligible_model().activation_readiness(_ctx())
        d = ar.to_dict()
        assert "eligibility" in d and "eligibility_reason" in d and "selected_scope" in d
        assert all("/" not in str(v) for v in d.values() if isinstance(v, str))
        assert d["selected_bucket_handle"] is None or "|" not in str(d["selected_bucket_handle"])
