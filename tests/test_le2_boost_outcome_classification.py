"""B2b-2: discrete boost outcome classification + per-class model effect + storage."""
from __future__ import annotations

from dataclasses import replace

import pytest

from custom_components.thermosmart.learning.models import (
    BoostModel, BoostOutcomeClass as C, BoostParameters, class_effect,
    classify_boost_outcome)
from custom_components.thermosmart.learning.models.boost import _sample_dict, _sample_from
from tests.helpers_boost import boost_episode, boost_context, boost_context_with_comparison

P = BoostParameters()


def _cls(**kw):
    base = dict(dispatch_status="fully_succeeded", authority="live_record",
                confounder_flags=(), attribution=0.8, has_temp=True, overshoot_c=0.0,
                has_baseline=True, relative_benefit=0.3, target_reached=True, params=P,
                is_partial=False)
    base.update(kw)
    return classify_boost_outcome(**base)


class TestEachClass:
    def test_success(self):
        assert _cls() is C.BOOST_SUCCESS

    def test_failed(self):
        assert _cls(dispatch_status="failed") is C.BOOST_FAILED_DISPATCH

    def test_not_attempted(self):
        assert _cls(dispatch_status="not_attempted") is C.BOOST_NOT_ATTEMPTED

    def test_partial(self):
        assert _cls(is_partial=True) is C.BOOST_PARTIAL_DISPATCH

    def test_interrupted(self):
        assert _cls(confounder_flags=("sensor_gap",)) is C.BOOST_INTERRUPTED

    def test_confounded_flag(self):
        assert _cls(confounder_flags=("solar_gain",)) is C.BOOST_CONFOUNDED

    def test_confounded_low_attribution(self):
        assert _cls(attribution=0.1) is C.BOOST_CONFOUNDED

    def test_overshoot(self):
        assert _cls(overshoot_c=1.0) is C.BOOST_OVERSHOOT

    def test_insufficient_comparison(self):
        assert _cls(has_baseline=False) is C.INSUFFICIENT_COMPARISON

    def test_unnecessary(self):
        assert _cls(relative_benefit=-0.1, target_reached=True) is C.BOOST_UNNECESSARY

    def test_no_effect(self):
        assert _cls(relative_benefit=0.03) is C.BOOST_NO_EFFECT


class TestPriority:
    def test_overshoot_beats_good_target(self):
        assert _cls(overshoot_c=1.0, relative_benefit=0.9) is C.BOOST_OVERSHOOT

    def test_partial_beats_apparent_success(self):
        assert _cls(is_partial=True, relative_benefit=0.9) is C.BOOST_PARTIAL_DISPATCH

    def test_confounder_beats_apparent_success(self):
        assert _cls(confounder_flags=("solar_gain",), relative_benefit=0.9) is C.BOOST_CONFOUNDED

    def test_no_baseline_fast_arrival_is_insufficient_not_success(self):
        assert _cls(has_baseline=False, relative_benefit=0.9) is C.INSUFFICIENT_COMPARISON

    def test_overshoot_detectable_without_baseline(self):
        assert _cls(has_baseline=False, overshoot_c=1.0) is C.BOOST_OVERSHOOT

    def test_confounded_beats_overshoot(self):
        assert _cls(confounder_flags=("solar_gain",), overshoot_c=1.0) is C.BOOST_CONFOUNDED


class TestModelEffectMatrix:
    def test_success_can_grow_evidence(self):
        m = BoostModel("lz")
        before = m._state.general.effective_factor.effective_n
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1")
        ctx = replace(boost_context(ep, requested_offset_c=1.5),
                      authority="live_record", dispatch_status="fully_succeeded",
                      outcome_reliability="full", expected_non_boost_duration_s=2400.0,
                      baseline_reliability=0.9)   # B2b-4: a baseline carries a reliability
        m.update(ep, ctx)
        assert m._state.general.effective_factor.effective_n > before
        assert m._state.recent_samples[-1].outcome_class in (
            C.BOOST_SUCCESS.value, C.BOOST_NO_EFFECT.value, C.BOOST_UNNECESSARY.value,
            C.BOOST_OVERSHOOT.value)

    def test_confounded_does_not_move_factor(self):
        m = BoostModel("lz")
        # prime with a clean sample so there is a learned value to protect
        ep0 = boost_episode("e0", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d0")
        m.update(ep0, replace(boost_context(ep0, requested_offset_c=1.5),
                              authority="live_record", dispatch_status="fully_succeeded"))
        learned = m._state.general.effective_factor.value
        n_before = m._state.general.effective_factor.effective_n
        ep = boost_episode("e1", [18.0, 20.0, 23.0, 24.0], decision_id="d1",
                           confounder_flags=("solar_gain",))
        # solar_gain is rejected at eligibility -> counted as confounded, no model move
        m.update(ep, replace(boost_context(ep, requested_offset_c=3.0),
                            authority="live_record", dispatch_status="fully_succeeded"))
        assert m._state.general.effective_factor.value == learned
        assert m._state.general.effective_factor.effective_n == n_before
        assert m._state.outcome_class_counts.get(C.BOOST_CONFOUNDED.value, 0) >= 1

    def test_legacy_fallback_does_not_increase_offset(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1")
        n_before = m._state.general.effective_factor.effective_n
        m.update(ep, replace(boost_context(ep, requested_offset_c=2.0),
                            authority="legacy_fallback", dispatch_status=None,
                            outcome_reliability="partial"))
        # diagnostic only: stored as sample/count, learned dims untouched
        assert m._state.general.effective_factor.effective_n == n_before

    def test_partial_does_not_increase_factor(self):
        m = BoostModel("lz")
        ep0 = boost_episode("e0", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d0")
        m.update(ep0, replace(boost_context(ep0, requested_offset_c=1.0),
                             authority="live_record", dispatch_status="fully_succeeded"))
        learned = m._state.general.effective_factor.value
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1")
        m.update(ep, replace(boost_context(ep, requested_offset_c=3.0),
                            authority="live_record", dispatch_status="partially_succeeded",
                            outcome_reliability="partial"))
        assert m._state.general.effective_factor.value <= learned + 1e-9


class TestStorage:
    def test_sample_roundtrip_with_class(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1")
        m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        state = m.serialize_state()
        m2 = BoostModel("lz")
        m2.deserialize_state(state)
        assert m2._state.recent_samples[-1].outcome_class
        assert m2._state.outcome_class_counts == m._state.outcome_class_counts

    def test_old_sample_without_class_gets_neutral_default(self):
        ep = boost_episode("e1", [18.0, 20.0, 21.0], decision_id="d1")
        from custom_components.thermosmart.learning.models.boost import BoostSample
        s = BoostSample(source_episode_id="e1", decision_id="d1", learning_zone_id="lz",
                        requested_offset_c=1.5, effective_factor_c=1.5, temperature_gain_c=3.0,
                        overshoot_c=0.0, comfort_score=0.9, controller_score=0.8,
                        boost_duration_s=1800.0, attribution_reliability=0.8,
                        effective_weight=0.7, reason="reached", partial=False,
                        evaluation_version=1)
        d = _sample_dict(s)
        del d["outcome_class"]   # simulate an old persisted sample
        restored = _sample_from(d)
        assert restored.outcome_class == C.INSUFFICIENT_COMPARISON.value

    def test_diagnostics_exposes_class_counts(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1")
        m.update(ep, replace(boost_context(ep, requested_offset_c=1.5),
                            authority="live_record", dispatch_status="fully_succeeded"))
        diag = m.diagnostics()
        assert isinstance(diag.outcome_class_counts, dict) and diag.outcome_class_counts


def _adaptive_snapshot(m):
    g = m._state.general.effective_factor
    return (round(m.confidence().value, 9), round(m.confidence().reliability, 9),
            round(g.value, 9), round(g.effective_n, 9), g.sample_count,
            m._state.full_count, m._state.partial_count)


class TestLegacyFallbackIsolation:
    """Correction #2: legacy_fallback adds ZERO adaptive maturity."""

    def test_legacy_does_not_change_any_adaptive_value(self):
        base = BoostModel("lz")
        for i in range(6):
            ep = boost_episode(f"a{i}", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id=f"a{i}")
            base.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        ref = _adaptive_snapshot(base)
        # same authoritative samples + 100 legacy-fallback samples
        flooded = BoostModel("lz")
        for i in range(6):
            ep = boost_episode(f"a{i}", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id=f"a{i}")
            flooded.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        for i in range(100):
            ep = boost_episode(f"L{i}", [18.0, 21.0, 23.0, 24.0], decision_id=f"L{i}")
            flooded.update(ep, replace(boost_context(ep, requested_offset_c=4.0),
                                       authority="legacy_fallback", dispatch_status=None))
        assert _adaptive_snapshot(flooded) == ref

    def test_legacy_only_raises_diagnostic_counter(self):
        m = BoostModel("lz")
        ep = boost_episode("L1", [18.0, 20.0, 21.0], decision_id="L1")
        m.update(ep, replace(boost_context(ep, requested_offset_c=2.0),
                            authority="legacy_fallback", dispatch_status=None))
        assert sum(m._state.outcome_class_counts.values()) >= 1
        assert m._state.general.effective_factor.effective_n == 0.0
        assert not [s for s in m._state.recent_samples]  # no adaptive sample


class TestPartialNoFactorMovement:
    """Correction #3: partial dispatch never moves the factor / adaptive confidence."""

    def test_existing_factor_unchanged_by_partials(self):
        m = BoostModel("lz")
        for i in range(6):
            ep = boost_episode(f"a{i}", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id=f"a{i}")
            m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        ref = _adaptive_snapshot(m)
        for i in range(50):
            ep = boost_episode(f"p{i}", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id=f"p{i}")
            m.update(ep, replace(boost_context_with_comparison(ep, requested_offset_c=4.0),
                                 dispatch_status="partially_succeeded",
                                 outcome_reliability="partial"))
        assert _adaptive_snapshot(m) == ref

    def test_partial_factor_update_false(self):
        assert class_effect(C.BOOST_PARTIAL_DISPATCH).factor_update is False
        assert class_effect(C.BOOST_PARTIAL_DISPATCH).confidence_increase is False
        assert class_effect(C.BOOST_PARTIAL_DISPATCH).rollback_evidence is False
