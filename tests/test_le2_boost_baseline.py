"""B2b-3: trustworthy baseline-versus-boost comparison."""
from __future__ import annotations

from dataclasses import replace

import pytest

from custom_components.thermosmart.learning.models import (
    BaselineComparisonStore, BaselineParams, BaselineSource, BoostModel, BoostOutcomeClass,
    build_non_boost_sample, estimate_non_boost_duration, similarity)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
from tests.helpers_baseline import query, sample
from tests.helpers_boost import boost_context, boost_episode

NOW = "2025-01-10T00:00:00+00:00"


# ── sample creation gating (§4) ─────────────────────────────────────────────
class TestSampleCreation:
    def _ep(self, **kw):
        return boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1", **kw)

    def test_authoritative_non_boost_stored(self):
        s = build_non_boost_sample(self._ep(), BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="fully_succeeded"))
        assert s is not None and s.boost_applied_c == 0.0 and s.authoritative

    def test_boost_episode_not_a_baseline(self):
        assert build_non_boost_sample(self._ep(), BoostDispatchRecord(
            "d1", boost_applied_c=1.5, dispatch_status="fully_succeeded")) is None

    def test_not_attempted_excluded(self):
        assert build_non_boost_sample(self._ep(), BoostDispatchRecord(
            "d1", dispatch_status="not_attempted")) is None

    def test_partial_excluded(self):
        assert build_non_boost_sample(self._ep(), BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="partially_succeeded")) is None

    def test_failed_excluded(self):
        assert build_non_boost_sample(self._ep(), BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="failed")) is None

    def test_confounded_excluded(self):
        from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
        ep = boost_episode("e1", [18.0, 19.5, 21.0], decision_id="d1",
                           confounder_flags=("solar_gain",))
        assert build_non_boost_sample(ep, BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="fully_succeeded")) is None

    def test_not_reached_excluded(self):
        from custom_components.thermosmart.learning.episode_schemas import EpisodeReason
        ep = boost_episode("e1", [18.0, 19.0, 19.5], decision_id="d1",
                           reason=EpisodeReason.TIMEOUT)
        assert build_non_boost_sample(ep, BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="fully_succeeded")) is None


# ── store: zone isolation / dedup / retention ───────────────────────────────
class TestStore:
    def test_cross_zone_rejected(self):
        st = BaselineComparisonStore("z")
        assert st.add(sample("d1", 1800, zone="other")) is False and st.size == 0

    def test_dedup_by_decision(self):
        st = BaselineComparisonStore("z")
        assert st.add(sample("d1", 1800))
        assert st.add(sample("d1", 1900)) is False and st.size == 1

    def test_count_cap(self):
        st = BaselineComparisonStore("z", BaselineParams(count_cap=10))
        for i in range(25):
            st.add(sample(f"d{i}", 1800, reached=f"2025-01-01T{i % 24:02d}:00:00+00:00"))
        assert st.size <= 10

    def test_age_prune(self):
        st = BaselineComparisonStore("z", BaselineParams(max_age_s=3600))
        st.add(sample("old", 1800, reached="2025-01-01T00:00:00+00:00"))
        st.prune(now_ts="2025-02-01T00:00:00+00:00")
        assert st.size == 0

    def test_serialize_restore(self):
        st = BaselineComparisonStore("z")
        for i in range(4):
            st.add(sample(f"d{i}", 1800 + i * 20, reached=f"2025-01-0{i+1}T00:00:00+00:00"))
        st2 = BaselineComparisonStore("z")
        st2.restore(st.serialize())
        assert st2.size == st.size

    def test_corrupt_sample_isolated(self):
        st = BaselineComparisonStore("z")
        data = {"learning_zone_id": "z", "schema_version": 1,
                "samples": [sample("ok", 1800).to_dict(), {"bad": "row"}]}
        st.restore(data)
        assert st.size == 1

    def test_old_schema_discarded(self):
        st = BaselineComparisonStore("z")
        d = sample("d1", 1800).to_dict()
        d["schema_version"] = 0
        st.restore({"learning_zone_id": "z", "samples": [d]})
        assert st.size == 0


# ── matching / similarity (§5) ──────────────────────────────────────────────
class TestMatching:
    def test_similar_high(self):
        assert similarity(sample("d", 1800), query(), BaselineParams()) > 0.8

    def test_wrong_zone_zero(self):
        assert similarity(sample("d", 1800, zone="other"), query(), BaselineParams()) == 0.0

    def test_large_deficit_excluded(self):
        assert similarity(sample("d", 1800, deficit=8.0), query(deficit=3.0),
                          BaselineParams()) == 0.0

    def test_large_outdoor_excluded(self):
        assert similarity(sample("d", 1800, outdoor=-20.0), query(outdoor=10.0),
                          BaselineParams()) == 0.0

    def test_different_device_type_excluded(self):
        assert similarity(sample("d", 1800, device="direct_valve"), query(device="setpoint"),
                          BaselineParams()) == 0.0


# ── causality (§9) ──────────────────────────────────────────────────────────
class TestCausality:
    def test_future_samples_excluded(self):
        samples = [sample(f"d{i}", 1800, reached=f"2025-03-0{i+1}T00:00:00+00:00")
                   for i in range(3)]
        est = estimate_non_boost_duration(query(before="2025-01-01T00:00:00+00:00"),
                                          samples, now_ts="2025-03-10T00:00:00+00:00")
        assert est.expected_non_boost_duration_s is None and est.rejection_reason == "no_candidates"

    def test_own_episode_excluded(self):
        samples = [sample("dQ", 1800, reached="2025-01-01T00:00:00+00:00")] + \
                  [sample(f"d{i}", 1800, reached=f"2025-01-0{i+2}T00:00:00+00:00")
                   for i in range(2)]
        est = estimate_non_boost_duration(query(decision_id="dQ"), samples, now_ts=NOW)
        # own decision excluded -> only 2 remain -> too few
        assert est.rejection_reason == "too_few_samples"


# ── aggregation + gates (§7/§8) ─────────────────────────────────────────────
class TestAggregation:
    def _good(self, n=5, dur=1800):
        return [sample(f"d{i}", dur, reached=f"2025-01-{i+1:02d}T00:00:00+00:00")
                for i in range(n)]

    def test_robust_estimate(self):
        est = estimate_non_boost_duration(query(), self._good(), now_ts=NOW)
        assert est.expected_non_boost_duration_s == pytest.approx(1800, abs=60)
        assert est.source == BaselineSource.HISTORICAL_ONLY.value

    def test_outlier_bounded(self):
        s = self._good(5, 1800) + [sample("out", 100000, reached="2025-01-09T00:00:00+00:00")]
        est = estimate_non_boost_duration(query(), s, now_ts=NOW)
        # robust median is not dragged to the outlier
        assert est.expected_non_boost_duration_s < 3000

    def test_too_few_samples_none(self):
        est = estimate_non_boost_duration(query(), self._good(2), now_ts=NOW)
        assert est.expected_non_boost_duration_s is None and est.rejection_reason == "too_few_samples"

    def test_single_episode_not_high_reliability(self):
        est = estimate_non_boost_duration(query(), self._good(1), now_ts=NOW)
        assert est.expected_non_boost_duration_s is None

    def test_low_similarity_excluded(self):
        s = [sample(f"d{i}", 1800, target=25.0, reached=f"2025-01-0{i+1}T00:00:00+00:00")
             for i in range(5)]
        est = estimate_non_boost_duration(query(target=21.0), s, now_ts=NOW)
        assert est.expected_non_boost_duration_s is None

    def test_high_dispersion_rejected(self):
        s = [sample(f"d{i}", 600 + i * 1500, reached=f"2025-01-0{i+1}T00:00:00+00:00")
             for i in range(5)]
        est = estimate_non_boost_duration(query(), s, now_ts=NOW,
                                          params=BaselineParams(max_dispersion_ratio=0.3))
        assert est.expected_non_boost_duration_s is None


# ── outcome integration (§11) ───────────────────────────────────────────────
class TestOutcomeIntegration:
    def _update(self, observed_vals, *, expected_s, offset=1.5):
        m = BoostModel("lz")
        ep = boost_episode("e1", observed_vals, decision_id="d1")
        ctx = replace(boost_context(ep, requested_offset_c=offset,
                                    expected_non_boost_duration_s=expected_s),
                      authority="live_record", dispatch_status="fully_succeeded")
        m.update(ep, ctx)
        return m._state.recent_samples[-1].outcome_class if m._state.recent_samples \
            else m._state.diagnostic_samples[-1].outcome_class

    def test_reliable_benefit_success(self):
        # boost reaches target much faster than the baseline expectation -> SUCCESS
        cls = self._update([18.0, 20.0, 21.0], expected_s=3600.0)
        assert cls == BoostOutcomeClass.BOOST_SUCCESS.value

    def test_no_baseline_insufficient(self):
        cls = self._update([18.0, 20.0, 21.0], expected_s=None)
        assert cls == BoostOutcomeClass.INSUFFICIENT_COMPARISON.value

    def test_overshoot_beats_benefit(self):
        # dense, plausible rise that reaches target then overshoots (passes data-quality gate)
        cls = self._update([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 23.0], expected_s=3600.0)
        assert cls == BoostOutcomeClass.BOOST_OVERSHOOT.value
