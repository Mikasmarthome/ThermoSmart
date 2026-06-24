"""B2b-3a: baseline causality + timing hardening regression tests."""
from __future__ import annotations

from dataclasses import replace

import pytest

from custom_components.thermosmart.learning.models import (
    BaselineComparisonStore, BaselineParams, BaselineTimingBasis, BoostModel,
    NonBoostBaselineSample, build_non_boost_sample, estimate_non_boost_duration)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
from tests.helpers_baseline import query, sample
from tests.helpers_boost import boost_context, boost_episode


def _five(reached_prefix="2025-01-0", onset=1800):
    return [sample(f"d{i}", onset, reached=f"{reached_prefix}{i+1}T00:00:00+00:00")
            for i in range(5)]


# ── §1 cutoff at decision time ───────────────────────────────────────────────
class TestCutoff:
    def test_sample_before_decision_accepted(self):
        est = estimate_non_boost_duration(query(before="2025-01-10T00:00:00+00:00"), _five())
        assert est.expected_non_boost_duration_s is not None

    def test_sample_completed_during_boost_episode_excluded(self):
        # boost decision at 2025-01-06; a non-boost sample completing 2025-01-07 (after the
        # decision, before outcome resolution) must NOT be usable.
        before = [sample(f"d{i}", 1800, reached=f"2025-01-0{i+1}T00:00:00+00:00")
                  for i in range(2)]
        during = sample("late", 1800, reached="2025-01-07T00:00:00+00:00")
        est = estimate_non_boost_duration(
            query(before="2025-01-06T00:00:00+00:00"), before + [during])
        # only the 2 earlier remain -> too few (the 'during' sample is excluded)
        assert est.rejection_reason == "too_few_samples" and est.sample_count == 2

    def test_sample_after_outcome_excluded(self):
        future = [sample(f"f{i}", 1800, reached=f"2025-02-0{i+1}T00:00:00+00:00")
                  for i in range(3)]
        est = estimate_non_boost_duration(query(before="2025-01-06T00:00:00+00:00"), future)
        assert est.expected_non_boost_duration_s is None

    def test_retry_uses_same_cutoff_and_data(self):
        q = query(before="2025-01-10T00:00:00+00:00")
        samples = _five()
        a = estimate_non_boost_duration(q, samples)
        b = estimate_non_boost_duration(q, samples)        # later "retry"
        assert (a.expected_non_boost_duration_s == b.expected_non_boost_duration_s
                and a.sample_refs == b.sample_refs and a.baseline_cutoff_ts == b.baseline_cutoff_ts)


# ── §2 provenance ────────────────────────────────────────────────────────────
class TestProvenance:
    def test_estimate_carries_provenance(self):
        est = estimate_non_boost_duration(query(before="2025-01-10T00:00:00+00:00"), _five())
        assert est.baseline_cutoff_ts == "2025-01-10T00:00:00+00:00"
        assert est.decision_id == "dQ" and est.query_source == "live_decision"
        assert len(est.sample_refs) >= 3 and est.query_context_version >= 1


# ── §3 frozen query features ─────────────────────────────────────────────────
class TestFrozenFeatures:
    def test_matching_uses_start_context_not_end(self):
        # samples match start context A (target 21); a query whose features were frozen at A
        # matches, even though the episode "ended" under a different context B.
        samples = _five()
        q_start = query(before="2025-01-10T00:00:00+00:00", target=21.0, deficit=3.0)
        est = estimate_non_boost_duration(q_start, samples)
        assert est.expected_non_boost_duration_s is not None
        # a query reconstructed from a shifted end context (target 25) would not match
        q_end = query(before="2025-01-10T00:00:00+00:00", target=25.0)
        assert estimate_non_boost_duration(q_end, samples).expected_non_boost_duration_s is None


# ── §4/§5 effective onset vs command duration ───────────────────────────────
class TestTimingBasis:
    def test_thermal_uses_effective_onset(self):
        est = estimate_non_boost_duration(query(before="2025-01-10T00:00:00+00:00"), _five())
        assert est.timing_basis == BaselineTimingBasis.EFFECTIVE_ONSET.value
        assert est.thermal_baseline_available and est.effective_onset_available

    def test_missing_onset_no_silent_thermal_fallback(self):
        # samples with NO effective onset (command duration only) -> no thermal baseline
        s = [replace(sample(f"c{i}", 1800, reached=f"2025-01-0{i+1}T00:00:00+00:00"),
                     onset_to_target_duration_s=None) for i in range(5)]
        est = estimate_non_boost_duration(query(before="2025-01-10T00:00:00+00:00"), s)
        assert est.expected_non_boost_duration_s is None
        assert est.timing_basis == BaselineTimingBasis.UNAVAILABLE.value
        assert est.rejection_reason == "no_effective_onset"
        # command axis is still computed separately for diagnostics
        assert est.expected_non_boost_command_duration_s is not None

    def test_supply_delay_changes_command_not_thermal(self):
        # two sets identical in onset->target but very different command->target (supply delay)
        fast = [replace(sample(f"a{i}", 1800, reached=f"2025-01-0{i+1}T00:00:00+00:00"),
                        command_to_target_duration_s=1900.0) for i in range(5)]
        slow = [replace(sample(f"b{i}", 1800, reached=f"2025-01-0{i+1}T00:00:00+00:00"),
                        command_to_target_duration_s=5400.0) for i in range(5)]
        q = query(before="2025-01-10T00:00:00+00:00")
        ef, es = estimate_non_boost_duration(q, fast), estimate_non_boost_duration(q, slow)
        # thermal baseline identical; only the command axis differs
        assert ef.expected_non_boost_duration_s == es.expected_non_boost_duration_s
        assert ef.expected_non_boost_command_duration_s != es.expected_non_boost_command_duration_s


# ── §6 baseline reliability reduces adaptive weight ─────────────────────────
class TestReliabilityModelEffect:
    def _attr(self, baseline_rel):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 21.0], decision_id="d1")
        ctx = replace(boost_context(ep, requested_offset_c=1.5,
                                    expected_non_boost_duration_s=3600.0),
                      authority="live_record", dispatch_status="fully_succeeded",
                      baseline_reliability=baseline_rel)
        return m.evaluate_boost(ep, ctx).effect.attribution_reliability

    def test_robust_baseline_higher_weight_than_weak(self):
        assert self._attr(0.9) > self._attr(0.45)

    def test_baseline_reliability_applied_once(self):
        # without baseline_reliability the attribution is the unscaled value;
        # with rel=1.0 it must be (essentially) unchanged (no double counting).
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 21.0], decision_id="d1")
        base = replace(boost_context(ep, requested_offset_c=1.5,
                                     expected_non_boost_duration_s=3600.0),
                       authority="live_record", dispatch_status="fully_succeeded")
        none_rel = m.evaluate_boost(ep, base).effect.attribution_reliability
        full_rel = m.evaluate_boost(ep, replace(base, baseline_reliability=1.0)
                                    ).effect.attribution_reliability
        assert full_rel == pytest.approx(none_rel, abs=1e-9)

    def test_weak_baseline_strictly_reduces(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 21.0], decision_id="d1")
        base = replace(boost_context(ep, requested_offset_c=1.5,
                                     expected_non_boost_duration_s=3600.0),
                       authority="live_record", dispatch_status="fully_succeeded")
        none_rel = m.evaluate_boost(ep, base).effect.attribution_reliability
        weak = m.evaluate_boost(ep, replace(base, baseline_reliability=0.4)
                                ).effect.attribution_reliability
        assert weak < none_rel


# ── §7 zero-boost authority ──────────────────────────────────────────────────
class TestZeroBoostAuthority:
    def _ep(self, **kw):
        return boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 21.0], decision_id="d1", **kw)

    def test_applied_zero_with_full_dispatch_is_baseline(self):
        s = build_non_boost_sample(self._ep(), BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="fully_succeeded"))
        assert s is not None and s.boost_applied_c == 0.0

    def test_applied_zero_without_real_dispatch_not_baseline(self):
        # candidate gated to 0 but no real heating control executed (not fully_succeeded)
        assert build_non_boost_sample(self._ep(), BoostDispatchRecord(
            "d1", boost_candidate_c=2.0, boost_applied_c=0.0,
            dispatch_status="not_attempted")) is None

    def test_no_onset_no_baseline(self):
        # flat trajectory: no effective heating onset -> not a thermal baseline
        ep = boost_episode("e1", [21.0, 21.0, 21.0], decision_id="d1")
        assert build_non_boost_sample(ep, BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="fully_succeeded")) is None
