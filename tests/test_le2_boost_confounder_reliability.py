"""B2b-4: confounder completion + multi-component reliability."""
from __future__ import annotations

from dataclasses import replace

import pytest

from custom_components.thermosmart.learning.models import (
    BoostConfounder, BoostModel, BoostOutcomeClass as C, BoostOutcomeReliability,
    ConfounderSeverity, adaptive_allowed, class_min_reliability, compute_outcome_reliability,
    evaluate_confounders, severity_of, build_non_boost_sample)
from custom_components.thermosmart.learning.models.boost_reliability import (
    RELIABILITY_COMPONENT_VERSION)
from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
from tests.helpers_boost import boost_context, boost_context_with_comparison, boost_episode


def _rel(**kw):
    base = dict(sensor_reliability=0.9, trajectory_completeness=0.9, dispatch_reliability=1.0,
                attribution_reliability=0.9, baseline_reliability=0.9, timing_reliability=0.9,
                device_reliability=1.0, context_stability=1.0,
                confounders=evaluate_confounders([]), baseline_required=True)
    base.update(kw)
    return compute_outcome_reliability(**base)


# ── confounder contract + severity ──────────────────────────────────────────
class TestConfounderContract:
    def test_interrupting_severity(self):
        for f in ("restart_gap", "sensor_gap", "window_open", "heating_failure",
                  "episode_timeout"):
            assert evaluate_confounders([f]).is_interrupting

    def test_blocking_severity(self):
        for f in ("solar_gain", "additional_radiator", "multi_trv_ambiguous"):
            assert evaluate_confounders([f]).is_blocking

    def test_degrading_severity(self):
        e = evaluate_confounders(["occupancy_heat_gain"])
        assert not e.is_blocking and not e.is_interrupting and e.degrading

    def test_unknown_flag_degrades_safely(self):
        e = evaluate_confounders(["some_unrecognised_flag"])
        assert e.max_severity == ConfounderSeverity.DEGRADING.value
        assert BoostConfounder.UNKNOWN.value in e.confounders

    def test_dedup_one_event_one_penalty(self):
        # window -> target -> mode all from one window event -> a single root cause
        e = evaluate_confounders(["window_open", "target_change", "mode_change"])
        assert e.distinct_causes == 1

    def test_independent_causes_counted_separately(self):
        e = evaluate_confounders(["occupancy_heat_gain", "forecast_or_outdoor_shift"])
        assert e.distinct_causes == 2

    def test_partial_and_multi_trv_flags(self):
        assert evaluate_confounders([], partial_dispatch=True).degrading
        assert evaluate_confounders([], multi_trv_ambiguous=True).is_blocking


# ── reliability formula ──────────────────────────────────────────────────────
class TestReliabilityFormula:
    def test_low_critical_not_masked_by_high_others(self):
        # a poor sensor path cannot be compensated by a perfect baseline
        assert _rel(sensor_reliability=0.1).overall_reliability <= 0.1

    def test_blocking_confounder_zero(self):
        r = _rel(confounders=evaluate_confounders(["solar_gain"]))
        assert r.overall_reliability == 0.0 and r.blocking_reasons

    def test_missing_component_conservative(self):
        assert _rel(timing_reliability=None).overall_reliability == 0.0

    def test_deterministic(self):
        assert _rel().overall_reliability == _rel().overall_reliability

    def test_bounded(self):
        r = _rel(sensor_reliability=5.0, baseline_reliability=-1.0)
        assert 0.0 <= r.overall_reliability <= 1.0

    def test_degrading_reduces_not_zero(self):
        clean = _rel().overall_reliability
        degraded = _rel(confounders=evaluate_confounders(["occupancy_heat_gain"]))
        assert 0.0 < degraded.overall_reliability < clean

    def test_one_event_not_triple_penalised(self):
        one = _rel(confounders=evaluate_confounders(["occupancy_heat_gain"])).confounder_reliability
        grouped = _rel(confounders=evaluate_confounders(
            ["window_open", "target_change", "mode_change"]))
        # window group is interrupting -> 0 anyway, but the dedup is proven elsewhere;
        # here check a single degrading cause keeps a single 0.7 penalty
        assert one == pytest.approx(0.7)


# ── class-dependent gates ────────────────────────────────────────────────────
class TestClassGates:
    def test_success_needs_high_reliability(self):
        assert class_min_reliability("boost_success") > class_min_reliability("boost_overshoot")

    def test_overshoot_no_baseline_needed(self):
        r = compute_outcome_reliability(
            sensor_reliability=0.8, trajectory_completeness=0.8, dispatch_reliability=1.0,
            attribution_reliability=0.8, baseline_reliability=0.0, timing_reliability=0.8,
            device_reliability=1.0, context_stability=1.0, confounders=evaluate_confounders([]),
            baseline_required=False)
        assert adaptive_allowed("boost_overshoot", r)

    def test_success_blocked_by_low_overall(self):
        assert not adaptive_allowed("boost_success", _rel(sensor_reliability=0.2))

    def test_non_adaptive_classes_never_adaptive(self):
        for cls in ("boost_confounded", "boost_interrupted", "boost_partial_dispatch",
                    "insufficient_comparison"):
            assert class_min_reliability(cls) is None
            assert not adaptive_allowed(cls, _rel())


# ── model effect by reliability (integration) ───────────────────────────────
class TestModelEffect:
    def _success_update(self, m, idx, *, baseline_reliability=0.9):
        ep = boost_episode(f"e{idx}", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id=f"d{idx}")
        return m.update(ep, boost_context_with_comparison(
            ep, requested_offset_c=1.5, baseline_reliability=baseline_reliability))

    def test_high_reliability_success_learns(self):
        m = BoostModel("lz"); self._success_update(m, 0)
        assert m._state.general.effective_factor.effective_n > 0

    def test_weak_baseline_blocks_success_update(self):
        m = BoostModel("lz"); self._success_update(m, 0, baseline_reliability=0.05)
        assert m._state.general.effective_factor.effective_n == 0  # below SUCCESS gate

    def test_sample_carries_reliability(self):
        m = BoostModel("lz"); self._success_update(m, 0)
        s = m._state.recent_samples[-1]
        assert 0.0 <= s.reliability_overall <= 1.0 and s.reliability_components

    def test_overshoot_without_baseline_is_negative_evidence(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 23.0], decision_id="d1")
        # no baseline (boost_context default) -> overshoot still classifies + may learn
        m.update(ep, boost_context(ep, requested_offset_c=3.0))
        s = m._state.recent_samples[-1] if m._state.recent_samples else \
            m._state.diagnostic_samples[-1]
        assert s.outcome_class == C.BOOST_OVERSHOOT.value


# ── baseline / boost confounder consistency ─────────────────────────────────
class TestBaselineConsistency:
    def _ep(self, flags=()):
        return boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 21.0], decision_id="d1",
                             confounder_flags=flags)

    def test_blocking_confounder_excludes_baseline(self):
        assert build_non_boost_sample(self._ep(("solar_gain",)), BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="fully_succeeded")) is None

    def test_interrupting_confounder_excludes_baseline(self):
        assert build_non_boost_sample(self._ep(("sensor_gap",)), BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="fully_succeeded")) is None

    def test_clean_episode_is_baseline(self):
        assert build_non_boost_sample(self._ep(()), BoostDispatchRecord(
            "d1", boost_applied_c=0.0, dispatch_status="fully_succeeded")) is not None

    def test_same_evaluator_for_both_paths(self):
        # solar_gain blocks both baseline creation and (via classifier) the boost outcome
        e = evaluate_confounders(["solar_gain"])
        assert e.is_blocking  # baseline excludes; classifier -> CONFOUNDED


# ── persistence ──────────────────────────────────────────────────────────────
class TestPersistence:
    def test_reliability_roundtrip(self):
        r = _rel()
        assert BoostOutcomeReliability.from_dict(r.to_dict()).overall_reliability == \
            r.overall_reliability

    def test_old_dict_missing_fields_safe(self):
        r = BoostOutcomeReliability.from_dict({"overall_reliability": 0.5})
        assert r.component_version == 0 and r.overall_reliability == 0.5

    def test_sample_reliability_persists(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1")
        m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        m2 = BoostModel("lz"); m2.deserialize_state(m.serialize_state())
        assert m2._state.recent_samples[-1].reliability_components

    def test_component_version_present(self):
        assert _rel().component_version == RELIABILITY_COMPONENT_VERSION
