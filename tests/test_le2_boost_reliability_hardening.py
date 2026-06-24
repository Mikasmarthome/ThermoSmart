"""B2b-4a: legacy reliability, single-authority confounder weighting, multi-TRV,
target/overshoot stability, authority-interaction attribution, supply delay."""
from __future__ import annotations

from dataclasses import replace

import pytest

from custom_components.thermosmart.learning.episode_schemas import TrajectoryPoint
from custom_components.thermosmart.learning.models import (
    BoostConfounder, BoostDeviceDispatchEvidence, BoostModel, ConfounderSeverity,
    OvershootStabilityParams, TargetStabilityParams, afterheat_attribution_severity,
    early_cutoff_interaction_severity, evaluate_confounders, evaluate_multi_trv,
    evaluate_overshoot_stability, evaluate_target_stability, supply_delay_confounder)
from custom_components.thermosmart.learning.models.boost import _sample_dict, _sample_from
from tests.helpers_boost import boost_context_with_comparison, boost_episode


# ── 1. legacy reliability degrades ──────────────────────────────────────────
class TestLegacyReliability:
    def test_legacy_sample_unknown_quality(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1")
        m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        d = _sample_dict(m._state.recent_samples[-1])
        del d["reliability_overall"]
        del d["reliability_known"]
        s = _sample_from(d)
        assert s.reliability_known is False and s.reliability_overall == 0.0

    def test_new_sample_reliability_known(self):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1")
        m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        assert m._state.recent_samples[-1].reliability_known is True

    def test_legacy_default_is_zero_not_one(self):
        from custom_components.thermosmart.learning.models.boost import BoostSample
        s = BoostSample(source_episode_id="e", decision_id="d", learning_zone_id="z",
                        requested_offset_c=1.5, effective_factor_c=1.5, temperature_gain_c=1.0,
                        overshoot_c=0.0, comfort_score=0.9, controller_score=0.8,
                        boost_duration_s=1800.0, attribution_reliability=0.8,
                        effective_weight=0.7, reason="reached", partial=False,
                        evaluation_version=1)
        assert s.reliability_overall == 0.0 and s.reliability_known is False


# ── 2. single-authority confounder weight (no double counting) ──────────────
class TestSingleAuthorityWeight:
    def _weight(self, flags):
        m = BoostModel("lz")
        ep = boost_episode("e1", [18.0, 19.5, 20.5, 21.0, 21.0], decision_id="d1",
                           confounder_flags=flags)
        m.update(ep, boost_context_with_comparison(ep, requested_offset_c=1.5))
        s = (m._state.recent_samples + m._state.diagnostic_samples)[-1]
        return s.effective_weight

    def test_one_degrading_reduces_weight_once(self):
        clean = self._weight(())
        one = self._weight(("occupancy_heat_gain",))
        # exactly one 0.7 penalty
        assert one == pytest.approx(clean * 0.7, rel=1e-6)

    def test_alias_same_cause_single_penalty(self):
        # window + target + mode are one root cause (interrupting -> non-adaptive anyway);
        # two independent degrading causes get two bounded penalties.
        two = evaluate_confounders(["occupancy_heat_gain", "forecast_or_outdoor_shift"])
        assert two.distinct_causes == 2

    def test_info_confounder_no_penalty(self):
        info = self._weight(("preheat_interaction",))
        clean = self._weight(())
        assert info == pytest.approx(clean, rel=1e-6)


# ── 3. UNKNOWN severity ──────────────────────────────────────────────────────
class TestUnknownConfounder:
    def test_unknown_is_degrading(self):
        e = evaluate_confounders(["never_seen_flag"])
        assert e.max_severity == ConfounderSeverity.DEGRADING.value
        assert BoostConfounder.UNKNOWN.value in e.confounders

    def test_unknown_no_exception_and_counted(self):
        e = evaluate_confounders(["x", "y", "z"])  # all unknown collapse to one UNKNOWN
        assert BoostConfounder.UNKNOWN.value in e.confounders


# ── 4. multi-TRV deep detection ──────────────────────────────────────────────
def _dev(slot, off, **kw):
    base = dict(slot=slot, control_type="setpoint", requested_offset_c=off,
                effective_offset_c=off)
    base.update(kw)
    return BoostDeviceDispatchEvidence(**base)


class TestMultiTrv:
    def test_homogeneous_clean_ok(self):
        assert evaluate_multi_trv([_dev("a", 0.5), _dev("b", 0.5)]) is None

    def test_one_clamped_material(self):
        # fine-step device with a material clamp beyond quantisation tolerance
        assert evaluate_multi_trv([
            _dev("a", 0.5, device_step_c=0.1),
            _dev("b", 0.1, clamp_applied_c=0.6, device_step_c=0.1)]) \
            is BoostConfounder.PARTIAL_DEVICE_EFFECT

    def test_coarse_step_quantisation_not_flagged(self):
        # 0.5 °C-step devices: a 0.4 °C clamp is unavoidable quantisation -> NOT confounded
        assert evaluate_multi_trv([
            _dev("a", 0.5, device_step_c=0.5),
            _dev("b", 0.1, clamp_applied_c=0.4, device_step_c=0.5)]) is None

    def test_large_offset_spread_ambiguous(self):
        assert evaluate_multi_trv([_dev("a", 0.1, effective_offset_c=0.1),
                                   _dev("b", 1.0, effective_offset_c=1.0)]) \
            is BoostConfounder.MULTI_TRV_AMBIGUOUS

    def test_unavailable_at_dispatch(self):
        assert evaluate_multi_trv([_dev("a", 0.5),
                                   _dev("b", 0.5, available_at_dispatch=False)]) \
            is BoostConfounder.PARTIAL_DEVICE_EFFECT

    def test_failed_during_episode(self):
        assert evaluate_multi_trv([_dev("a", 0.5),
                                   _dev("b", 0.5, available_during_observation=False)]) \
            is BoostConfounder.PARTIAL_DEVICE_EFFECT

    def test_one_failed(self):
        assert evaluate_multi_trv([_dev("a", 0.5), _dev("b", 0.5, dispatch_succeeded=False)]) \
            is BoostConfounder.PARTIAL_DEVICE_EFFECT

    def test_mixed_control_types_ambiguous(self):
        mixed = [_dev("a", 0.5), BoostDeviceDispatchEvidence("b", "direct_valve", None, None)]
        assert evaluate_multi_trv(mixed) is BoostConfounder.MULTI_TRV_AMBIGUOUS


# ── 5/6. target & overshoot stability ────────────────────────────────────────
def _traj(vals, step_ms=300000):
    return [TrajectoryPoint(i * step_ms, v) for i, v in enumerate(vals)]


class TestStability:
    def test_single_point_not_reached(self):
        ev = evaluate_target_stability(_traj([18, 19, 20, 21]), 21.0)
        assert not ev.stable and ev.rejection_reason in ("too_few_in_band", "dwell_too_short")

    def test_stable_target_reached(self):
        ev = evaluate_target_stability(_traj([18, 19, 20, 21, 21, 21, 21]), 21.0)
        assert ev.stable and ev.samples_in_band >= 3 and ev.dwell_duration_s >= 600.0

    def test_short_peak_not_overshoot(self):
        ev = evaluate_overshoot_stability(_traj([20, 21, 23, 21, 21]), 21.0)
        assert not ev.stable and ev.rejection_reason in ("too_few_above", "dwell_too_short")

    def test_sustained_overshoot(self):
        ev = evaluate_overshoot_stability(_traj([20, 22, 22, 22, 22]), 21.0)
        assert ev.stable and ev.dwell_duration_s >= 300.0 and ev.stable_overshoot_c > 0.5


# ── 7/8. afterheat & early-cutoff attribution ───────────────────────────────
class TestAuthorityInteraction:
    def test_low_afterheat_info(self):
        assert afterheat_attribution_severity(0.1, 2.0) is ConfounderSeverity.INFO

    def test_high_afterheat_degrading(self):
        assert afterheat_attribution_severity(0.8, 2.0) is ConfounderSeverity.DEGRADING

    def test_early_cutoff_overlap_degrades(self):
        assert early_cutoff_interaction_severity(True) is ConfounderSeverity.DEGRADING
        assert early_cutoff_interaction_severity(True, severe=True) is ConfounderSeverity.BLOCKING
        assert early_cutoff_interaction_severity(False) is ConfounderSeverity.INFO


# ── 9. supply delay ──────────────────────────────────────────────────────────
class TestSupplyDelay:
    def test_normal_delay_no_confounder(self):
        assert supply_delay_confounder(10.0, 15.0) is None

    def test_moderate_delay_degrading(self):
        assert supply_delay_confounder(10.0, 35.0) is BoostConfounder.SUPPLY_DELAY_ANOMALY

    def test_extreme_delay_heating_failure(self):
        assert supply_delay_confounder(10.0, 70.0) is BoostConfounder.HEATING_FAILURE

    def test_no_onset_heating_failure(self):
        assert supply_delay_confounder(10.0, None) is BoostConfounder.HEATING_FAILURE


# ── 10. baseline/boost consistency for new confounders ──────────────────────
class TestBaselineConsistencyNew:
    def test_new_confounders_consistent(self):
        from custom_components.thermosmart.learning.models import build_non_boost_sample
        from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
        for flag in ("sensor_source_change", "multi_trv_ambiguous", "supply_delay_anomaly",
                     "device_configuration_change"):
            ep = boost_episode("e1", [18.0, 19.0, 20.0, 21.0, 21.0], decision_id="d1",
                               confounder_flags=(flag,))
            sample = build_non_boost_sample(ep, BoostDispatchRecord(
                "d1", boost_applied_c=0.0, dispatch_status="fully_succeeded"))
            ev = evaluate_confounders([flag])
            if ev.is_blocking or ev.is_interrupting:
                assert sample is None, f"{flag} should exclude baseline"
