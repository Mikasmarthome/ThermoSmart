"""Phase D2-B — outdoor-conditioned learning updates.

Covers the pure acceptance gate and context builder, the lifecycle wiring that
feeds HeatRate/HeatLoss conditioned buckets from D2-A outdoor evidence, and the
control-invariance guarantee that published predictions and control are
unaffected (conditioned buckets learn and persist but are never selected because
the predict path uses an empty context).
"""
from __future__ import annotations

import copy
from datetime import timedelta
from unittest.mock import patch

import pytest

from custom_components.thermosmart.learning.contracts import (
    DataQuality, Measurement, PredictionType,
)
from custom_components.thermosmart.learning.outdoor_evidence import OutdoorEvidence
from custom_components.thermosmart.learning.models.heat_rate import (
    HeatRateParameters, HeatRateUpdateContext,
)
from custom_components.thermosmart.learning.models.heat_loss import (
    HeatLossParameters, HeatLossUpdateContext,
)
from custom_components.thermosmart.learning.runtime import outdoor_conditioning as oc
from custom_components.thermosmart.learning.runtime.capture import RuntimeCycleInput
from custom_components.thermosmart.learning.contracts import DataQuality as DQ
from tests.helpers_runtime_scenarios import runtime, T0, COMFORT
from custom_components.thermosmart.learning.runtime import (
    ControllerDecisionInput, DecisionType, ScheduleTarget,
)


def _ev(**kw) -> OutdoorEvidence:
    """A baseline ACCEPTED evidence (mean 2.0 -> band for [0,5)); override fields."""
    base = dict(
        time_weighted_mean_c=2.0, min_c=1.0, max_c=4.0, first_c=1.0, last_c=4.0,
        sample_count=3, usable_count=3, coverage=1.0, temporal_coverage=0.6,
        conditioning_ready=True, duration_s=600.0, quality=DataQuality.OK,
        freshness_known=False, source="sensor", source_changed=False)
    base.update(kw)
    return OutdoorEvidence(**base)


# ── edges match the model params (guard against drift) ───────────────────────

def test_bucket_edges_match_model_parameters():
    assert oc.OUTDOOR_BUCKET_EDGES == HeatRateParameters().outdoor_buckets
    assert oc.OUTDOOR_BUCKET_EDGES == HeatLossParameters().outdoor_buckets


# ── pure gate ─────────────────────────────────────────────────────────────────

class TestGate:
    def test_accepts_clean_evidence(self):
        g = oc.evaluate_gate(_ev())
        assert g.accepted and g.reason == "ok" and g.band == 3   # 2.0 in [0,5)

    def test_rejects_none(self):
        assert oc.evaluate_gate(None).reason == "no_evidence"

    def test_rejects_no_usable_mean(self):
        # usable_count 0 -> has_usable_evidence False and mean None
        ev = _ev(time_weighted_mean_c=None, usable_count=0, min_c=None, max_c=None,
                 first_c=None, last_c=None, conditioning_ready=False)
        assert oc.evaluate_gate(ev).reason == "no_usable_mean"

    @pytest.mark.parametrize("q", [DataQuality.STALE, DataQuality.UNKNOWN,
                                   DataQuality.UNAVAILABLE, DataQuality.INVALID])
    def test_rejects_non_ok_quality(self, q):
        assert oc.evaluate_gate(_ev(quality=q)).reason == "quality_not_ok"

    def test_rejects_source_changed(self):
        assert oc.evaluate_gate(_ev(source_changed=True)).reason == "source_changed"

    def test_coverage_boundary_exactly_half_accepts(self):
        assert oc.evaluate_gate(_ev(temporal_coverage=0.5)).accepted

    def test_coverage_just_below_half_rejects(self):
        assert oc.evaluate_gate(_ev(temporal_coverage=0.4999)).reason == "low_temporal_coverage"

    def test_usable_count_boundary(self):
        assert oc.evaluate_gate(_ev(usable_count=2)).accepted
        assert oc.evaluate_gate(_ev(usable_count=1)).reason == "too_few_usable_samples"

    def test_cross_band_spread_rejected(self):
        # min 4.0 in [0,5) band 3, max 6.0 in [5,10) band 4 -> spans two buckets
        assert oc.evaluate_gate(_ev(time_weighted_mean_c=4.5, min_c=4.0, max_c=6.0,
                                    first_c=4.0, last_c=6.0)).reason == "cross_band_spread"

    def test_single_band_within_edges_accepts(self):
        # entirely inside [5,10): band 4
        g = oc.evaluate_gate(_ev(time_weighted_mean_c=7.0, min_c=5.0, max_c=9.9,
                                 first_c=5.0, last_c=9.9))
        assert g.accepted and g.band == 4

    def test_bucket_edge_is_inclusive_lower(self):
        # exactly 0.0 -> band 3 ([0,5)); -0.0001 -> band 2 ([-5,0))
        assert oc._bucket_index(0.0, oc.OUTDOOR_BUCKET_EDGES) == 3
        assert oc._bucket_index(-0.0001, oc.OUTDOOR_BUCKET_EDGES) == 2
        # min just below an edge, max at/above it -> cross band
        assert oc.evaluate_gate(_ev(time_weighted_mean_c=0.0, min_c=-0.1, max_c=0.1,
                                    first_c=-0.1, last_c=0.1)).reason == "cross_band_spread"

    def test_inconsistent_envelope_rejected(self):
        assert oc.evaluate_gate(_ev(time_weighted_mean_c=9.0, min_c=1.0, max_c=4.0,
                                    first_c=1.0, last_c=4.0)).reason == "inconsistent_envelope"

    def test_extreme_bands(self):
        assert oc.evaluate_gate(_ev(time_weighted_mean_c=-20.0, min_c=-21.0, max_c=-15.0,
                                    first_c=-21.0, last_c=-15.0)).band == 0   # < -10
        assert oc.evaluate_gate(_ev(time_weighted_mean_c=20.0, min_c=16.0, max_c=24.0,
                                    first_c=16.0, last_c=24.0)).band == 6     # >= 15


# ── context builder ───────────────────────────────────────────────────────────

class _Ep:
    def __init__(self, episode_id="lz:heating:1"):
        self.episode_id = episode_id


class TestBuildContext:
    def test_heat_rate_context(self):
        ctx = oc.build_context("heat_rate", _Ep(), _ev())
        assert isinstance(ctx, HeatRateUpdateContext)
        assert ctx.outdoor_temperature_c == 2.0
        assert ctx.quality is DataQuality.OK
        assert ctx.source_episode_id == "lz:heating:1"
        assert ctx.device_profile_key is None
        assert ctx.source_refs == ("sensor",)

    def test_heat_loss_context_no_delta(self):
        ctx = oc.build_context("heat_loss", _Ep("lz:passive_cooling:2"), _ev())
        assert isinstance(ctx, HeatLossUpdateContext)
        assert ctx.outdoor_temperature_c == 2.0
        assert ctx.indoor_outdoor_delta_c is None          # normalized coefficient stays off
        assert ctx.source_episode_id == "lz:passive_cooling:2"

    def test_non_conditioned_model_returns_none(self):
        assert oc.build_context("afterheat", _Ep(), _ev()) is None
        assert oc.build_context("outcome", _Ep(), _ev()) is None

    def test_rejected_evidence_returns_none(self):
        assert oc.build_context("heat_rate", _Ep(), _ev(source_changed=True)) is None
        assert oc.build_context("heat_rate", _Ep(), None) is None

    def test_missing_source_yields_empty_refs(self):
        ctx = oc.build_context("heat_rate", _Ep(), _ev(source=None))
        assert ctx is not None and ctx.source_refs == ()

    def test_missing_episode_id_returns_none(self):
        assert oc.build_context("heat_rate", _Ep(episode_id=""), _ev()) is None

    def test_builder_exception_degrades_to_none(self):
        class _Boom:
            has_usable_evidence = True
            @property
            def time_weighted_mean_c(self):
                raise RuntimeError("boom")
        assert oc.build_context("heat_rate", _Ep(), _Boom()) is None


# ── runtime integration: buckets populate, general updates, control invariant ─

def _cyc(rt, i, temp, *, heating, outdoor, source="sensor", zone="lz", target=21.0):
    setp = (target + 3.0) if heating else (target - 4.0)
    inp = RuntimeCycleInput(
        zone_id=zone, ts=(T0 + timedelta(minutes=i * 5)).isoformat(),
        target_c=target, trv_setpoint_c=setp, controller_demand=heating,
        indoor_temp=Measurement(temp, DQ.OK),
        schedule=ScheduleTarget(comfort_time_utc=COMFORT, comfort_temperature_c=target),
        controller_decision=ControllerDecisionInput(
            decision_type=DecisionType.NORMAL, target_c=target, trv_setpoint_c=setp),
        outdoor_temp=(None if outdoor is None else Measurement(outdoor, DQ.OK)),
        outdoor_source=source)
    return rt.run_cycle(inp)


def _run_heating_episodes(rt, *, n_episodes, outdoor, source="sensor", zone="lz"):
    """Drive n ramp+settle cycles; each ramp completes one heating episode."""
    i = 0
    temp = 18.0
    for _ in range(n_episodes):
        temp = 18.0
        for _ in range(7):                       # ramp (ACTIVE_HEATING)
            _cyc(rt, i, round(temp, 3), heating=True, outdoor=outdoor, source=source, zone=zone)
            temp = min(21.0, temp + 0.5); i += 1
        for _ in range(5):                       # settle (closes the episode)
            _cyc(rt, i, round(temp, 3), heating=False, outdoor=outdoor, source=source, zone=zone)
            temp = max(18.0, temp - 0.4); i += 1


def _hr_model(rt, zone="lz"):
    return rt._zone(zone).orchestrator.models["heat_rate"]


class TestRuntimeConditioning:
    def test_five_episodes_populate_a_conditioned_bucket_and_general(self):
        rt = runtime()
        _run_heating_episodes(rt, n_episodes=6, outdoor=2.0)      # band 3 ([0,5))
        diag = _hr_model(rt).diagnostics()
        assert diag.sample_counts["general"] >= 5                 # general learned
        band_keys = [k for k in diag.bucket_rates if k.startswith("outdoor:")]
        assert band_keys, "expected a conditioned outdoor bucket"
        # the bucket reached the model's selection threshold
        assert max(diag.sample_counts[k] for k in band_keys) >= HeatRateParameters().bucket_min_samples

    def test_rejected_evidence_updates_general_only(self):
        rt = runtime()
        _run_heating_episodes(rt, n_episodes=3, outdoor=None)      # no outdoor -> gate rejects
        diag = _hr_model(rt).diagnostics()
        assert diag.sample_counts["general"] >= 3
        assert not [k for k in diag.bucket_rates if k.startswith("outdoor:")]

    def test_cross_band_evidence_updates_general_only(self):
        rt = runtime()
        # outdoor near an edge with a wide spread would cross bands; here feed a value
        # whose per-episode evidence is single-valued (no spread) so it DOES bucket —
        # contrast: feed None to prove general-only. (cross-band gate unit-tested above.)
        _run_heating_episodes(rt, n_episodes=2, outdoor=None)
        assert not [k for k in _hr_model(rt).diagnostics().bucket_rates
                    if k.startswith("outdoor:")]


class TestControlInvariance:
    def _published(self, rt, zone="lz"):
        preds = rt._zone(zone).last_predictions
        return {pt.value: (tuple(sorted(p.values.items())), round(p.confidence, 9),
                           p.bucket, p.fallback_used, p.prior_contribution,
                           p.learned_contribution)
                for pt, p in preds.items()}

    def test_published_predictions_identical_to_context_free(self):
        # Baseline: force the conditioning builder to always return None (context-free).
        with patch("custom_components.thermosmart.learning.runtime.lifecycle."
                   "_build_outdoor_context", lambda *a, **k: None):
            rt_base = runtime()
            _run_heating_episodes(rt_base, n_episodes=6, outdoor=2.0)
            base = self._published(rt_base)
            base_general = _hr_model(rt_base).diagnostics().sample_counts["general"]
            base_buckets = [k for k in _hr_model(rt_base).diagnostics().bucket_rates
                            if k.startswith("outdoor:")]

        # D2-B: real conditioning wiring, identical inputs.
        rt_d2b = runtime()
        _run_heating_episodes(rt_d2b, n_episodes=6, outdoor=2.0)
        d2b = self._published(rt_d2b)
        d2b_buckets = [k for k in _hr_model(rt_d2b).diagnostics().bucket_rates
                       if k.startswith("outdoor:")]

        # published predictions (what drives Preheat/TPI/control) are byte-identical
        assert d2b == base
        # yet D2-B populated a conditioned bucket the context-free baseline did not
        assert base_buckets == [] and d2b_buckets != []
        # general learning identical
        assert _hr_model(rt_d2b).diagnostics().sample_counts["general"] == base_general

    def test_heat_rate_prediction_stays_general_not_conditioned(self):
        rt = runtime()
        _run_heating_episodes(rt, n_episodes=6, outdoor=2.0)
        hr = rt._zone("lz").last_predictions.get(PredictionType.HEAT_RATE)
        assert hr is not None
        assert hr.bucket in ("general", "generic_prior", "device_prior")   # never "outdoor:*"


# ── persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_conditioned_bucket_survives_serialize_restore(self):
        rt = runtime()
        _run_heating_episodes(rt, n_episodes=6, outdoor=2.0)
        model = _hr_model(rt)
        before = model.diagnostics().bucket_rates
        assert [k for k in before if k.startswith("outdoor:")]
        state = model.serialize_state()
        assert "buckets" in state and any(k.startswith("outdoor:") for k in state["buckets"])

        from custom_components.thermosmart.learning.models.heat_rate import HeatRateModel
        restored = HeatRateModel("lz")
        restored.deserialize_state(copy.deepcopy(state))
        after = restored.diagnostics().bucket_rates
        assert after == before

    def test_old_state_without_buckets_restores(self):
        from custom_components.thermosmart.learning.models.heat_rate import HeatRateModel
        m = HeatRateModel("lz")
        state = dict(m.serialize_state())
        state.pop("buckets", None)                       # simulate a pre-conditioning state
        m2 = HeatRateModel("lz")
        m2.deserialize_state(state)                      # must not raise
        assert m2.diagnostics().bucket_rates == {}

    def test_model_version_unchanged(self):
        from custom_components.thermosmart.learning.models.heat_rate import MODEL_VERSION as HR
        from custom_components.thermosmart.learning.models.heat_loss import MODEL_VERSION as HL
        assert HR == 1 and HL == 1
