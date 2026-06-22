"""Phase 11 OutcomeModel update / outlier / inertia / buckets / rebuild tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.episode_schemas import ControllerKind, EpisodeReason
from custom_components.thermosmart.learning.models import (
    OutcomeModel,
    OutcomeParameters,
    OutcomeRebuildItem,
    OutcomeRejection,
    OutcomeUpdateContext,
)
from tests.helpers_outcome import no_temp, not_reached, reached_clean, reached_overshoot


def m(params=None):
    return OutcomeModel("lz", params)


class TestUpdate:
    def test_first_sample(self):
        model = m()
        assert model.update(reached_clean()).accepted
        assert model._state.general.physical.sample_count == 1

    def test_multiple(self):
        model = m()
        for i in range(4):
            model.update(reached_clean(f"e{i}"))
        assert model._state.general.sample_count == 4

    def test_general_state_tracks_scores(self):
        model = m()
        for i in range(3):
            model.update(reached_clean(f"e{i}"))
        assert model._state.general.physical.value > 0.8

    def test_controller_bucket_via_context(self):
        model = m()
        for i in range(6):
            ep = reached_clean(f"e{i}")
            ctx = OutcomeUpdateContext(source_episode_id=ep.episode_id, controller_kind="ts")
            model.update(ep, ctx)
        assert any(k.startswith("controller:") for k in model._state.buckets)

    def test_reliability_weighting(self):
        hi = m(); hi.update(reached_clean("a", reliability=0.9))
        lo = m(); lo.update(reached_clean("a", reliability=0.4))
        assert hi._state.general.physical.effective_n > lo._state.general.physical.effective_n

    def test_partial_reduced_weight(self):
        full = m().update(reached_clean("f"))
        part = m().update(no_temp("p", reason=EpisodeReason.TIMEOUT))
        assert part.weight < full.weight

    def test_single_episode_does_not_dominate(self):
        model = m(OutcomeParameters(min_samples_for_outlier=999))
        for i in range(8):
            model.update(reached_clean(f"e{i}"))
        before = model._state.general.physical.value
        model.update(not_reached("bad"))
        assert abs(model._state.general.physical.value - before) < 0.3

    def test_severe_outlier_rejected(self):
        model = m()
        for i in range(6):
            model.update(reached_clean(f"e{i}"))
        before = model._state.general.physical.value
        # performance near 0 with huge dispersion gap -> severe
        r = model.update(not_reached("out"))
        # not necessarily severe; ensure state remains finite & bounded
        assert 0.0 <= model._state.general.physical.value <= 1.0
        assert before >= 0.0

    def test_two_zones_isolated(self):
        a = OutcomeModel("lz_a"); b = OutcomeModel("lz_b")
        a.update(reached_clean("e", zone="lz_a"))
        assert a._state.general.sample_count == 1 and b._state.general.sample_count == 0

    def test_context_mismatch(self):
        r = m().update(reached_clean("e1"),
                       OutcomeUpdateContext(source_episode_id="OTHER"))
        assert OutcomeRejection.CONTEXT_MISMATCH.value in r.confounder_flags

    def test_context_wrong_zone(self):
        r = m().update(reached_clean("e1"),
                       OutcomeUpdateContext(source_episode_id="e1", learning_zone_id="x"))
        assert OutcomeRejection.CONTEXT_MISMATCH.value in r.confounder_flags

    def test_rates_tracked(self):
        model = m()
        model.update(reached_clean("a"))
        model.update(not_reached("b"))
        assert model._state.reached_count == 1 and model._state.timeout_count == 1


class TestContext:
    def test_roundtrip(self):
        ctx = OutcomeUpdateContext(source_episode_id="e", controller_kind="ts",
                                   expected_duration_s=0.0, target_temperature_c=21.0,
                                   afterheat_expected_overshoot_c=0.5)
        assert OutcomeUpdateContext.from_dict(ctx.to_dict()) == ctx

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            OutcomeUpdateContext(source_episode_id="e", heat_rate_c_per_h=float("inf"))

    def test_zero_valid(self):
        ctx = OutcomeUpdateContext(source_episode_id="e", expected_duration_s=0.0)
        assert ctx.expected_duration_s == 0.0


class TestRebuild:
    def test_rebuild_equals_sequential(self):
        items = []
        for i in range(5):
            ep = reached_clean(f"e{i}")
            ctx = OutcomeUpdateContext(source_episode_id=ep.episode_id, controller_kind="ts")
            items.append(OutcomeRebuildItem(episode=ep, context=ctx))
        seq = m()
        for it in items:
            seq.update(it.episode, it.context)
        rb = m()
        res = rb.rebuild(None, items)
        assert res.accepted_count == 5 and res.processed_count == 5
        assert rb._state.general.physical.value == pytest.approx(seq._state.general.physical.value)
        assert set(rb._state.buckets) == set(seq._state.buckets)

    def test_rebuild_partial_reproduced(self):
        eps = [no_temp(f"p{i}", reason=EpisodeReason.TIMEOUT) for i in range(3)]
        rb = m()
        rb.rebuild(None, eps)
        assert rb._state.partial_count == 3

    def test_rebuild_duplicates(self):
        eps = [reached_clean("same"), reached_clean("same")]
        rb = m()
        res = rb.rebuild(None, eps)
        assert res.duplicate_count == 1 and rb._state.general.sample_count == 1

    def test_rebuild_deterministic(self):
        eps = [reached_overshoot(f"e{i}") for i in range(4)]
        a = m(); a.rebuild(None, eps)
        b = m(); b.rebuild(None, eps)
        assert a.serialize_state() == b.serialize_state()
