"""Phase 11 OutcomeModel eligibility + registry tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import (
    Model,
    PredictionType,
    Regime,
)
from custom_components.thermosmart.learning.episode_schemas import (
    ControllerKind,
    EpisodeReason,
    EpisodeType,
    HeatingEpisode,
    Trajectory,
    TrajectoryPoint,
)
from custom_components.thermosmart.learning.models import (
    OutcomeModel,
    OutcomeRejection,
    OutcomeUpdateContext,
    outcome_model_definition,
)
from custom_components.thermosmart.learning.registry import (
    EpisodeDefinition,
    EpisodeRegistry,
    ModelRegistry,
    RawTrackRegistry,
    RetentionPolicy,
    validate_registry_graph,
)
from tests.helpers_outcome import (
    T0,
    no_temp,
    not_reached,
    outcome_episode,
    reached_clean,
)


def m():
    return OutcomeModel("lz")


class TestEligibility:
    def test_valid_outcome(self):
        r = m().update_eligibility(reached_clean())
        assert r.accepted

    def test_is_model(self):
        assert isinstance(m(), Model)

    def test_wrong_episode_type(self):
        traj = Trajectory(points=(TrajectoryPoint(0, 18.0), TrajectoryPoint(60000, 18.5)),
                          max_points=10)
        he = HeatingEpisode(episode_id="h", learning_zone_id="lz", episode_schema_version=1,
                            builder_version=1, classifier_version=1, start_ts=T0,
                            end_ts=T0.replace(minute=10), regime=Regime.ACTIVE_HEATING,
                            reliability=0.8, start_temp=18.0, target=21.0, trajectory=traj)
        assert OutcomeRejection.WRONG_EPISODE_TYPE.value in \
            m().update_eligibility(he).confounder_flags

    def test_wrong_zone(self):
        assert OutcomeRejection.WRONG_ZONE.value in \
            m().update_eligibility(reached_clean(zone="other")).confounder_flags

    def test_disturbed_blocked(self):
        r = m().update_eligibility(reached_clean(reason=EpisodeReason.DISTURBED))
        assert OutcomeRejection.DISTURBED.value in r.confounder_flags

    def test_insufficient_data_not_learned(self):
        r = m().update_eligibility(reached_clean(reason=EpisodeReason.INSUFFICIENT_DATA))
        assert OutcomeRejection.INSUFFICIENT_DATA.value in r.confounder_flags

    def test_low_data_quality(self):
        r = m().update_eligibility(reached_clean(reliability=0.1))
        assert OutcomeRejection.LOW_DATA_QUALITY.value in r.confounder_flags

    def test_version_mismatch(self):
        r = m().update_eligibility(reached_clean(schema_version=2))
        assert OutcomeRejection.VERSION_MISMATCH.value in r.confounder_flags

    def test_duplicate(self):
        model = m()
        model.update(reached_clean("dup"))
        assert OutcomeRejection.DUPLICATE_EPISODE.value in \
            model.update_eligibility(reached_clean("dup")).confounder_flags

    def test_decision_match_via_context(self):
        ep = reached_clean("e")  # decision_id == "dec_e"
        r = m().update(ep, OutcomeUpdateContext(source_episode_id="e", decision_id="dec_e"))
        assert r.accepted

    def test_decision_mismatch_via_context(self):
        r = m().update(reached_clean("e"),
                       OutcomeUpdateContext(source_episode_id="e", decision_id="zzz"))
        assert OutcomeRejection.DECISION_MISMATCH.value in r.confounder_flags

    def test_no_temp_partial_accepted(self):
        # timeout without temperature still has controller/time evidence
        r = m().update_eligibility(no_temp(reason=EpisodeReason.TIMEOUT))
        assert r.accepted  # partial, controller-based

    def test_timeout_with_good_data_accepted(self):
        assert m().update_eligibility(not_reached()).accepted


class TestSupportedReasons:
    @pytest.mark.parametrize("reason", [
        EpisodeReason.REACHED, EpisodeReason.TIMEOUT, EpisodeReason.INTERRUPTED,
        EpisodeReason.MANUAL_CORRECTION, EpisodeReason.SUPERSEDED, EpisodeReason.MODE_CHANGE,
    ])
    def test_supported(self, reason):
        assert m().update_eligibility(reached_clean(reason=reason)).accepted


class TestDecisionBinding:
    def test_episode_carries_decision_id(self):
        assert reached_clean("e").decision_id == "dec_e"

    def test_empty_decision_id_rejected(self):
        from custom_components.thermosmart.learning.episode_schemas import OutcomeEpisode
        from custom_components.thermosmart.learning.episode_schemas import (
            ControllerKind as CK,
        )
        from tests.helpers_outcome import trajectory
        with pytest.raises(ValueError):
            OutcomeEpisode(
                episode_id="e", learning_zone_id="lz", decision_id="",
                episode_schema_version=1, builder_version=1, classifier_version=1,
                start_ts=T0, end_ts=T0.replace(minute=10), regime=Regime.ACTIVE_HEATING,
                reliability=0.8, start_temp=19.0, end_temp=21.0, target=21.0,
                comfort_tolerance_at_start=0.5, reason=EpisodeReason.REACHED,
                controller=CK.THERMOSMART, trajectory=trajectory([19.0, 20.0, 21.0]))

    def test_identical_decision_no_duplicate(self):
        model = m()
        # two distinct episode ids, same controller decision -> learn once
        model.update(reached_clean("e1", decision_id="dec_X"))
        r = model.update(reached_clean("e2", decision_id="dec_X"))
        assert OutcomeRejection.DUPLICATE_EPISODE.value in r.confounder_flags
        assert model._state.general.sample_count == 1

    def test_two_zones_same_external_token_not_mixed(self):
        a = OutcomeModel("lz_a")
        b = OutcomeModel("lz_b")
        a.update(reached_clean("e", zone="lz_a", decision_id="shared"))
        rb = b.update(reached_clean("e", zone="lz_b", decision_id="shared"))
        assert rb.accepted  # isolated per zone, no unvalidated cross-zone collision
        assert a._state.general.sample_count == 1 and b._state.general.sample_count == 1

    def test_rebuild_decision_mismatch_rejected(self):
        from custom_components.thermosmart.learning.models import OutcomeRebuildItem
        ep = reached_clean("e", decision_id="dec_e")
        bad = OutcomeRebuildItem(
            episode=ep, context=OutcomeUpdateContext(source_episode_id="e", decision_id="other"))
        rb = m()
        res = rb.rebuild(None, [bad])
        assert res.accepted_count == 0 and res.rejected_count == 1

    def test_sample_keeps_internal_decision_id(self):
        model = m()
        model.update(reached_clean("e", decision_id="dec_secret"))
        assert model._state.recent_samples[0].decision_id == "dec_secret"

    def test_support_export_no_decision_id(self):
        from custom_components.thermosmart.learning.contracts import ExportScope
        model = m()
        model.update(reached_clean("e", decision_id="dec_secret"))
        import json
        assert "dec_secret" not in json.dumps(model.export(ExportScope.SUPPORT))

    def test_research_export_no_decision_id(self):
        from custom_components.thermosmart.learning.contracts import ExportScope
        model = m()
        model.update(reached_clean("e", decision_id="dec_secret"))
        import json
        assert "dec_secret" not in json.dumps(model.export(ExportScope.RESEARCH))

    def test_serialization_keeps_internal_decision_id(self):
        model = m()
        model.update(reached_clean("e", decision_id="dec_internal"))
        s = model.serialize_state()
        assert "dec_internal" in s["processed_decision_ids"]
        b = m()
        b.deserialize_state(s)
        assert b._state.processed_decision_ids == ("dec_internal",)


class TestRegistry:
    def test_definition_valid(self):
        raw = RawTrackRegistry()
        ep = EpisodeRegistry()
        from custom_components.thermosmart.learning.episode_schemas import OutcomeEpisode
        ep.register(EpisodeDefinition(
            episode_type=EpisodeType.OUTCOME, schema_type=OutcomeEpisode,
            episode_schema_version=1, builder_version=1, consumed_raw_tracks=(),
            max_trajectory_points=120, max_duration_seconds=10800,
            retention=RetentionPolicy(), trajectory_required=True, materialized=True))
        models = ModelRegistry()
        models.register(outcome_model_definition())
        assert validate_registry_graph(raw, ep, models) == []

    def test_definition_properties(self):
        d = outcome_model_definition()
        assert d.min_trv_only and d.advisory and not d.control_relevant and d.rebuildable
        assert EpisodeType.OUTCOME in d.consumed_episode_types
        assert PredictionType.OUTCOME_QUALITY in d.supported_prediction_types
        assert PredictionType.CONTROLLER_QUALITY in d.supported_prediction_types

    def test_factory_produces_model(self):
        assert isinstance(outcome_model_definition().create(), Model)
