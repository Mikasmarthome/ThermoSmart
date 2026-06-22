"""Phase 15 ManualCorrectionModel eligibility + registry tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import Model, PredictionType
from custom_components.thermosmart.learning.raw_schemas import (
    CorrectionActionType,
    CorrectionSource,
    ManualCorrectionEvent,
    RawTrackName,
    RoomObservation,
)
from custom_components.thermosmart.learning.models import (
    ManualCorrectionModel,
    ManualCorrectionRejection,
    manual_correction_model_definition,
)
from custom_components.thermosmart.learning.contracts import ExportScope
from custom_components.thermosmart.learning.registry import (
    EpisodeRegistry,
    ModelRegistry,
    PrivacyClass,
    RawTrackDefinition,
    RawTrackRegistry,
    RetentionPolicy,
    validate_registry_graph,
)
from tests.helpers_manual_correction import correction_context, correction_event


def m():
    return ManualCorrectionModel("lz")


class TestEligibility:
    def test_is_model(self):
        assert isinstance(m(), Model)

    def test_valid_correction(self):
        assert m().update_eligibility(correction_event("e", 21.0, 21.5)).accepted

    def test_wrong_event_type(self):
        from datetime import datetime, timezone
        ro = RoomObservation(learning_zone_id="lz", ts=datetime(2025, 1, 1, tzinfo=timezone.utc),
                             indoor_temp=20.0, target=21.0)
        assert ManualCorrectionRejection.WRONG_EVENT_TYPE.value in \
            m().update_eligibility(ro).confounder_flags

    def test_wrong_zone(self):
        assert ManualCorrectionRejection.WRONG_ZONE.value in \
            m().update_eligibility(correction_event("e", 21.0, 21.5, zone="other")).confounder_flags

    def test_zero_correction(self):
        assert ManualCorrectionRejection.ZERO_CORRECTION.value in \
            m().update_eligibility(correction_event("e", 21.0, 21.0)).confounder_flags

    def test_tiny_correction_below_threshold(self):
        assert ManualCorrectionRejection.ZERO_CORRECTION.value in \
            m().update_eligibility(correction_event("e", 21.0, 21.05)).confounder_flags

    def test_implausible_correction(self):
        assert ManualCorrectionRejection.IMPLAUSIBLE_CORRECTION.value in \
            m().update_eligibility(correction_event("e", 21.0, 40.0)).confounder_flags

    def test_mode_change_rejected(self):
        ev = correction_event("e", 21.0, 21.5, action=CorrectionActionType.MODE_CHANGE)
        assert ManualCorrectionRejection.MODE_CHANGE.value in \
            m().update_eligibility(ev).confounder_flags

    def test_internal_echo(self):
        ev = correction_event("e", 21.0, 21.5)
        ctx = correction_context(ev, last_sent_setpoint_c=21.5)
        assert ManualCorrectionRejection.INTERNAL_COMMAND_ECHO.value in \
            m().update(ev, ctx).confounder_flags

    def test_device_sync(self):
        ev = correction_event("e", 21.0, 21.5)
        ctx = correction_context(ev, is_device_sync=True)
        assert ManualCorrectionRejection.DEVICE_SYNC.value in m().update(ev, ctx).confounder_flags

    def test_duplicate(self):
        model = m()
        model.update(correction_event("dup", 21.0, 21.5))
        assert ManualCorrectionRejection.DUPLICATE_EVENT.value in \
            model.update_eligibility(correction_event("dup", 21.0, 21.5)).confounder_flags

    def test_context_mismatch(self):
        ev = correction_event("e", 21.0, 21.5)
        ctx = correction_context(ev)
        bad = type(ctx)(**{**ctx.to_dict(), "correction_event_id": "other"})
        assert ManualCorrectionRejection.CONTEXT_MISMATCH.value in \
            m().update(ev, bad).confounder_flags

    def test_temporary_exception(self):
        ev = correction_event("e", 21.0, 21.5)
        ctx = correction_context(ev, reverted_within_s=120.0)
        assert ManualCorrectionRejection.TEMPORARY_EXCEPTION.value in \
            m().update(ev, ctx).confounder_flags


class TestRegistry:
    def test_definition_valid(self):
        raw = RawTrackRegistry()
        raw.register(RawTrackDefinition(
            track_name=RawTrackName.MANUAL_CORRECTION, schema_type=ManualCorrectionEvent,
            track_schema_version=1, privacy_class=PrivacyClass.BEHAVIORAL,
            allowed_export_scopes=(ExportScope.SUPPORT, ExportScope.RESEARCH),
            retention=RetentionPolicy(max_records=2000, segmented=True), segmented=True,
            immutable_events=True))
        ep = EpisodeRegistry()
        models = ModelRegistry()
        models.register(manual_correction_model_definition())
        assert validate_registry_graph(raw, ep, models) == []

    def test_definition_properties(self):
        d = manual_correction_model_definition()
        assert d.min_trv_only and d.advisory and not d.control_relevant and d.rebuildable
        assert RawTrackName.MANUAL_CORRECTION in d.consumed_raw_tracks
        assert PredictionType.MANUAL_PREFERENCE_BIAS in d.supported_prediction_types
        assert PredictionType.CONTROLLER_CALIBRATION in d.supported_prediction_types

    def test_no_capability_tier(self):
        assert "tier" not in manual_correction_model_definition().required_capability_flags

    def test_factory_produces_model(self):
        assert isinstance(manual_correction_model_definition().create(), Model)
