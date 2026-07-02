"""Canonical LE2 raw-track and episode-type registry definitions.

Builds the real ``RawTrackRegistry``/``EpisodeRegistry`` graph — the single
authoritative place declaring which raw tracks and episode types exist, and
what retention each one declares. Pure Python: no HA imports, no storage I/O.
Constructing these registries never writes anything — they are declarations,
matching the existing ``RetentionPolicy`` docstring contract ("no pruning
engine here").

Foundation only. See ``capture_stores.py`` for the thin, live-reachable,
write-nothing store-access facade built from these registries.

Retention values here are conservative, explicit defaults — not final
production tuning. Raw samples get a short window (they are the highest-volume,
lowest-value-per-record data); episodes get a longer window (bounded,
higher-value summaries). Both are deliberately small numbers chosen for a safe
foundation, easy to recalibrate later without any structural change.
"""
from __future__ import annotations

from ..contracts import ExportScope
from ..episode_schemas import (
    AfterheatEpisode,
    EpisodeType,
    HeatingEpisode,
    OutcomeEpisode,
    PassiveCoolingEpisode,
    WindowCoolingEpisode,
)
from ..raw_schemas import (
    ForecastDecisionEvent,
    ForecastEvaluationEvent,
    HeatingDriveEndEvent,
    ManualCorrectionEvent,
    RawTrackName,
    RoomObservation,
    TrvObservation,
)
from ..registry import (
    EpisodeDefinition,
    EpisodeRegistry,
    PrivacyClass,
    RawTrackDefinition,
    RawTrackRegistry,
    RetentionPolicy,
)

# Raw samples: short retention, bounded count — highest volume, lowest value
# per individual record once superseded by later samples of the same track.
RAW_RETENTION_DEFAULT = RetentionPolicy(max_age_days=14, max_records=200)

# Episodes: longer retention, bounded count — each one is a materialised,
# already-condensed summary, so keeping more of them for longer is cheap
# relative to raw samples and far more useful for research/debugging.
EPISODE_RETENTION_DEFAULT = RetentionPolicy(max_age_days=90, max_records=500)


def build_raw_track_registry() -> RawTrackRegistry:
    """Build the canonical RawTrackRegistry with all 6 declared raw tracks.

    Never raises for the fixed, hand-verified definitions below (each one
    matches a canonical schema pairing enforced by RawTrackDefinition itself).
    """
    reg = RawTrackRegistry()
    reg.register(RawTrackDefinition(
        track_name=RawTrackName.ROOM, schema_type=RoomObservation,
        track_schema_version=1, privacy_class=PrivacyClass.LOW,
        allowed_export_scopes=(ExportScope.SUPPORT, ExportScope.RESEARCH),
        retention=RAW_RETENTION_DEFAULT, segmented=True, immutable_events=False,
    ))
    reg.register(RawTrackDefinition(
        track_name=RawTrackName.TRV, schema_type=TrvObservation,
        track_schema_version=1, privacy_class=PrivacyClass.LOW,
        allowed_export_scopes=(ExportScope.SUPPORT, ExportScope.RESEARCH),
        retention=RAW_RETENTION_DEFAULT, segmented=True, immutable_events=False,
    ))
    reg.register(RawTrackDefinition(
        track_name=RawTrackName.HEATING_DRIVE_END, schema_type=HeatingDriveEndEvent,
        track_schema_version=1, privacy_class=PrivacyClass.LOW,
        allowed_export_scopes=(ExportScope.SUPPORT, ExportScope.RESEARCH),
        retention=RAW_RETENTION_DEFAULT, segmented=True, immutable_events=True,
    ))
    reg.register(RawTrackDefinition(
        track_name=RawTrackName.FORECAST_DECISION, schema_type=ForecastDecisionEvent,
        track_schema_version=1, privacy_class=PrivacyClass.LOW,
        allowed_export_scopes=(ExportScope.SUPPORT, ExportScope.RESEARCH),
        retention=RAW_RETENTION_DEFAULT, segmented=True, immutable_events=True,
    ))
    reg.register(RawTrackDefinition(
        track_name=RawTrackName.FORECAST_EVALUATION, schema_type=ForecastEvaluationEvent,
        track_schema_version=1, privacy_class=PrivacyClass.LOW,
        allowed_export_scopes=(ExportScope.SUPPORT, ExportScope.RESEARCH),
        retention=RAW_RETENTION_DEFAULT, segmented=True, immutable_events=True,
    ))
    reg.register(RawTrackDefinition(
        track_name=RawTrackName.MANUAL_CORRECTION, schema_type=ManualCorrectionEvent,
        track_schema_version=1, privacy_class=PrivacyClass.BEHAVIORAL,
        allowed_export_scopes=(ExportScope.SUPPORT,),
        retention=RAW_RETENTION_DEFAULT, segmented=True, immutable_events=True,
    ))
    return reg


def build_episode_registry() -> EpisodeRegistry:
    """Build the canonical EpisodeRegistry with all 5 declared episode types.

    Never raises for the fixed, hand-verified definitions below.
    """
    reg = EpisodeRegistry()
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.HEATING, schema_type=HeatingEpisode,
        episode_schema_version=1, builder_version=1,
        consumed_raw_tracks=(RawTrackName.ROOM, RawTrackName.TRV),
        max_trajectory_points=120, max_duration_seconds=7200,
        retention=EPISODE_RETENTION_DEFAULT,
        trajectory_required=False, materialized=True,
    ))
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.AFTERHEAT, schema_type=AfterheatEpisode,
        episode_schema_version=1, builder_version=1,
        consumed_raw_tracks=(RawTrackName.ROOM, RawTrackName.HEATING_DRIVE_END),
        max_trajectory_points=120, max_duration_seconds=7200,
        retention=EPISODE_RETENTION_DEFAULT,
        trajectory_required=True, materialized=True,
    ))
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.PASSIVE_COOLING, schema_type=PassiveCoolingEpisode,
        episode_schema_version=1, builder_version=1,
        consumed_raw_tracks=(RawTrackName.ROOM,),
        max_trajectory_points=120, max_duration_seconds=14400,
        retention=EPISODE_RETENTION_DEFAULT,
        trajectory_required=True, materialized=True,
    ))
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.WINDOW_COOLING, schema_type=WindowCoolingEpisode,
        episode_schema_version=1, builder_version=1,
        consumed_raw_tracks=(RawTrackName.ROOM,),
        max_trajectory_points=120, max_duration_seconds=7200,
        retention=EPISODE_RETENTION_DEFAULT,
        trajectory_required=False, materialized=True,
    ))
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.OUTCOME, schema_type=OutcomeEpisode,
        episode_schema_version=1, builder_version=1,
        consumed_raw_tracks=(RawTrackName.ROOM,),
        max_trajectory_points=120, max_duration_seconds=7200,
        retention=EPISODE_RETENTION_DEFAULT,
        trajectory_required=True, materialized=True,
    ))
    return reg
