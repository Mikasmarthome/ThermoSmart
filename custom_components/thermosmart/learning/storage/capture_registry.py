"""Canonical Learning episode-type registry definitions.

Builds the real ``EpisodeRegistry`` — the single authoritative place declaring
which episode types exist and what retention each one declares. Pure Python: no
HA imports, no storage I/O. Constructing the registry never writes anything — it
is a declaration, matching the existing ``RetentionPolicy`` docstring contract
("no pruning engine here").

See ``capture_stores.py`` for the thin store-access facade built from it.

The retention values are conservative, explicit defaults — not final production
tuning: bounded, higher-value summaries, easy to recalibrate later without any
structural change.
"""
from __future__ import annotations

from ..episode_schemas import (
    AfterheatEpisode,
    EpisodeType,
    HeatingEpisode,
    OutcomeEpisode,
    PassiveCoolingEpisode,
    WindowCoolingEpisode,
)
from ..registry import (
    EpisodeDefinition,
    EpisodeRegistry,
    RetentionPolicy,
)

# Episodes: bounded count and age — each one is a materialised, already-condensed
# summary, so keeping a good number of them is cheap and useful for
# research/debugging.
EPISODE_RETENTION_DEFAULT = RetentionPolicy(max_age_days=90, max_records=500)


def build_episode_registry() -> EpisodeRegistry:
    """Build the canonical EpisodeRegistry with all 5 declared episode types.

    Never raises for the fixed, hand-verified definitions below.
    """
    reg = EpisodeRegistry()
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.HEATING, schema_type=HeatingEpisode,
        episode_schema_version=1, builder_version=1,
        max_trajectory_points=120, max_duration_seconds=7200,
        retention=EPISODE_RETENTION_DEFAULT,
        trajectory_required=False, materialized=True,
    ))
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.AFTERHEAT, schema_type=AfterheatEpisode,
        episode_schema_version=1, builder_version=1,
        max_trajectory_points=120, max_duration_seconds=7200,
        retention=EPISODE_RETENTION_DEFAULT,
        trajectory_required=True, materialized=True,
    ))
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.PASSIVE_COOLING, schema_type=PassiveCoolingEpisode,
        episode_schema_version=1, builder_version=1,
        max_trajectory_points=120, max_duration_seconds=14400,
        retention=EPISODE_RETENTION_DEFAULT,
        trajectory_required=True, materialized=True,
    ))
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.WINDOW_COOLING, schema_type=WindowCoolingEpisode,
        episode_schema_version=1, builder_version=1,
        max_trajectory_points=120, max_duration_seconds=7200,
        retention=EPISODE_RETENTION_DEFAULT,
        trajectory_required=False, materialized=True,
    ))
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.OUTCOME, schema_type=OutcomeEpisode,
        episode_schema_version=1, builder_version=1,
        max_trajectory_points=120, max_duration_seconds=7200,
        retention=EPISODE_RETENTION_DEFAULT,
        trajectory_required=True, materialized=True,
    ))
    return reg
