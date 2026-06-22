"""Shared builders for LE 2.0 OutcomeModel tests."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

from custom_components.thermosmart.learning.contracts import DataQuality, Regime
from custom_components.thermosmart.learning.episode_schemas import (
    ControllerKind,
    EpisodeReason,
    OutcomeEpisode,
    Trajectory,
    TrajectoryPoint,
)

T0 = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)


def trajectory(vals: Sequence[Optional[float]], *, step_ms: int = 120000,
               first_offset_ms: int = 0, max_points: int = 120) -> Trajectory:
    pts = []
    for i, v in enumerate(vals):
        if v is None:
            pts.append(TrajectoryPoint(first_offset_ms + i * step_ms, None,
                                       DataQuality.UNAVAILABLE, gap=True))
        else:
            pts.append(TrajectoryPoint(first_offset_ms + i * step_ms, float(v)))
    return Trajectory(points=tuple(pts), max_points=max_points)


def outcome_episode(episode_id: str, vals: Sequence[Optional[float]], *,
                    reason: EpisodeReason = EpisodeReason.REACHED,
                    target: float = 21.0, start_temp: float = 19.0,
                    end_temp: Optional[float] = None, tolerance: float = 0.5,
                    controller: ControllerKind = ControllerKind.THERMOSMART,
                    reliability: float = 0.85, zone: str = "lz",
                    decision_id: Optional[str] = None,
                    duration_min: int = 20, time_quality: DataQuality = DataQuality.OK,
                    confounder_flags=(), schema_version: int = 1) -> OutcomeEpisode:
    real_vals = [v for v in vals if v is not None]
    if end_temp is None:
        end_temp = real_vals[-1] if real_vals else start_temp
    if decision_id is None:
        decision_id = f"dec_{episode_id}"
    return OutcomeEpisode(
        episode_id=episode_id, learning_zone_id=zone, decision_id=decision_id,
        episode_schema_version=schema_version,
        builder_version=1, classifier_version=1, start_ts=T0,
        end_ts=T0.replace(minute=duration_min), regime=Regime.ACTIVE_HEATING,
        reliability=reliability, start_temp=start_temp, end_temp=end_temp, target=target,
        comfort_tolerance_at_start=tolerance, reason=reason, controller=controller,
        trajectory=trajectory(vals), time_quality=time_quality,
        confounder_flags=tuple(confounder_flags))


def reached_clean(eid="r", reason=EpisodeReason.REACHED, **kw):
    return outcome_episode(eid, [19.0, 20.0, 21.0, 21.0], reason=reason, **kw)


def reached_overshoot(eid="o", reason=EpisodeReason.REACHED, **kw):
    return outcome_episode(eid, [19.0, 21.0, 23.0, 22.5], reason=reason, **kw)


def not_reached(eid="nr", reason=EpisodeReason.TIMEOUT, **kw):
    return outcome_episode(eid, [19.0, 19.3, 19.6, 19.8], reason=reason, **kw)


def no_temp(eid="nt", reason=EpisodeReason.TIMEOUT, **kw):
    return outcome_episode(eid, [None, None, None, None], reason=reason, **kw)
