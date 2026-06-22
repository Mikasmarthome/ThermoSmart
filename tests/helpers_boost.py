"""Shared builders for LE 2.0 BoostModel tests."""
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
from custom_components.thermosmart.learning.models import BoostUpdateContext

T0 = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)


def trajectory(vals: Sequence[Optional[float]], *, step_ms: int = 300000) -> Trajectory:
    pts = []
    for i, v in enumerate(vals):
        if v is None:
            pts.append(TrajectoryPoint(i * step_ms, None, DataQuality.UNAVAILABLE, gap=True))
        else:
            pts.append(TrajectoryPoint(i * step_ms, float(v)))
    return Trajectory(points=tuple(pts), max_points=120)


def boost_episode(episode_id: str, vals: Sequence[Optional[float]], *, decision_id: Optional[str] = None,
                  reason: EpisodeReason = EpisodeReason.REACHED, target: float = 21.0,
                  start_temp: float = 18.0, tolerance: float = 0.5, zone: str = "lz",
                  reliability: float = 0.85, duration_min: int = 20,
                  confounder_flags=(), schema_version: int = 1) -> OutcomeEpisode:
    real = [v for v in vals if v is not None]
    end_temp = real[-1] if real else start_temp
    if decision_id is None:
        decision_id = f"dec_{episode_id}"
    return OutcomeEpisode(
        episode_id=episode_id, learning_zone_id=zone, decision_id=decision_id,
        episode_schema_version=schema_version, builder_version=1, classifier_version=1,
        start_ts=T0, end_ts=T0.replace(minute=duration_min), regime=Regime.ACTIVE_HEATING,
        reliability=reliability, start_temp=start_temp, end_temp=end_temp, target=target,
        comfort_tolerance_at_start=tolerance, reason=reason,
        controller=ControllerKind.THERMOSMART, trajectory=trajectory(vals),
        confounder_flags=tuple(confounder_flags))


def boost_context(episode, *, requested_offset_c: float = 1.5, start_deficit_c: float = 3.0,
                  actual_duration_s: float = 1800.0, planned_duration_s: float = 1800.0,
                  controller_kind: str = "ts", expected_non_boost_duration_s=None,
                  expected_afterheat_rise_c=None, manual_correction_ref=None,
                  zone=None) -> BoostUpdateContext:
    return BoostUpdateContext(
        source_episode_id=episode.episode_id, decision_id=episode.decision_id,
        learning_zone_id=zone, requested_offset_c=requested_offset_c,
        start_deficit_c=start_deficit_c, actual_duration_s=actual_duration_s,
        planned_duration_s=planned_duration_s, target_temperature_c=episode.target,
        controller_kind=controller_kind,
        expected_non_boost_duration_s=expected_non_boost_duration_s,
        expected_afterheat_rise_c=expected_afterheat_rise_c,
        manual_correction_ref=manual_correction_ref)


def good_boost(model, idx, *, offset=1.5, vals=(18.0, 19.5, 20.5, 21.0, 21.0)):
    ep = boost_episode(f"e{idx}", list(vals), decision_id=f"dec{idx}")
    return model.update(ep, boost_context(ep, requested_offset_c=offset))
