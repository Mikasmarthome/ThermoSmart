"""Shared helpers for Phase 10 AfterheatModel tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.thermosmart.learning.contracts import Regime
from custom_components.thermosmart.learning.episode_schemas import (
    AfterheatEpisode,
    Trajectory,
    TrajectoryPoint,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def afterheat_episode(
    episode_id: str, residual_rise_c: float, *, close_temp: float = 20.0,
    duration_min: float = 10.0, points: int = 6, reliability: float = 0.8,
    regime: Regime = Regime.AFTERHEAT, confounder_flags=(),
    setpoint_before: float = 22.0, setpoint_after: float = 16.0,
    episode_schema_version: int = 1, start_offset_min: float = 0.0, zone: str = "lz",
) -> AfterheatEpisode:
    """Build an AfterheatEpisode rising to close_temp+residual_rise then easing."""
    peak = close_temp + residual_rise_c
    rise_points = max(1, points - 1)
    vals = [close_temp + (peak - close_temp) * min(i / rise_points, 1.0) for i in range(points)]
    if residual_rise_c > 0 and points >= 3:
        vals[-1] = peak - min(0.02, residual_rise_c)  # slight ease so peak precedes the end
    step_ms = int(duration_min * 60_000 / (points - 1)) if points > 1 else 60_000
    traj = Trajectory(
        points=tuple(TrajectoryPoint(i * step_ms, float(v)) for i, v in enumerate(vals)),
        max_points=240)
    start = T0 + timedelta(minutes=start_offset_min)
    return AfterheatEpisode(
        episode_id=episode_id, learning_zone_id=zone,
        episode_schema_version=episode_schema_version, builder_version=1, classifier_version=1,
        start_ts=start, end_ts=start + timedelta(minutes=duration_min), regime=regime,
        reliability=reliability, indoor_temp_at_close=close_temp, target=21.0,
        trv_setpoint_before=setpoint_before, trv_setpoint_after=setpoint_after,
        trajectory=traj, confounder_flags=tuple(confounder_flags))
