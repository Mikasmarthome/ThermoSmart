"""Shared helpers for Phase 9 HeatLossModel tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.thermosmart.learning.contracts import Regime
from custom_components.thermosmart.learning.episode_schemas import (
    PassiveCoolingEpisode,
    Trajectory,
    TrajectoryPoint,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n == 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def cooling_episode(
    episode_id: str, loss_c_per_h: float, *, reliability: float = 0.8,
    duration_min: float = 30.0, points: int = 5, start_temp: float = 21.0,
    regime: Regime = Regime.PASSIVE_COOLING, confounder_flags=(),
    episode_schema_version: int = 1, start_offset_min: float = 0.0, zone: str = "lz",
) -> PassiveCoolingEpisode:
    """Build a PassiveCoolingEpisode whose trajectory yields ~loss_c_per_h."""
    drop = loss_c_per_h * (duration_min / 60.0)
    vals = _linspace(start_temp, start_temp - drop, points)  # may rise if loss<0
    step_ms = int(duration_min * 60_000 / (points - 1)) if points > 1 else 60_000
    traj = Trajectory(
        points=tuple(TrajectoryPoint(i * step_ms, float(v)) for i, v in enumerate(vals)),
        max_points=240)
    start = T0 + timedelta(minutes=start_offset_min)
    return PassiveCoolingEpisode(
        episode_id=episode_id, learning_zone_id=zone,
        episode_schema_version=episode_schema_version, builder_version=1,
        classifier_version=1, start_ts=start, end_ts=start + timedelta(minutes=duration_min),
        regime=regime, reliability=reliability, start_temp=start_temp,
        end_temp=vals[-1], trajectory=traj, confounder_flags=tuple(confounder_flags))
