"""Shared builders for B2b-3 baseline-comparison tests."""
from __future__ import annotations

from custom_components.thermosmart.learning.models import NonBoostBaselineSample, BaselineQuery


def sample(did, onset_dur, *, zone="z", deficit=3.0, target=21.0, outdoor=5.0,
           heat_rate=2.0, tpi=0.5, hour=6, reached="2025-01-01T06:00:00+00:00",
           rel=0.9, sensor=0.9, device="setpoint", confounders=(),
           applied=0.0, authoritative=True, cmd_extra=300.0) -> NonBoostBaselineSample:
    return NonBoostBaselineSample(
        episode_id="e" + did, decision_id=did, learning_zone_id=zone,
        started_at="2025-01-01T05:30:00+00:00",
        effective_heating_onset_at="2025-01-01T05:35:00+00:00", target_reached_at=reached,
        command_to_target_duration_s=onset_dur + cmd_extra, onset_to_target_duration_s=onset_dur,
        start_deficit_c=deficit, target_temperature_c=target, start_temperature_c=target - deficit,
        external_temperature_c=outdoor, heat_rate_used=heat_rate, heat_loss_used=0.5,
        tpi_duty=tpi, baseline_setpoint_c=target, comfort_time_utc=None,
        time_of_day_bucket=hour, presence_context=None, device_control_type=device,
        dispatch_quality="full", sensor_reliability=sensor, confounder_flags=tuple(confounders),
        outcome_reliability=rel, boost_applied_c=applied, authoritative=authoritative)


def query(zone="z", *, deficit=3.0, target=21.0, outdoor=5.0, heat_rate=2.0, tpi=0.5,
          hour=6, before="2025-01-15T00:00:00+00:00", device="setpoint",
          decision_id="dQ", episode_id="eQ") -> BaselineQuery:
    return BaselineQuery(
        learning_zone_id=zone, decision_id=decision_id, episode_id=episode_id,
        baseline_cutoff_ts=before, start_deficit_c=deficit, target_temperature_c=target,
        external_temperature_c=outdoor, heat_rate_used=heat_rate, tpi_duty=tpi,
        time_of_day_bucket=hour, device_control_type=device)
