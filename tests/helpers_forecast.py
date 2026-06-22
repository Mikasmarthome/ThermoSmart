"""Shared builders for LE 2.0 ForecastModel tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from custom_components.thermosmart.learning.raw_schemas import (
    ForecastDecisionEvent,
    ForecastEvaluationEvent,
)

T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)


def decision(decision_id: str, forecast_high: float, *, created_offset_h: float = 0.0,
             zone: str = "lz", target: float = 21.0, suppression: float = 0.0,
             current_temp: Optional[float] = None) -> ForecastDecisionEvent:
    return ForecastDecisionEvent(
        learning_zone_id=zone, ts=T0 + timedelta(hours=created_offset_h),
        decision_id=decision_id, decision_target=target, raw_target=target,
        forecast_high=forecast_high, suppression=suppression,
        current_temp_at_decision=current_temp)


def evaluation(evaluation_id: str, decision_id: str, observed: float, *,
               created_offset_h: float = 0.0, horizon_h: float = 6.0,
               zone: str = "lz") -> ForecastEvaluationEvent:
    return ForecastEvaluationEvent(
        learning_zone_id=zone, ts=T0 + timedelta(hours=created_offset_h + horizon_h),
        evaluation_id=evaluation_id, decision_id=decision_id, observed_temp_at_eval=observed)


def pair(model, idx, *, forecast=10.0, observed=10.0, horizon_h=6.0, zone="lz"):
    """Register a decision and apply its evaluation; return the update result."""
    off = idx * 24.0
    d = decision(f"d{idx}", forecast, created_offset_h=off, zone=zone)
    model.register_decision(d)
    return model.update(evaluation(f"e{idx}", f"d{idx}", observed,
                                   created_offset_h=off, horizon_h=horizon_h, zone=zone))
