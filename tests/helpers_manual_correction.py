"""Shared builders for LE 2.0 ManualCorrectionModel tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from custom_components.thermosmart.learning.raw_schemas import (
    CorrectionActionType,
    CorrectionSource,
    ManualCorrectionEvent,
)
from custom_components.thermosmart.learning.models import ManualCorrectionUpdateContext

T0 = datetime(2025, 1, 1, 18, 0, 0, tzinfo=timezone.utc)


def correction_event(event_id: str, prior: float, new: float, *,
                     source: CorrectionSource = CorrectionSource.UI,
                     action: CorrectionActionType = CorrectionActionType.SETPOINT_CHANGE,
                     day: float = 0.0, hour_offset: float = 0.0,
                     zone: str = "lz") -> ManualCorrectionEvent:
    return ManualCorrectionEvent(
        learning_zone_id=zone, ts=T0 + timedelta(days=day, hours=hour_offset),
        event_id=event_id, action_type=action, prior_target=prior, new_target=new,
        source=source)


def correction_context(event, *, decision_id=None, time_since_decision_s=600.0,
                       decision_type=None, schedule_period=None, in_boost=False,
                       in_preheat=False, current_temperature_c=None, outcome_overshoot_c=None,
                       outcome_undershoot_c=None, physical_trv=False, is_internal_echo=False,
                       is_device_sync=False, last_sent_setpoint_c=None, reverted_within_s=None,
                       attribution_reliability=None, source_class_hint=None,
                       zone=None) -> ManualCorrectionUpdateContext:
    return ManualCorrectionUpdateContext(
        correction_event_id=event.event_id, learning_zone_id=zone, decision_id=decision_id,
        time_since_decision_s=time_since_decision_s, decision_type=decision_type,
        schedule_period=schedule_period, in_boost=in_boost, in_preheat=in_preheat,
        current_temperature_c=current_temperature_c, outcome_overshoot_c=outcome_overshoot_c,
        outcome_undershoot_c=outcome_undershoot_c, physical_trv=physical_trv,
        is_internal_echo=is_internal_echo, is_device_sync=is_device_sync,
        last_sent_setpoint_c=last_sent_setpoint_c, reverted_within_s=reverted_within_s,
        attribution_reliability=attribution_reliability, source_class_hint=source_class_hint)


def recurring_direct(model, n=6, *, prior=21.0, new=21.3, source=CorrectionSource.UI):
    """Apply n consistent direct user corrections on distinct days."""
    for d in range(n):
        ev = correction_event(f"d{d}", prior, new, source=source, day=d)
        model.update(ev, correction_context(ev))
