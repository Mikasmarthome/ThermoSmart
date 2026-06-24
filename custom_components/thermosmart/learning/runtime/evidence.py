"""Evidence materialisation for LE 2.0 shadow runtime (pure Python).

Turns captured snapshots/decisions into reproducible, persistable typed update
contexts BEFORE any model update — so updates never depend on transient values.
The foundation materialises the event-driven contexts that are fully derivable
from a single cycle (manual correction, boost decision); the trajectory-based
model contexts are produced by the episode pipeline as it closes episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..contracts import DataQuality
from ..models import BoostUpdateContext, ManualCorrectionUpdateContext
from .capture import CapturedSnapshot, DecisionType


@dataclass(frozen=True)
class EvidenceItem:
    model_name: str
    episode: Any            # the episode/event the model.update consumes
    context: Any            # the typed update context


@dataclass(frozen=True)
class MaterializedEvidence:
    zone_id: str
    decision_id: str
    ts: str
    manual_correction_contexts: tuple[ManualCorrectionUpdateContext, ...] = ()
    boost_context: Optional[BoostUpdateContext] = None
    items: tuple[EvidenceItem, ...] = ()
    notes: tuple[str, ...] = ()


class EvidenceMaterializer:
    """Pure: snapshot -> typed reproducible update contexts."""

    def materialize(self, snapshot: CapturedSnapshot) -> MaterializedEvidence:
        notes: list[str] = []

        mc_contexts: list[ManualCorrectionUpdateContext] = []
        for mc in snapshot.accepted_manual_corrections:
            mc_contexts.append(ManualCorrectionUpdateContext(
                correction_event_id=mc.event_id, learning_zone_id=None,
                decision_id=snapshot.decision_id,
                previous_thermosmart_target_c=mc.prior_target_c,
                target_temperature_c=snapshot.target_c,
                current_temperature_c=snapshot.indoor_temp.value
                if snapshot.indoor_temp is not None else None,
                decision_type=snapshot.decision_type.value,
                in_boost=snapshot.boost_active, in_preheat=snapshot.preheat_active,
                source_class_hint=None, context_quality=1.0))
        if snapshot.rejected_echo_corrections:
            notes.append(f"echo_filtered:{len(snapshot.rejected_echo_corrections)}")

        # B2b-1: the authoritative boost context comes from the bound LiveDecisionRecord
        # dispatch result, NOT the setpoint-target heuristic. Dispatch status gates whether
        # any boost outcome may form at all.
        boost_ctx = None
        bd = snapshot.boost_dispatch
        if bd is not None:
            applied = bd.boost_applied_c
            if bd.dispatch_status == "failed":
                notes.append("boost_dispatch_failed")            # no normal outcome
            elif bd.dispatch_status == "not_attempted":
                notes.append("boost_not_attempted")              # no dispatch-effect outcome
            elif applied is None or applied <= 0.0:
                notes.append("boost_zero_applied")               # candidate but nothing written
            elif bd.dispatch_status in ("fully_succeeded", "partially_succeeded"):
                partial = bd.dispatch_status == "partially_succeeded"
                boost_ctx = BoostUpdateContext(
                    source_episode_id=snapshot.decision_id,  # bound at episode close
                    decision_id=snapshot.decision_id, learning_zone_id=snapshot.zone_id,
                    authority="live_record",
                    requested_offset_c=float(applied), boost_applied_c=float(applied),
                    start_deficit_c=bd.start_deficit_c,
                    expected_non_boost_duration_s=bd.expected_non_boost_duration_s,
                    target_temperature_c=snapshot.target_c,
                    baseline_setpoint_c=bd.baseline_setpoint_c,
                    final_setpoint_c=bd.final_setpoint_c,
                    effective_setpoint_min_c=bd.effective_setpoint_min_c,
                    effective_setpoint_max_c=bd.effective_setpoint_max_c,
                    dispatch_status=bd.dispatch_status,
                    outcome_reliability=("partial" if partial else "full"),
                    context_quality=DataQuality.OK,
                    controller_kind=None)
                notes.append("boost_authoritative" + (":partial" if partial else ""))
        elif snapshot.decision_type is DecisionType.BOOST and snapshot.target_c is not None \
                and snapshot.trv_setpoint_c is not None:
            # LEGACY/DIAGNOSTIC fallback only when no authoritative record exists. Clearly
            # marked, reduced reliability — never treated as a full applied-boost outcome.
            offset = max(0.0, snapshot.trv_setpoint_c - snapshot.target_c)
            if offset > 0.0:
                boost_ctx = BoostUpdateContext(
                    source_episode_id=snapshot.decision_id,
                    decision_id=snapshot.decision_id, learning_zone_id=snapshot.zone_id,
                    authority="legacy_fallback",
                    requested_offset_c=offset, boost_applied_c=None,
                    start_deficit_c=None, target_temperature_c=snapshot.target_c,
                    outcome_reliability="partial", context_quality=DataQuality.STALE,
                    controller_kind=None)
                notes.append("boost_legacy_fallback")

        return MaterializedEvidence(
            zone_id=snapshot.zone_id, decision_id=snapshot.decision_id, ts=snapshot.ts,
            manual_correction_contexts=tuple(mc_contexts), boost_context=boost_ctx,
            notes=tuple(notes))
