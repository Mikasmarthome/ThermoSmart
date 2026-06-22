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

        boost_ctx = None
        if snapshot.decision_type is DecisionType.BOOST and snapshot.target_c is not None \
                and snapshot.trv_setpoint_c is not None:
            offset = max(0.0, snapshot.trv_setpoint_c - snapshot.target_c)
            if offset > 0.0:
                boost_ctx = BoostUpdateContext(
                    source_episode_id=snapshot.decision_id,  # bound at episode close
                    decision_id=snapshot.decision_id, requested_offset_c=offset,
                    start_deficit_c=None, target_temperature_c=snapshot.target_c,
                    controller_kind=None)

        return MaterializedEvidence(
            zone_id=snapshot.zone_id, decision_id=snapshot.decision_id, ts=snapshot.ts,
            manual_correction_contexts=tuple(mc_contexts), boost_context=boost_ctx,
            notes=tuple(notes))
