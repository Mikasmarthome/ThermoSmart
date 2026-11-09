"""Evidence materialisation for Learning shadow runtime (pure Python).

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
from ..models import BoostUpdateContext
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
    boost_context: Optional[BoostUpdateContext] = None
    items: tuple[EvidenceItem, ...] = ()
    notes: tuple[str, ...] = ()


class EvidenceMaterializer:
    """Pure: snapshot -> typed reproducible update contexts."""

    def materialize(self, snapshot: CapturedSnapshot) -> MaterializedEvidence:
        notes: list[str] = []


        # B2b-1: the authoritative boost context comes from the bound LiveDecisionRecord
        # dispatch result, NOT the setpoint-target heuristic. Dispatch status gates whether
        # any boost outcome may form at all.
        boost_ctx = None
        bd = snapshot.boost_dispatch
        if bd is not None:
            # F1: adaptive learning must use the POST-dispatch CONFIRMED effective
            # offset (effective_offset_c), NOT the pre-dispatch APPROVED offset
            # (boost_applied_c). effective_offset_c is None only on older records;
            # fall back to the approved offset then (no fabricated confirmation).
            approved = bd.boost_applied_c
            effective = getattr(bd, "effective_offset_c", None)
            learn_offset = effective if effective is not None else approved
            if bd.dispatch_status == "failed":
                notes.append("boost_dispatch_failed")            # no normal outcome
            elif bd.dispatch_status == "not_attempted":
                notes.append("boost_not_attempted")              # no dispatch-effect outcome
            elif bd.dispatch_status == "already_satisfied":
                # D5a: the device already held the requested (boosted) setpoint — the
                # control state is satisfied, but NO new service call was issued, so
                # this cycle produces NO fresh actuation evidence. Form no boost
                # outcome (conservative): a held setpoint must never, on its own,
                # create a new successful-dispatch learning sample. The boost offset
                # IS reflected in applied_boost_offset_c for attribution/diagnostics.
                notes.append("boost_held_no_new_dispatch")
            elif learn_offset is None or learn_offset <= 0.0:
                # Nothing was EFFECTIVELY applied (e.g. approved but clamped to
                # baseline, or zero) — no boost outcome from this cycle.
                notes.append("boost_zero_applied")
            elif bd.dispatch_status in ("fully_succeeded", "partially_succeeded"):
                partial = bd.dispatch_status == "partially_succeeded"
                boost_ctx = BoostUpdateContext(
                    source_episode_id=snapshot.decision_id,  # bound at episode close
                    decision_id=snapshot.decision_id, learning_zone_id=snapshot.zone_id,
                    authority="live_record",
                    requested_offset_c=float(learn_offset), boost_applied_c=float(learn_offset),
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
        elif snapshot.decision_type is DecisionType.BOOST:
            # M2: DecisionType.BOOST is the controller's DECISION to boost — it is NOT
            # proof that a boost was dispatched, succeeded, or is physically in effect.
            # The only authoritative evidence of an applied boost is the LiveDecisionRecord
            # dispatch result (``boost_dispatch`` above). When that record is absent we
            # have no dispatch status at all, so we CANNOT tell a successful boost from a
            # failed / not-attempted / merely-requested one. The former setpoint-heuristic
            # fallback (trv_setpoint > target) would form a boost sample here regardless —
            # and D4 made this branch reachable — so a failed or not-attempted boost could
            # produce learning evidence through it. Form NO boost context: with no
            # authoritative dispatch record, no boost outcome is attributed.
            notes.append("boost_unconfirmed_no_dispatch_record")

        return MaterializedEvidence(
            zone_id=snapshot.zone_id, decision_id=snapshot.decision_id, ts=snapshot.ts,
            boost_context=boost_ctx,
            notes=tuple(notes))
