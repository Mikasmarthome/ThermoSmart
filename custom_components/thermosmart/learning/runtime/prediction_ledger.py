"""Prediction snapshot ledger for LE 2.0 shadow runtime (pure Python).

Records the predictions that were actually available at a controller/shadow
decision, so later outcomes are evaluated against the prediction that existed
THEN — never recomputed with a later-improved model. Bounded, deduplicated,
restart-surviving. No HA, no I/O. IDs are stripped/pseudonymised by the export
layer; this ledger holds the authoritative internal record.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PredictionSnapshot:
    decision_id: str
    learning_zone_id: str
    prediction_type: str
    value: float
    unit: str
    confidence: float
    reliability: float
    fallback_used: bool
    prior_contribution: float
    learned_contribution: float
    bucket: Optional[str]
    model_version: int
    parameter_version: int
    created_ts: str
    target_ts: Optional[str] = None

    def key(self) -> tuple[str, str]:
        return (self.decision_id, self.prediction_type)


class PredictionSnapshotLedger:
    def __init__(self, max_entries: int = 500) -> None:
        self._max = max_entries
        self._snapshots: dict[tuple[str, str], PredictionSnapshot] = {}

    def record(self, snapshot: PredictionSnapshot) -> None:
        self._snapshots[snapshot.key()] = snapshot  # dedup by (decision, type)
        if len(self._snapshots) > self._max:
            ordered = sorted(self._snapshots.values(),
                             key=lambda s: (s.created_ts, s.decision_id, s.prediction_type))
            for victim in ordered[: len(self._snapshots) - self._max]:
                self._snapshots.pop(victim.key(), None)

    def record_shadow_prediction(self, sp: Any) -> None:
        if not sp.values:
            return
        key, value = next(iter(sp.values.items()))
        self.record(PredictionSnapshot(
            decision_id=sp.decision_id, learning_zone_id="", prediction_type=sp.prediction_type,
            value=float(value), unit=sp.units.get(key, ""), confidence=sp.confidence,
            reliability=sp.reliability, fallback_used=sp.fallback_used,
            prior_contribution=sp.prior_contribution,
            learned_contribution=sp.learned_contribution, bucket=None,
            model_version=sp.model_version, parameter_version=sp.parameter_version,
            created_ts=getattr(sp, "_created_ts", ""), target_ts=None))

    def get(self, decision_id: str, prediction_type: str) -> Optional[PredictionSnapshot]:
        return self._snapshots.get((decision_id, prediction_type))

    def for_decision(self, decision_id: str) -> tuple[PredictionSnapshot, ...]:
        return tuple(s for s in self._snapshots.values() if s.decision_id == decision_id)

    def prune_decisions(self, keep_decision_ids: Sequence[str]) -> int:
        keep = set(keep_decision_ids)
        removed = 0
        for k, s in list(self._snapshots.items()):
            if s.decision_id not in keep:
                self._snapshots.pop(k, None)
                removed += 1
        return removed

    @property
    def size(self) -> int:
        return len(self._snapshots)

    def serialize(self) -> list:
        return [_to_dict(s) for s in sorted(
            self._snapshots.values(),
            key=lambda s: (s.created_ts, s.decision_id, s.prediction_type))]

    def restore(self, data: Sequence[Mapping]) -> None:
        self._snapshots = {}
        for d in data:
            self.record(_from_dict(d))


def _to_dict(s: PredictionSnapshot) -> dict:
    return {
        "decision_id": s.decision_id, "learning_zone_id": s.learning_zone_id,
        "prediction_type": s.prediction_type, "value": s.value, "unit": s.unit,
        "confidence": s.confidence, "reliability": s.reliability,
        "fallback_used": s.fallback_used, "prior_contribution": s.prior_contribution,
        "learned_contribution": s.learned_contribution, "bucket": s.bucket,
        "model_version": s.model_version, "parameter_version": s.parameter_version,
        "created_ts": s.created_ts, "target_ts": s.target_ts,
    }


def _from_dict(d: Mapping) -> PredictionSnapshot:
    return PredictionSnapshot(
        decision_id=d["decision_id"], learning_zone_id=d.get("learning_zone_id", ""),
        prediction_type=d["prediction_type"], value=d["value"], unit=d.get("unit", ""),
        confidence=d["confidence"], reliability=d["reliability"],
        fallback_used=d["fallback_used"], prior_contribution=d["prior_contribution"],
        learned_contribution=d["learned_contribution"], bucket=d.get("bucket"),
        model_version=d["model_version"], parameter_version=d["parameter_version"],
        created_ts=d.get("created_ts", ""), target_ts=d.get("target_ts"))
