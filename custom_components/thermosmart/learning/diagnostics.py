"""Engine-wide diagnostics orchestration for LE 2.0 (pure Python).

Merges per-model ``diagnostics()`` / ``confidence()`` output and injected typed
summaries into a privacy-safe, deterministic engine/zone/model diagnostics tree.
Read-only: never updates models, never touches storage / HA / coordinator. Builds
from sanitised aggregates (model diagnostics already carry no raw trajectories or
ids) and runs the privacy scanner as a second barrier.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from .export import (
    EngineSummary,
    EpisodeSummary,
    RegistrySummary,
    StorageSummary,
    ZoneExportInput,
    _confidence_dict,
    _sanitize_value,
)
from .privacy import PrivacyPolicy, Pseudonymizer, PseudonymNamespace, scan_payload

DIAGNOSTICS_SCHEMA_VERSION = 1


class ModelDiagnosticsStatus(Enum):
    OK = "ok"
    NOT_CONFIGURED = "not_configured"
    NO_EVIDENCE = "no_evidence"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelDiagnosticsEntry:
    name: str
    status: ModelDiagnosticsStatus
    confidence: Optional[float]
    aggregates: Mapping[str, Any]
    error: Optional[str] = None


@dataclass(frozen=True)
class ZoneDiagnostics:
    zone: str
    capability_flags: Mapping[str, bool]
    trv_only: bool
    initialized: bool
    missing_evidence: tuple[str, ...]
    model_availability: Mapping[str, bool]
    confidence_purposes: tuple[str, ...]
    models: tuple[ModelDiagnosticsEntry, ...]
    regime_summary: Optional[str] = None
    storage: Optional[Mapping[str, Any]] = None
    episodes: Optional[Mapping[str, int]] = None


@dataclass(frozen=True)
class DiagnosticsInput:
    engine: EngineSummary
    zones: tuple[ZoneExportInput, ...]
    pseudonymizer: Pseudonymizer
    generated_at: str
    policy: PrivacyPolicy = field(default_factory=PrivacyPolicy)
    registry: Optional[RegistrySummary] = None


@dataclass(frozen=True)
class DiagnosticsResult:
    payload: Mapping[str, Any]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


class DiagnosticsOrchestrator:
    def __init__(self, input: DiagnosticsInput) -> None:
        self._in = input
        self._warnings: list[str] = []
        self._errors: list[str] = []

    def build_diagnostics(self) -> DiagnosticsResult:
        self._warnings = []
        self._errors = []
        i = self._in
        zones = [self._zone(z) for z in i.zones]
        zones.sort(key=lambda d: d["zone"])
        payload = {
            "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "generated_at": i.generated_at,
            "engine": self._engine(),
            "zones": zones,
        }
        if i.registry is not None:
            payload["registry"] = {
                "raw_tracks": sorted(i.registry.raw_tracks),
                "episode_types": sorted(i.registry.episode_types),
                "models": sorted(i.registry.models),
                "graph_valid": i.registry.graph_valid, "issues": sorted(i.registry.issues)}
        for v in scan_payload(payload):
            self._errors.append(f"privacy:{v.kind}:{v.path}")
        return DiagnosticsResult(payload=payload, warnings=tuple(sorted(self._warnings)),
                                 errors=tuple(sorted(self._errors)))

    def _engine(self) -> dict:
        e = self._in.engine
        return {
            "le_version": e.le_version, "storage_schema_version": e.storage_schema_version,
            "initialized": e.initialized, "mode": e.mode, "registry_valid": e.registry_valid,
            "last_capture_ts": e.last_capture_ts, "last_model_update_ts": e.last_model_update_ts,
            "last_flush_ts": e.last_flush_ts, "warnings": sorted(e.warnings),
            "errors": sorted(e.errors),
        }

    def _zone(self, z: ZoneExportInput) -> dict:
        pz = self._in.pseudonymizer.pseudonymize(PseudonymNamespace.ZONE, z.zone_id)
        models = [self._model(pz, name, model)
                  for name, model in sorted(z.models.items(), key=lambda kv: kv[0])]
        out = {
            "zone": pz,
            "capability_flags": {k: bool(v) for k, v in sorted(z.capability_flags.items())},
            "trv_only": bool(z.trv_only), "initialized": bool(z.initialized),
            "missing_evidence": sorted(z.missing_evidence),
            "model_availability": {name: model is not None
                                   for name, model in sorted(z.models.items())},
            "confidence_purposes": sorted(str(p) for p in z.confidence_results),
            "confidence": sorted(
                ([_confidence_dict(r) for r in z.confidence_results.values() if r is not None]),
                key=lambda d: d.get("purpose", "")),
            "models": [self._model_dict(m) for m in models],
        }
        if z.regime_summary is not None:
            out["regime_summary"] = z.regime_summary
        if z.storage is not None:
            out["storage"] = {"initialized": z.storage.initialized,
                              "schema_version": z.storage.schema_version,
                              "record_counts": {k: int(v) for k, v in
                                                sorted(z.storage.record_counts.items())}}
        if z.episodes is not None:
            out["episodes"] = {k: int(v) for k, v in sorted(z.episodes.counts_by_type.items())}
        if z.pending_attribution is not None:
            # Expose counts only — no decision IDs, no entity IDs.
            pa = z.pending_attribution
            out["pending_attribution"] = {
                "schema_version": int(pa.get("schema_version", 1)),
                "pending_context_count": int(pa.get("pending_context_count", 0)),
                "pending_dispatch_count": int(pa.get("pending_dispatch_count", 0)),
                "outcome_eligible_count": int(pa.get("outcome_eligible_count", 0)),
                "dispatch_statuses": {str(k): int(v)
                                      for k, v in pa.get("dispatch_statuses", {}).items()},
                "control_types": sorted(str(c) for c in pa.get("control_types", [])),
                "total_targets": int(pa.get("total_targets", 0)),
                "failed_targets": int(pa.get("failed_targets", 0)),
                "pending_outcome_active": bool(pa.get("pending_outcome_active", False)),
            }
        return out

    def _model(self, pz: str, name: str, model: Any) -> ModelDiagnosticsEntry:
        if model is None:
            return ModelDiagnosticsEntry(name=name, status=ModelDiagnosticsStatus.NOT_CONFIGURED,
                                         confidence=None, aggregates={})
        try:
            diag = model.diagnostics()
            conf = model.confidence()
        except Exception as err:  # isolate one broken model
            self._errors.append(f"model_diagnostics_failed:{pz}/{name}:{type(err).__name__}")
            return ModelDiagnosticsEntry(name=name, status=ModelDiagnosticsStatus.FAILED,
                                         confidence=None, aggregates={},
                                         error=type(err).__name__)
        aggregates = self._sanitize_diag(diag)
        conf_value = getattr(conf, "value", None)
        evidence_count = getattr(conf, "evidence_count", None)
        status = ModelDiagnosticsStatus.OK
        if isinstance(evidence_count, (int, float)) and evidence_count == 0:
            status = ModelDiagnosticsStatus.NO_EVIDENCE
        return ModelDiagnosticsEntry(
            name=name, status=status,
            confidence=round(conf_value, 6) if isinstance(conf_value, (int, float)) else None,
            aggregates=aggregates)

    def _sanitize_diag(self, diag: Any) -> dict:
        if dataclasses.is_dataclass(diag) and not isinstance(diag, type):
            raw = dataclasses.asdict(diag)
        elif isinstance(diag, Mapping):
            raw = dict(diag)
        else:
            return {}
        clean: dict[str, Any] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                continue
            kl = k.lower()
            if any(bad in kl for bad in ("_id", "entity", "name", "trajectory")):
                continue
            clean[k] = _sanitize_value(v, lambda x: round(x, 6), 256)
        # drop non-finite leftovers and forbidden values via scan-aware redaction
        return clean

    @staticmethod
    def _model_dict(m: ModelDiagnosticsEntry) -> dict:
        return {"name": m.name, "status": m.status.value, "confidence": m.confidence,
                "aggregates": m.aggregates, "error": m.error}


def build_diagnostics(input: DiagnosticsInput) -> DiagnosticsResult:
    return DiagnosticsOrchestrator(input).build_diagnostics()
