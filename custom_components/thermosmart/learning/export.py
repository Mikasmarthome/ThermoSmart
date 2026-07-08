"""Support / Learning export orchestration for Learning (pure Python).

Merges already-computed model exports, confidence results and injected summaries
into two typed, privacy-safe, deterministic, JSON-compatible payloads — a compact
Support Export and a pseudonymised, budgeted Learning Export. Builds strictly from
an explicit allowlist (never serialise raw model state), isolates per-component
failures, and runs a defensive privacy scanner as a second barrier.

Scope: queries already-instantiated models (read-only: ``export`` / ``confidence``
are pure) and consumes injected typed summaries. No model updates, no storage I/O,
no HA state, no coordinator, no file I/O.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from .contracts import ExportScope
from .privacy import (
    PRIVACY_POLICY_VERSION,
    PrivacyPolicy,
    Pseudonymizer,
    PseudonymNamespace,
    scan_payload,
)

EXPORT_SCHEMA_VERSION = 1

# user-facing names (internal scope stays ExportScope.SUPPORT / .RESEARCH)
SUPPORT_EXPORT_NAME = "Support Export"
LEARNING_EXPORT_NAME = "Learning Export"


class ExportComponentStatus(Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"
    NOT_INITIALIZED = "not_initialized"
    NO_EVIDENCE = "no_evidence"
    FAILED = "failed"
    OMITTED = "omitted"
    TRUNCATED = "truncated"


# -- allowlists ---------------------------------------------------------------

# Top-level schema keys permitted in a model's SUPPORT section. Anything else is
# dropped with a warning (allowlist, not blacklist).
SUPPORT_FIELD_ALLOWLIST = frozenset({
    "model_version", "parameter_version", "scoring_version", "evaluation_version",
    "sample_count", "confidence", "fallback", "buckets", "rejection_summary",
    # heat rate / loss
    "general_rate_c_per_h", "general_loss_c_per_h", "normalized_coefficient_per_h",
    "relative_only",
    # afterheat
    "general_rise_c", "general_time_to_peak_s", "general_duration_s", "general_decay",
    "direct_drive_end_count", "inferred_drive_end_count", "near_zero_count",
    # outcome
    "physical", "comfort", "controller", "data_quality", "full_partial", "reached_rate",
    "timeout_rate", "overshoot_rate", "reason_counts",
    # forecast
    "general_trust", "bias_c", "abs_error_c", "horizon_trust", "source_trust",
    # boost
    "recommended_offset_c", "recommended_duration_s", "gain_c", "overshoot_c",
    # manual correction
    "preference_bias_c", "warmer_count", "cooler_count", "direct_count", "automation_count",
    "unknown_count", "revert_rate", "intent_distribution",
})

LEARNING_SECTION_ALLOWLIST = frozenset({
    "samples", "model_version", "parameter_version", "scoring_version", "evaluation_version",
})

LEARNING_SAMPLE_FIELD_ALLOWLIST = frozenset({
    # outcome
    "physical", "comfort", "controller", "data_quality", "performance", "reason", "partial",
    "scoring_version",
    # forecast
    "temperature_error_c", "absolute_error_c", "horizon_bucket", "reality_quality",
    "model_version",
    # boost
    "requested_offset_c", "effective_factor_c", "temperature_gain_c", "overshoot_c",
    "boost_duration_s", "attribution_reliability", "evaluation_version",
    # manual correction
    "direction", "signed_magnitude_c", "source_class", "intent", "reverted",
    # thermal models (heat rate / loss / afterheat sample fields)
    "rate_c_per_h", "loss_c_per_h", "rise_c", "time_to_peak_s", "duration_s", "decay",
    "outdoor_bucket", "delta_c",
})

# learning-sample numeric quantisation by field-name suffix/name
_SCORE_FIELDS = frozenset({
    "physical", "comfort", "controller", "data_quality", "performance", "reality_quality",
    "attribution_reliability", "decay"})


# -- budget -------------------------------------------------------------------

@dataclass(frozen=True)
class ExportBudget:
    max_zones: int = 50
    max_models_per_zone: int = 16
    max_samples_per_model: int = 50
    max_total_bytes: int = 1_000_000
    max_string_len: int = 512
    max_depth: int = 14
    max_warnings: int = 300
    max_errors: int = 300
    budget_version: int = 1


# -- typed inputs -------------------------------------------------------------

@dataclass(frozen=True)
class EngineSummary:
    le_version: str
    storage_schema_version: Optional[int] = None
    initialized: bool = False
    mode: str = "inactive"            # inactive/capture-only/shadow/advisory/control
    registry_valid: Optional[bool] = None
    last_capture_ts: Optional[str] = None
    last_model_update_ts: Optional[str] = None
    last_flush_ts: Optional[str] = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StorageSummary:
    initialized: bool = False
    record_counts: Mapping[str, int] = field(default_factory=dict)
    schema_version: Optional[int] = None


@dataclass(frozen=True)
class EpisodeSummary:
    counts_by_type: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistrySummary:
    raw_tracks: tuple[str, ...] = ()
    episode_types: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    graph_valid: Optional[bool] = None
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ZoneExportInput:
    zone_id: str
    capability_flags: Mapping[str, bool] = field(default_factory=dict)
    trv_only: bool = True
    initialized: bool = True
    models: Mapping[str, Any] = field(default_factory=dict)
    confidence_results: Mapping[str, Any] = field(default_factory=dict)
    storage: Optional[StorageSummary] = None
    episodes: Optional[EpisodeSummary] = None
    regime_summary: Optional[str] = None
    missing_evidence: tuple[str, ...] = ()
    pending_attribution: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class ExportInput:
    engine: EngineSummary
    zones: tuple[ZoneExportInput, ...]
    pseudonymizer: Pseudonymizer
    generated_at: str
    policy: PrivacyPolicy = field(default_factory=PrivacyPolicy)
    budget: ExportBudget = field(default_factory=ExportBudget)
    registry: Optional[RegistrySummary] = None
    ha_version: Optional[str] = None
    integration_version: Optional[str] = None


@dataclass(frozen=True)
class ExportWarning:
    code: str
    component: str
    message: str


@dataclass(frozen=True)
class ExportError:
    code: str
    component: str
    message: str


@dataclass(frozen=True)
class ExportResult:
    payload: Mapping[str, Any]
    warnings: tuple[ExportWarning, ...]
    errors: tuple[ExportError, ...]
    truncated: bool


# -- orchestrator -------------------------------------------------------------

class ExportOrchestrator:
    def __init__(self, input: ExportInput) -> None:
        self._in = input
        self._warnings: list[ExportWarning] = []
        self._errors: list[ExportError] = []
        self._truncated = False

    # public API
    def build_support_export(self) -> ExportResult:
        return self._build(ExportScope.SUPPORT)

    def build_learning_export(self) -> ExportResult:
        return self._build(ExportScope.RESEARCH)

    def _build(self, scope: ExportScope) -> ExportResult:
        self._warnings = []
        self._errors = []
        self._truncated = False
        i = self._in
        zones_in = list(i.zones)[: i.budget.max_zones]
        if len(i.zones) > i.budget.max_zones:
            self._warn("budget_zones", "engine", "zone count truncated")
            self._truncated = True
        zone_sections = []
        for z in zones_in:
            zone_sections.append(self._zone_section(z, scope))
        zone_sections.sort(key=lambda s: s["zone"])

        payload = {
            "metadata": self._metadata(scope),
            "engine": self._engine_section(),
            "zones": zone_sections,
            "warnings": [self._warn_dict(w) for w in sorted(
                self._warnings, key=lambda w: (w.component, w.code, w.message))],
            "errors": [self._err_dict(e) for e in sorted(
                self._errors, key=lambda e: (e.component, e.code, e.message))],
        }
        if i.registry is not None:
            payload["registry"] = self._registry_section(i.registry)

        # second barrier: scan; forbidden findings are removed and flagged
        payload = self._enforce_size(payload)
        violations = scan_payload(payload)
        if violations:
            for v in violations[: i.budget.max_errors]:
                self._errors.append(ExportError("privacy_violation", v.path, v.kind))
            payload = _redact_paths(payload, {v.path for v in violations})
            payload["errors"] = [self._err_dict(e) for e in sorted(
                self._errors, key=lambda e: (e.component, e.code, e.message))]
        return ExportResult(payload=payload, warnings=tuple(self._warnings),
                            errors=tuple(self._errors), truncated=self._truncated)

    # -- sections -------------------------------------------------------

    def _metadata(self, scope: ExportScope) -> dict:
        i = self._in
        scope_name = SUPPORT_EXPORT_NAME if scope is ExportScope.SUPPORT else LEARNING_EXPORT_NAME
        return {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "privacy_policy_version": i.policy.version,
            "le_engine_version": i.engine.le_version,
            "scope": scope.value, "scope_name": scope_name,
            "generated_at": i.generated_at if scope is ExportScope.SUPPORT
            else _day_bucket(i.generated_at),
            "ha_version": i.ha_version, "integration_version": i.integration_version,
            "pseudonym_stable": i.pseudonymizer.stable,
        }

    def _engine_section(self) -> dict:
        e = self._in.engine
        return {
            "le_version": e.le_version, "storage_schema_version": e.storage_schema_version,
            "initialized": e.initialized, "mode": e.mode, "registry_valid": e.registry_valid,
            "last_capture_ts": e.last_capture_ts, "last_model_update_ts": e.last_model_update_ts,
            "last_flush_ts": e.last_flush_ts, "warnings": sorted(e.warnings),
            "errors": sorted(e.errors),
        }

    def _registry_section(self, r: RegistrySummary) -> dict:
        return {
            "raw_tracks": sorted(r.raw_tracks), "episode_types": sorted(r.episode_types),
            "models": sorted(r.models), "graph_valid": r.graph_valid,
            "issues": sorted(r.issues),
        }

    def _zone_section(self, z: ZoneExportInput, scope: ExportScope) -> dict:
        pz = self._in.pseudonymizer.pseudonymize(PseudonymNamespace.ZONE, z.zone_id)
        models_in = list(z.models.items())[: self._in.budget.max_models_per_zone]
        if len(z.models) > self._in.budget.max_models_per_zone:
            self._warn("budget_models", pz, "model count truncated")
            self._truncated = True
        model_sections = {}
        for name, model in sorted(models_in, key=lambda kv: kv[0]):
            model_sections[name] = self._model_section(pz, name, model, scope)
        section = {
            "zone": pz,
            "capability_flags": {k: bool(v) for k, v in sorted(z.capability_flags.items())},
            "trv_only": bool(z.trv_only), "initialized": bool(z.initialized),
            "missing_evidence": sorted(z.missing_evidence),
            "models": model_sections,
            "confidence": self._confidence_section(z.confidence_results),
        }
        if z.regime_summary is not None:
            section["regime_summary"] = z.regime_summary
        if z.storage is not None:
            section["storage"] = {"initialized": z.storage.initialized,
                                  "schema_version": z.storage.schema_version,
                                  "record_counts": {k: int(v) for k, v in
                                                    sorted(z.storage.record_counts.items())}}
        if z.episodes is not None:
            section["episodes"] = {k: int(v) for k, v in sorted(z.episodes.counts_by_type.items())}
        return section

    def _model_section(self, pz: str, name: str, model: Any, scope: ExportScope) -> dict:
        if model is None:
            return {"status": ExportComponentStatus.NOT_CONFIGURED.value}
        try:
            raw = model.export(scope)
        except Exception as err:  # isolate a single broken model
            self._errors.append(ExportError("model_export_failed", f"{pz}/{name}",
                                            type(err).__name__))
            return {"status": ExportComponentStatus.FAILED.value}
        if not isinstance(raw, Mapping):
            self._errors.append(ExportError("model_export_invalid", f"{pz}/{name}",
                                            "export did not return a mapping"))
            return {"status": ExportComponentStatus.FAILED.value}
        if "unsupported" in raw:
            return {"status": ExportComponentStatus.OMITTED.value}
        if scope is ExportScope.SUPPORT:
            data = self._sanitize_support(pz, name, raw)
        else:
            data = self._sanitize_learning(pz, name, raw)
        status = ExportComponentStatus.OK.value
        if raw.get("sample_count", 1) == 0 and scope is ExportScope.SUPPORT:
            status = ExportComponentStatus.NO_EVIDENCE.value
        data["status"] = status
        return data

    def _sanitize_support(self, pz: str, name: str, raw: Mapping[str, Any]) -> dict:
        out: dict[str, Any] = {}
        digits = self._in.policy.quantization.support_round_digits
        for k, v in raw.items():
            if k not in SUPPORT_FIELD_ALLOWLIST:
                self._warn("dropped_field", f"{pz}/{name}", f"non-allowlisted field '{k}'")
                continue
            out[k] = _sanitize_value(v, lambda x: round(x, digits), self._in.budget.max_string_len)
        return out

    def _sanitize_learning(self, pz: str, name: str, raw: Mapping[str, Any]) -> dict:
        out: dict[str, Any] = {}
        q = self._in.policy.quantization
        for k, v in raw.items():
            if k not in LEARNING_SECTION_ALLOWLIST:
                self._warn("dropped_field", f"{pz}/{name}", f"non-allowlisted section '{k}'")
                continue
            if k == "samples" and isinstance(v, (list, tuple)):
                samples = list(v)
                capped = samples[: self._in.budget.max_samples_per_model]
                if len(samples) > len(capped):
                    self._warn("budget_samples", f"{pz}/{name}", "samples truncated")
                    self._truncated = True
                out["samples"] = [self._sanitize_sample(pz, name, s) for s in capped]
            else:
                out[k] = _sanitize_value(v, lambda x: x, self._in.budget.max_string_len)
        return out

    def _sanitize_sample(self, pz: str, name: str, sample: Any) -> dict:
        if not isinstance(sample, Mapping):
            return {}
        q = self._in.policy.quantization
        out: dict[str, Any] = {}
        for k, v in sorted(sample.items()):
            if k not in LEARNING_SAMPLE_FIELD_ALLOWLIST:
                self._warn("dropped_field", f"{pz}/{name}/sample",
                           f"non-allowlisted sample field '{k}'")
                continue
            if isinstance(v, bool) or v is None or isinstance(v, str):
                out[k] = v if not isinstance(v, str) else v[: self._in.budget.max_string_len]
            elif isinstance(v, (int, float)) and math.isfinite(v):
                out[k] = self._quantize_field(k, float(v), q)
            # silently skip non-finite (never emit NaN/Inf)
            elif isinstance(v, (int, float)):
                self._warn("non_finite_dropped", f"{pz}/{name}/sample", k)
        return out

    def _quantize_field(self, key: str, value: float, q) -> float:
        if key in _SCORE_FIELDS:
            return q.quantize(value, q.score_step)
        if key.endswith("_c_per_h"):
            return q.quantize(value, q.rate_step_c_per_h)
        if key.endswith("_s"):
            return q.quantize(value, q.duration_step_s)
        if key.endswith("_c") or key.endswith("_error_c") or key == "delta_c":
            return q.quantize(value, q.temperature_step_c)
        return q.quantize(value, q.score_step)

    def _confidence_section(self, results: Mapping[str, Any]) -> list:
        out = []
        for purpose, res in results.items():
            if res is None:
                out.append({"purpose": str(purpose), "status": "missing"})
                continue
            out.append(_confidence_dict(res))
        out.sort(key=lambda d: d.get("purpose", ""))
        return out

    # -- budgets / helpers ----------------------------------------------

    def _enforce_size(self, payload: dict) -> dict:
        try:
            blob = to_canonical_json(payload)
        except ValueError:
            return payload
        if len(blob.encode("utf-8")) > self._in.budget.max_total_bytes:
            self._warn("budget_size", "engine", "payload exceeded size budget")
            self._truncated = True
        return payload

    def _warn(self, code: str, component: str, message: str) -> None:
        if len(self._warnings) < self._in.budget.max_warnings:
            self._warnings.append(ExportWarning(code, component, message))

    @staticmethod
    def _warn_dict(w: ExportWarning) -> dict:
        return {"code": w.code, "component": w.component, "message": w.message}

    @staticmethod
    def _err_dict(e: ExportError) -> dict:
        return {"code": e.code, "component": e.component, "message": e.message}


# -- module-level convenience -------------------------------------------------

def build_support_export(input: ExportInput) -> ExportResult:
    return ExportOrchestrator(input).build_support_export()


def build_learning_export(input: ExportInput) -> ExportResult:
    return ExportOrchestrator(input).build_learning_export()


# -- confidence serialisation -------------------------------------------------

def _confidence_dict(res: Any) -> dict:
    """Normalise a ConfidenceResult — no recomputation, no source references."""
    caps = [{"cap": c.cap.value, "max_value": round(c.max_value, 6), "reason": c.reason,
             "hard": c.hard, "priority": c.priority}
            for c in getattr(res, "applied_caps", ())]
    caps.sort(key=lambda d: (d["priority"], d["cap"]))
    gate = getattr(res, "gate", None)
    return {
        "purpose": res.purpose.value, "value": round(res.value, 6), "level": res.level.value,
        "status": res.status.value, "reliability": round(res.reliability, 6),
        "applied_caps": caps,
        "missing_required": sorted(res.missing_required),
        "missing_optional": sorted(res.missing_optional),
        "reason_codes": sorted(res.reason_codes),
        "control_allowed": res.control_allowed, "advisory_allowed": res.advisory_allowed,
        "gate": None if gate is None else {
            "allowed": gate.allowed, "required_minimum": round(gate.required_minimum, 6),
            "safe_fallback_required": gate.safe_fallback_required},
        "aggregator_version": res.aggregator_version,
        "parameter_version": res.parameter_version,
    }


# -- value sanitisation / serialisation ---------------------------------------

def _sanitize_value(value: Any, round_num, max_str: int, depth: int = 0) -> Any:
    if depth > 30:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value[:max_str]
    if isinstance(value, (int,)):
        return value
    if isinstance(value, float):
        return round_num(value) if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _sanitize_value(v, round_num, max_str, depth + 1)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v, round_num, max_str, depth + 1) for v in value]
    return str(value)[:max_str]


def _day_bucket(generated_at_iso: str) -> str:
    try:
        from datetime import datetime
        return datetime.fromisoformat(generated_at_iso).date().isoformat()
    except (TypeError, ValueError):
        return "unknown"


def _redact_paths(payload: Any, paths: set, current: str = "$") -> Any:
    if current in paths:
        return "<redacted>"
    if isinstance(payload, Mapping):
        return {k: _redact_paths(v, paths, f"{current}.{k}") for k, v in payload.items()}
    if isinstance(payload, list):
        return [_redact_paths(v, paths, f"{current}[{i}]") for i, v in enumerate(payload)]
    return payload


# -- validation / canonical json ----------------------------------------------

def validate_export_payload(payload: Mapping[str, Any],
                            budget: Optional[ExportBudget] = None) -> list[str]:
    budget = budget or ExportBudget()
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["payload is not a mapping"]
    meta = payload.get("metadata")
    if not isinstance(meta, Mapping):
        errors.append("missing metadata")
    else:
        for k in ("export_schema_version", "privacy_policy_version", "le_engine_version",
                  "scope"):
            if k not in meta:
                errors.append(f"metadata missing '{k}'")
    errors.extend(_validate_depth_finite(payload, budget))
    for v in scan_payload(payload):
        errors.append(f"privacy: {v.kind} at {v.path}")
    try:
        blob = to_canonical_json(payload)
        if len(blob.encode("utf-8")) > budget.max_total_bytes:
            errors.append("payload exceeds size budget")
    except ValueError as err:
        errors.append(f"not serialisable: {err}")
    return errors


def _validate_depth_finite(value: Any, budget: ExportBudget, depth: int = 0) -> list[str]:
    errors: list[str] = []
    if depth > budget.max_depth:
        return [f"max depth {budget.max_depth} exceeded"]
    if isinstance(value, float) and not math.isfinite(value):
        return ["non-finite numeric value"]
    if isinstance(value, Mapping):
        for k, v in value.items():
            errors.extend(_validate_depth_finite(v, budget, depth + 1))
    elif isinstance(value, (list, tuple)):
        for v in value:
            errors.extend(_validate_depth_finite(v, budget, depth + 1))
    return errors


def to_canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)
