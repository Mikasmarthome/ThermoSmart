"""LearningRuntime: passive shadow runtime foundation for LE 2.0 (pure core).

Ties capture -> evidence -> model orchestration -> shadow -> persistence into a
deterministic per-cycle pipeline across zones. STRICTLY passive: it returns only
typed diagnostics/shadow results and can never emit a control command. Any
failure is isolated per zone and surfaced as a structured RuntimeError —
"learning failure must never become heating failure". Async lifecycle methods use
an injected store + clock; there are no HA imports here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Optional

from ..confidence import ConfidencePurpose
from ..contracts import PredictionType
from .capture import CaptureCoordinator, RuntimeCycleInput
from .evidence import EvidenceMaterializer
from .orchestration import ModelOrchestrator
from .persistence import PersistenceOrchestrator, SavePolicy, SaveTrigger
from .pipeline import RuntimePipeline
from .prediction_ledger import PredictionSnapshot, PredictionSnapshotLedger
from .shadow import (
    ComparisonType,
    PreheatParameters,
    ShadowOrchestrator,
)

RUNTIME_SCHEMA_VERSION = 1


class LearningRuntimeMode(Enum):
    INACTIVE = "inactive"
    CAPTURE_ONLY = "capture_only"
    SHADOW = "shadow"


@dataclass(frozen=True)
class LearningRuntimeConfig:
    le_version: str = "2.0.0"
    mode: LearningRuntimeMode = LearningRuntimeMode.SHADOW
    save_policy: SavePolicy = field(default_factory=SavePolicy)
    preheat_params: PreheatParameters = field(default_factory=PreheatParameters)
    startup_grace_cycles: int = 1
    max_open_comparisons: int = 200


@dataclass(frozen=True)
class RuntimeWarning:
    code: str
    zone: str
    message: str


@dataclass(frozen=True)
class RuntimeError:
    code: str
    zone: str
    message: str


@dataclass(frozen=True)
class RuntimeCycleResult:
    zone_id: str
    ts: str
    mode: str
    decision_id: Optional[str]
    captured: bool
    shadow_predictions: tuple = ()
    preheat_plan: Optional[Any] = None
    comparisons: tuple = ()
    confidence_results: Mapping[str, Any] = field(default_factory=dict)
    model_errors: tuple[str, ...] = ()
    warnings: tuple[RuntimeWarning, ...] = ()
    errors: tuple[RuntimeError, ...] = ()
    dirty: bool = False
    # structural no-control marker: a coordinator must never read a command here
    has_control_effect: bool = False

    @property
    def control_commands(self) -> tuple:
        """There are never any control commands in shadow mode."""
        return ()

    def __post_init__(self) -> None:
        if self.has_control_effect:
            raise ValueError("RuntimeCycleResult must never carry a control effect")


@dataclass(frozen=True)
class RuntimeHealth:
    mode: str
    zones: int
    initialized: bool
    dirty: bool
    last_cycle_ts: Optional[str]
    last_save: Optional[str]
    open_decisions: int
    open_comparisons: int
    model_errors: int
    storage_warnings: int
    model_update_total: int = 0
    prediction_snapshots: int = 0


class _ZoneRuntime:
    def __init__(self, zone_id: str, config: LearningRuntimeConfig) -> None:
        self.zone_id = zone_id
        self.capture = CaptureCoordinator(zone_id)
        self.evidence = EvidenceMaterializer()
        self.pipeline = RuntimePipeline(zone_id)
        self.orchestrator = ModelOrchestrator(zone_id)
        self.shadow = ShadowOrchestrator(preheat_params=config.preheat_params,
                                         max_open_comparisons=config.max_open_comparisons)
        self.ledger = PredictionSnapshotLedger()
        self.open_comparisons: list = []
        self.outcomes: list = []
        self.model_update_counts: dict[str, int] = {}
        self.cycles = 0
        self.last_cycle_ts: Optional[str] = None

    def serialize(self) -> dict:
        return {"capture": self.capture.serialize(),
                "pipeline": self.pipeline.serialize(),
                "models": self.orchestrator.serialize_models(),
                "ledger": self.ledger.serialize(),
                "model_update_counts": dict(self.model_update_counts),
                "cycles": self.cycles, "last_cycle_ts": self.last_cycle_ts}

    def restore(self, data: Mapping[str, Any]) -> tuple[str, ...]:
        errors: list[str] = []
        if "capture" in data:
            try:
                self.capture.restore(data["capture"])
            except Exception as err:
                errors.append(f"capture:restore:{type(err).__name__}")
        if "pipeline" in data:
            errors += tuple(self.pipeline.restore(data["pipeline"]))
        if "models" in data:
            errors.extend(self.orchestrator.restore_models(data["models"]))
        if "ledger" in data:
            try:
                self.ledger.restore(data["ledger"])
            except Exception as err:
                errors.append(f"ledger:restore:{type(err).__name__}")
        self.model_update_counts = dict(data.get("model_update_counts", {}))
        self.cycles = data.get("cycles", 0)
        self.last_cycle_ts = data.get("last_cycle_ts")
        return tuple(errors)


class LearningRuntime:
    """Multi-zone passive runtime. Never mutates the existing controller."""

    def __init__(self, config: Optional[LearningRuntimeConfig] = None, *,
                 store: Optional[Any] = None,
                 clock: Optional[Callable[[], str]] = None) -> None:
        self._config = config or LearningRuntimeConfig()
        self._zones: dict[str, _ZoneRuntime] = {}
        self._store = store
        self._persistence = PersistenceOrchestrator(store, policy=self._config.save_policy) \
            if store is not None else None
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._initialized = False
        self._last_cycle_ts: Optional[str] = None

    @property
    def mode(self) -> LearningRuntimeMode:
        return self._config.mode

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _zone(self, zone_id: str) -> _ZoneRuntime:
        zr = self._zones.get(zone_id)
        if zr is None:
            zr = _ZoneRuntime(zone_id, self._config)
            self._zones[zone_id] = zr
        return zr

    # -- core cycle (pure-ish; mutates only LE2 state) ------------------

    def run_cycle(self, inp: RuntimeCycleInput) -> RuntimeCycleResult:
        mode = self._config.mode
        if mode is LearningRuntimeMode.INACTIVE:
            return RuntimeCycleResult(zone_id=inp.zone_id, ts=inp.ts, mode=mode.value,
                                      decision_id=None, captured=False)
        zr = self._zone(inp.zone_id)
        warnings: list[RuntimeWarning] = []
        snapshot = zr.capture.capture(inp)
        zr.cycles += 1
        zr.last_cycle_ts = inp.ts
        self._last_cycle_ts = inp.ts

        evidence = zr.evidence.materialize(snapshot)
        authoritative_change = False
        startup_ok = zr.cycles > self._config.startup_grace_cycles

        # cross-cycle episode pipeline -> completed authoritative episodes -> updates.
        # Transaction: completed episodes are fully materialised by the builders
        # before any model update; each update is eligibility-gated and isolated.
        pipe_result = zr.pipeline.process(snapshot)
        if startup_ok and not pipe_result.disturbed and not pipe_result.deduplicated:
            for ce in pipe_result.completed:
                ok = zr.orchestrator.apply_update(ce.model_name, ce.episode)
                if ok:
                    authoritative_change = True
                    zr.model_update_counts[ce.model_name] = \
                        zr.model_update_counts.get(ce.model_name, 0) + 1

        predictions: dict = {}
        confidence_results: dict = {}
        shadow_predictions: tuple = ()
        preheat_plan = None
        comparisons: list = []
        model_errors: tuple[str, ...] = ()

        if mode is LearningRuntimeMode.SHADOW:
            result = zr.orchestrator.run()
            predictions = dict(result.predictions)
            confidence_results = dict(result.confidence_results)
            model_errors = result.model_errors
            shadow_predictions = tuple(
                zr.shadow.shadow_predictions(snapshot.decision_id, predictions))
            # snapshot the predictions available at this decision (bounded, dedup)
            for sp in shadow_predictions:
                if sp.values:
                    key, value = next(iter(sp.values.items()))
                    zr.ledger.record(PredictionSnapshot(
                        decision_id=sp.decision_id, learning_zone_id=inp.zone_id,
                        prediction_type=sp.prediction_type, value=float(value),
                        unit=sp.units.get(key, ""), confidence=sp.confidence,
                        reliability=sp.reliability, fallback_used=sp.fallback_used,
                        prior_contribution=sp.prior_contribution,
                        learned_contribution=sp.learned_contribution, bucket=None,
                        model_version=sp.model_version, parameter_version=sp.parameter_version,
                        created_ts=inp.ts,
                        target_ts=snapshot.schedule_comfort_time_utc))
            preheat_plan = zr.shadow.build_preheat_plan(snapshot, predictions)
            preheat_conf = _purpose_value(confidence_results, ConfidencePurpose.PREHEAT)
            cmp = zr.shadow.compare_preheat(snapshot, preheat_plan,
                                            inp.legacy_preheat_start_utc, preheat_conf)
            if cmp is not None:
                comparisons.append(cmp)
                zr.open_comparisons.append(cmp)
            if inp.legacy_boost_offset_c is not None:
                boost_pred = predictions.get(PredictionType.BOOST_FACTOR)
                shadow_boost = boost_pred.values.get("boost_factor") if boost_pred else None
                comparisons.append(zr.shadow.compare_scalar(
                    snapshot, ComparisonType.BOOST_OFFSET, inp.legacy_boost_offset_c,
                    shadow_boost, "celsius_offset",
                    _purpose_value(confidence_results, ConfidencePurpose.BOOST)))
            # bound open comparisons
            if len(zr.open_comparisons) > self._config.max_open_comparisons:
                zr.open_comparisons = zr.open_comparisons[-self._config.max_open_comparisons:]

        if authoritative_change and self._persistence is not None:
            self._persistence.mark_dirty(inp.ts)

        return RuntimeCycleResult(
            zone_id=inp.zone_id, ts=inp.ts, mode=mode.value, decision_id=snapshot.decision_id,
            captured=True, shadow_predictions=shadow_predictions, preheat_plan=preheat_plan,
            comparisons=tuple(comparisons), confidence_results=confidence_results,
            model_errors=model_errors, warnings=tuple(warnings),
            dirty=self._persistence.dirty if self._persistence else False)

    def run_cycle_safe(self, inp: RuntimeCycleInput) -> RuntimeCycleResult:
        """Bridge entrypoint: NEVER raises; a failure cannot reach heating control."""
        try:
            return self.run_cycle(inp)
        except Exception as err:
            return RuntimeCycleResult(
                zone_id=getattr(inp, "zone_id", "?"), ts=getattr(inp, "ts", ""),
                mode=self._config.mode.value, decision_id=None, captured=False,
                errors=(RuntimeError("cycle_failed", getattr(inp, "zone_id", "?"),
                                     type(err).__name__),))

    # -- async lifecycle ------------------------------------------------

    async def async_setup(self) -> bool:
        self._initialized = True
        if self._persistence is None:
            return True
        data = await self._persistence.load()
        if data:
            self._restore_payload(data)
        return True

    def _restore_payload(self, data: Mapping[str, Any]) -> None:
        if data.get("runtime_schema_version") != RUNTIME_SCHEMA_VERSION:
            return
        for zid, zdata in data.get("zones", {}).items():
            self._zone(zid).restore(zdata)

    def _build_payload(self) -> dict:
        return {
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            "le_version": self._config.le_version,
            "zones": {zid: zr.serialize() for zid, zr in sorted(self._zones.items())},
        }

    async def async_save_if_due(self) -> SaveTrigger:
        if self._persistence is None:
            return SaveTrigger.NONE
        return await self._persistence.maybe_save(self._clock(), self._build_payload)

    async def async_flush(self) -> bool:
        if self._persistence is None:
            return False
        # graceful flush persists open builder snapshots even if not strictly dirty
        return await self._persistence.flush(self._clock(), self._build_payload, force=True)

    async def async_unload(self) -> bool:
        flushed = await self.async_flush()
        return flushed

    def remove_zone(self, zone_id: str) -> None:
        self._zones.pop(zone_id, None)

    def reset(self) -> None:
        self._zones = {}
        self._initialized = False

    def mark_dirty(self, *, important: bool = False) -> None:
        if self._persistence is not None:
            self._persistence.mark_dirty(self._clock(), important=important)

    def health(self) -> RuntimeHealth:
        return RuntimeHealth(
            mode=self._config.mode.value, zones=len(self._zones), initialized=self._initialized,
            dirty=self._persistence.dirty if self._persistence else False,
            last_cycle_ts=self._last_cycle_ts,
            last_save=self._persistence.health()["last_save"] if self._persistence else None,
            open_decisions=sum(1 for z in self._zones.values()
                               if z.capture.open_decision is not None),
            open_comparisons=sum(len(z.open_comparisons) for z in self._zones.values()),
            model_errors=0,
            storage_warnings=len(self._persistence.warnings) if self._persistence else 0,
            model_update_total=sum(sum(z.model_update_counts.values())
                                   for z in self._zones.values()),
            prediction_snapshots=sum(z.ledger.size for z in self._zones.values()))


class CoordinatorBridge:
    """Thin passive bridge: translate runtime values -> cycle, return diagnostics only.

    Guarantees: never raises into the caller, never returns a control command.
    """

    def __init__(self, runtime: LearningRuntime) -> None:
        self._runtime = runtime

    def process(self, inp: RuntimeCycleInput) -> RuntimeCycleResult:
        return self._runtime.run_cycle_safe(inp)


def _purpose_value(confidence_results: Mapping[str, Any], purpose: ConfidencePurpose) -> float:
    res = confidence_results.get(purpose.value)
    return float(res.value) if res is not None else 0.2
