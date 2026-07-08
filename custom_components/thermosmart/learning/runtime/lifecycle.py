"""LearningRuntime: passive shadow runtime foundation for Learning (pure core).

Ties capture -> evidence -> model orchestration -> shadow -> persistence into a
deterministic per-cycle pipeline across zones. STRICTLY passive: it returns only
typed diagnostics/shadow results and can never emit a control command. Any
failure is isolated per zone and surfaced as a structured RuntimeError —
"learning failure must never become heating failure". Async lifecycle methods use
an injected store + clock; there are no HA imports here.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime
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
from .control import (
    ControlApplicationLedger,
    ControlContext,
    ControlDecision,
    ControlFeature,
    ControlledPrediction,
    ControlPolicy,
    ControlPolicyParameters,
)
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
    ADVISORY = "advisory"   # predictions visible in diagnostics; no control effect
    CONTROL = "control"     # explicitly-released prediction paths may adjust control


@dataclass(frozen=True)
class LearningRuntimeConfig:
    le_version: str = "2.0.0"
    mode: LearningRuntimeMode = LearningRuntimeMode.SHADOW
    save_policy: SavePolicy = field(default_factory=SavePolicy)
    preheat_params: PreheatParameters = field(default_factory=PreheatParameters)
    startup_grace_cycles: int = 1
    max_open_comparisons: int = 200
    control_policy: ControlPolicyParameters = field(default_factory=ControlPolicyParameters)


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
    model_update_counts: Mapping[str, int] = field(default_factory=dict)
    control_policy_version: int = 0
    control_enabled: bool = False
    control_applied: int = 0
    control_fallback: int = 0
    control_rejections: Mapping[str, int] = field(default_factory=dict)
    reversion_count: int = 0


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
        self.last_predictions: Mapping[Any, Any] = {}
        self.last_confidence: Mapping[str, Any] = {}
        self.last_preheat_plan: Any = None
        self.last_decision_id: Optional[str] = None
        self.last_regime: Optional[str] = None
        self.control_ledger = ControlApplicationLedger()
        # B2b-1: boost update contexts materialised at decision time, awaiting their
        # OutcomeEpisode to close (cross-cycle). Bounded FIFO, keyed by decision_id.
        self.pending_boost_contexts: "OrderedDict[str, Any]" = OrderedDict()
        self._pending_boost_cap = 64
        # B2b-3: authoritative dispatch records awaiting their OutcomeEpisode + the
        # per-zone non-boost baseline comparison store.
        self.pending_dispatch_records: "OrderedDict[str, Any]" = OrderedDict()
        # B2b-3a: query features FROZEN at the boost decision (first boost cycle of a
        # decision), keyed by decision_id — never reconstructed from the episode end.
        self.pending_baseline_ctx: "OrderedDict[str, Any]" = OrderedDict()
        from ..models import BaselineComparisonStore
        self.baseline_store = BaselineComparisonStore(zone_id)
        self.last_baseline_estimate: Any = None
        self._last_snapshot: Any = None
        # B2b-4d: at most ONE active deferred boost finalization per zone (post-reach observer).
        self.pending_boost_outcome: Any = None

    def serialize(self) -> dict:
        # B2b-attribution: persist pending boost contexts and dispatch records so that a
        # real dispatch that occurred before episode-close is never silently lost on restart.
        # pending_boost_contexts (BoostUpdateContext) and pending_dispatch_records
        # (BoostDispatchRecord) are persisted alongside pending_baseline_ctx (plain dict).
        try:
            _pbc = {did: bctx.to_dict()
                    for did, bctx in self.pending_boost_contexts.items()}
        except Exception:
            _pbc = {}
        try:
            _pdr = {did: drec.to_dict()
                    for did, drec in self.pending_dispatch_records.items()}
        except Exception:
            _pdr = {}
        try:
            _pbx = {did: dict(ctx)
                    for did, ctx in self.pending_baseline_ctx.items()}
        except Exception:
            _pbx = {}
        return {"capture": self.capture.serialize(),
                "pipeline": self.pipeline.serialize(),
                "models": self.orchestrator.serialize_models(),
                "ledger": self.ledger.serialize(),
                "baseline_store": self.baseline_store.serialize(),
                "pending_boost_outcome": (self.pending_boost_outcome.to_dict()
                                          if self.pending_boost_outcome is not None else None),
                "pending_boost_contexts": _pbc,
                "pending_dispatch_records": _pdr,
                "pending_baseline_ctx": _pbx,
                "model_update_counts": dict(self.model_update_counts),
                "cycles": self.cycles, "last_cycle_ts": self.last_cycle_ts}

    def pending_attribution_summary(self) -> dict:
        """Compact, privacy-safe summary of pending pre-episode-close attribution state.

        Suitable for Support Export and Research Export.  Never raises; returns a
        zero-state summary on any error.  No raw decision IDs or entity IDs are exposed.
        """
        try:
            bc = self.pending_boost_contexts
            dr = self.pending_dispatch_records
            dispatch_statuses: dict[str, int] = {}
            for r in dr.values():
                s = getattr(r, "dispatch_status", "unknown")
                dispatch_statuses[s] = dispatch_statuses.get(s, 0) + 1
            control_types = sorted(set(
                getattr(r, "device_control_type", "setpoint") for r in dr.values()))
            return {
                "schema_version": 1,
                "pending_context_count": len(bc),
                "pending_dispatch_count": len(dr),
                "outcome_eligible_count": sum(
                    1 for r in dr.values() if getattr(r, "outcome_eligible", False)),
                "dispatch_statuses": dispatch_statuses,
                "control_types": control_types,
                "total_targets": sum(getattr(r, "targets_total", 0) for r in dr.values()),
                "failed_targets": sum(getattr(r, "targets_failed", 0) for r in dr.values()),
                "pending_outcome_active": self.pending_boost_outcome is not None,
            }
        except Exception:
            return {"schema_version": 1, "pending_context_count": 0,
                    "pending_dispatch_count": 0}

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
        if "baseline_store" in data:
            try:
                self.baseline_store.restore(data["baseline_store"])
            except Exception as err:
                errors.append(f"baseline_store:restore:{type(err).__name__}")
        if data.get("pending_boost_outcome") is not None:
            try:
                from .boost_pending import BoostPendingOutcome
                self.pending_boost_outcome = BoostPendingOutcome.from_dict(
                    data["pending_boost_outcome"])   # None on unknown version / corrupt
            except Exception as err:
                self.pending_boost_outcome = None
                errors.append(f"pending_boost:restore:{type(err).__name__}")
        # B2b-attribution: restore pending pre-episode-close contexts so a restart between
        # dispatch and episode-close does not silently lose the learning attribution.
        if "pending_boost_contexts" in data:
            try:
                from ..models import BoostUpdateContext
                for did, bctx_d in (data["pending_boost_contexts"] or {}).items():
                    try:
                        self.pending_boost_contexts[did] = BoostUpdateContext.from_dict(bctx_d)
                    except Exception:
                        pass  # corrupt entry isolated; decision loses attribution (logged above)
            except Exception as err:
                errors.append(f"pending_boost_contexts:restore:{type(err).__name__}")
        if "pending_dispatch_records" in data:
            try:
                from .capture import BoostDispatchRecord
                for did, drec_d in (data["pending_dispatch_records"] or {}).items():
                    try:
                        self.pending_dispatch_records[did] = BoostDispatchRecord.from_dict(drec_d)
                    except Exception:
                        pass  # corrupt entry isolated
            except Exception as err:
                errors.append(f"pending_dispatch_records:restore:{type(err).__name__}")
        if "pending_baseline_ctx" in data:
            try:
                for did, ctx in (data["pending_baseline_ctx"] or {}).items():
                    try:
                        self.pending_baseline_ctx[did] = dict(ctx)
                    except Exception:
                        pass
            except Exception as err:
                errors.append(f"pending_baseline_ctx:restore:{type(err).__name__}")
        self.model_update_counts = dict(data.get("model_update_counts", {}))
        self.cycles = data.get("cycles", 0)
        self.last_cycle_ts = data.get("last_cycle_ts")
        return tuple(errors)


class LearningRuntime:
    """Multi-zone passive runtime. Never mutates the existing controller."""

    def __init__(self, config: Optional[LearningRuntimeConfig] = None, *,
                 store: Optional[Any] = None,
                 clock: Optional[Callable[[], str]] = None,
                 episode_sink: Optional[Callable[[Any], None]] = None) -> None:
        self._config = config or LearningRuntimeConfig()
        self._zones: dict[str, _ZoneRuntime] = {}
        self._store = store
        # Optional, synchronous, memory-only sink for completed episodes (see
        # run_cycle()'s completed-episode loop). No HA imports, no storage I/O,
        # no async here — the sink itself (if provided) is responsible for
        # staying synchronous and non-fatal; run_cycle() also wraps every call
        # to it in its own try/except so a sink failure can never affect the
        # cycle's real result. None (the default) makes this a pure no-op —
        # existing/minimal setups and tests are unaffected.
        self._episode_sink = episode_sink
        self._persistence = PersistenceOrchestrator(store, policy=self._config.save_policy) \
            if store is not None else None
        # Clock authority: explicit injection required. No hidden wall-clock fallback.
        # A missing clock is a programming error surfaced by a ValueError.
        # Tests that are not testing clock behavior should pass an explicit lambda
        # (e.g. clock=lambda: "2025-01-01T00:00:00+00:00") or use helpers_runtime_scenarios.runtime().
        if clock is None:
            raise ValueError(
                "LearningRuntime requires an explicit clock callable — "
                "pass a SystemClock-backed callable (production) or "
                "lambda: <fixed_iso_string> / FakeClock (tests). "
                "Convenience wrapper: helpers_runtime_scenarios.runtime() injects a default."
            )
        self._clock = clock
        self._initialized = False
        self._last_cycle_ts: Optional[str] = None
        self._mode = self._config.mode
        self._reversion_count = 0
        self._policy = ControlPolicy(
            self._config.control_policy,
            runtime_is_control=self._mode is LearningRuntimeMode.CONTROL)

    @property
    def mode(self) -> LearningRuntimeMode:
        return self._mode

    @property
    def control_enabled(self) -> bool:
        return self._mode is LearningRuntimeMode.CONTROL

    def set_mode(self, mode: LearningRuntimeMode) -> None:
        """Switch runtime mode in-place (e.g. CONTROL<->SHADOW) with no state loss.

        Learning/builder/ledger state is preserved; only the control authority of
        the policy changes. A switch away from CONTROL counts as a reversion.
        """
        if mode is not LearningRuntimeMode.CONTROL and self._mode is LearningRuntimeMode.CONTROL:
            self._reversion_count += 1
        self._mode = mode
        self._policy = ControlPolicy(
            self._config.control_policy,
            runtime_is_control=mode is LearningRuntimeMode.CONTROL)

    def control_decision(self, zone_id: str, feature: ControlFeature,
                         baseline_value: Optional[float], proposed_value: Optional[float],
                         purpose: ConfidencePurpose, *, context: Optional[ControlContext] = None,
                         unit: str = "", model_version: Optional[int] = None,
                         parameter_version: Optional[int] = None,
                         ts: Optional[str] = None) -> ControlDecision:
        """Resolve one control feature from the last cycle's confidence result.

        Baseline-first: returns the baseline unchanged unless the policy applies.
        Records every attempt in the per-zone control ledger. No dispatch.
        """
        zr = self._zone(zone_id)
        cr = zr.last_confidence.get(purpose.value)
        prediction = None
        _auth = context is not None and context.authorized_override
        # For authorized override: build a prediction even without a confidence result so
        # ControlPolicy.resolve() can apply clamping; the confidence gates are bypassed.
        if proposed_value is not None and (cr is not None or _auth):
            prediction = ControlledPrediction(
                feature=feature, value=float(proposed_value), unit=unit,
                confidence_result=cr, model_version=model_version,
                parameter_version=parameter_version)
        decision = self._policy.resolve(feature, baseline_value, prediction,
                                        context=context, unit=unit)
        zr.control_ledger.record(zone_id, zr.last_decision_id or "?", decision,
                                 ts or self._clock())
        return decision

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _zone(self, zone_id: str) -> _ZoneRuntime:
        zr = self._zones.get(zone_id)
        if zr is None:
            zr = _ZoneRuntime(zone_id, self._config)
            self._zones[zone_id] = zr
        return zr

    def _emit_completed_episode(self, episode: Any) -> None:
        """Best-effort hand-off of one completed episode to the optional sink.

        Synchronous, no storage I/O here — the sink itself (see
        LearningShadowController.record_completed_episode_safe()) is required
        to be synchronous and memory-only too. A missing sink is a silent
        no-op. A sink exception is swallowed here so it can never affect
        run_cycle()'s real result — the whole point of a narrowly-scoped
        try/except at each call site rather than relying on run_cycle_safe()'s
        coarser outer catch.
        """
        if self._episode_sink is None:
            return
        try:
            self._episode_sink(episode)
        except Exception:
            pass

    # -- core cycle (pure-ish; mutates only Learning state) ------------------

    def run_cycle(self, inp: RuntimeCycleInput) -> RuntimeCycleResult:
        mode = self._mode
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

        # B2b-1: stash the authoritative boost update context (decision-time) until its
        # OutcomeEpisode closes cross-cycle. Bounded FIFO; keyed by decision_id.
        if evidence.boost_context is not None:
            did = evidence.boost_context.decision_id
            zr.pending_boost_contexts.pop(did, None)
            zr.pending_boost_contexts[did] = evidence.boost_context
            while len(zr.pending_boost_contexts) > zr._pending_boost_cap:
                zr.pending_boost_contexts.popitem(last=False)
            # B2b-3a: freeze the matching features at the DECISION (first boost cycle); do
            # not overwrite on later cycles of the same decision (avoids end-state leakage).
            if did not in zr.pending_baseline_ctx:
                zr.pending_baseline_ctx[did] = {
                    "external_temperature_c": (snapshot.outdoor_temp.value
                                               if getattr(snapshot, "outdoor_temp", None)
                                               is not None else None),
                    "heat_rate_used": self._baseline_heat_rate(zr),
                    "device_control_type": "setpoint", "decision_ts": snapshot.ts}
                while len(zr.pending_baseline_ctx) > zr._pending_boost_cap:
                    zr.pending_baseline_ctx.popitem(last=False)
        # B2b-3: stash the authoritative dispatch record per decision (for non-boost
        # baseline-sample creation and boost baseline-query at the OutcomeEpisode close).
        if snapshot.boost_dispatch is not None:
            ddid = snapshot.decision_id     # authoritative capture id (matches ce.decision_id)
            zr.pending_dispatch_records.pop(ddid, None)
            zr.pending_dispatch_records[ddid] = snapshot.boost_dispatch
            while len(zr.pending_dispatch_records) > zr._pending_boost_cap:
                zr.pending_dispatch_records.popitem(last=False)
        zr._last_snapshot = snapshot        # for baseline context at outcome close

        # B2b-4d: advance any deferred post-reach boost observer with THIS cycle's sample;
        # finalize (exactly one BoostModel.update) when its fixed deadline is reached or an
        # interruption breaks continuity. Runs before the pipeline so a finalize this cycle
        # is counted.
        if zr.pending_boost_outcome is not None:
            try:
                if self._tick_pending_boost(zr, snapshot):
                    authoritative_change = True
            except Exception as err:
                self._record_runtime_error("pending_boost_tick", err)

        # cross-cycle episode pipeline -> completed authoritative episodes -> updates.
        # Transaction: completed episodes are fully materialised by the builders
        # before any model update; each update is eligibility-gated and isolated.
        pipe_result = zr.pipeline.process(snapshot)
        zr.last_regime = pipe_result.regime
        if startup_ok and not pipe_result.disturbed and not pipe_result.deduplicated:
            for ce in pipe_result.completed:
                ok = zr.orchestrator.apply_update(ce.model_name, ce.episode)
                if ok:
                    authoritative_change = True
                    zr.model_update_counts[ce.model_name] = \
                        zr.model_update_counts.get(ce.model_name, 0) + 1
                # Episode-history hand-off (memory-only; see _emit_completed_episode()).
                # Independent of `ok` — episode completion is a different fact than
                # model acceptance, and persisting the history should not depend on
                # it. Outcome is excluded here: only the confounder-augmented
                # bound_episode below is persisted for outcome, never the raw one.
                if ce.model_name != "outcome":
                    self._emit_completed_episode(ce.episode)
                # B2b-1: fan out the closed OutcomeEpisode to the BoostModel with its
                # authoritative context — but ONLY when a boost was actually dispatched
                # for this decision (a pending context exists). No context => no boost
                # update (failed/not-attempted/zero-applied were filtered at materialise).
                if ce.model_name == "outcome" and ce.decision_id is not None:
                    bctx = zr.pending_boost_contexts.pop(ce.decision_id, None)
                    drec = zr.pending_dispatch_records.pop(ce.decision_id, None)
                    # B2b-4b: augment the episode with runtime confounders from authoritative
                    # evidence (multi-TRV, supply-delay, early-cutoff, afterheat). The SAME
                    # augmented episode feeds both the boost outcome and the baseline path.
                    extra_cf = self._runtime_confounders(zr, ce.episode, drec, snapshot)
                    bound_episode = self._augment_confounders(ce.episode, extra_cf)
                    # Episode-history hand-off for outcome: the augmented bound_episode
                    # (with runtime confounder flags), never the raw ce.episode. Emitted
                    # immediately here, independent of the boost REACHED/non-REACHED/
                    # baseline branching below — the outcome episode's own completion is
                    # unrelated to whether/when the (possibly deferred) boost model
                    # update resolves.
                    self._emit_completed_episode(bound_episode)
                    if bctx is not None:
                        # B2b-3: derive a trustworthy expected_non_boost_duration_s from
                        # comparable historical NON-boost episodes completed before now
                        # (baseline FROZEN at decision; cutoff unchanged).
                        bctx = self._inject_baseline(zr, bctx, bound_episode, snapshot, drec)
                        bctx = replace(bctx, source_episode_id=bound_episode.episode_id)
                        from ..episode_schemas import EpisodeReason as _ER
                        if bound_episode.reason is _ER.REACHED:
                            # B2b-4d: DEFER — a boost/afterheat overshoot may begin after the
                            # in-band dwell. Open the post-reach observer; finalize later.
                            self._open_pending_boost(zr, bound_episode, bctx, snapshot)
                        else:
                            # non-REACHED: no SUCCESS possible -> finalize immediately.
                            if zr.orchestrator.apply_update("boost", bound_episode, bctx):
                                authoritative_change = True
                                zr.model_update_counts["boost"] = \
                                    zr.model_update_counts.get("boost", 0) + 1
                    else:
                        # B2b-3: an authoritative NON-boost outcome becomes a baseline sample.
                        self._maybe_store_baseline(zr, bound_episode, drec, snapshot)

        predictions: dict = {}
        confidence_results: dict = {}
        shadow_predictions: tuple = ()
        preheat_plan = None
        comparisons: list = []
        model_errors: tuple[str, ...] = ()

        if mode in (LearningRuntimeMode.SHADOW, LearningRuntimeMode.ADVISORY,
                    LearningRuntimeMode.CONTROL):
            result = zr.orchestrator.run(now=inp.ts)
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

        # expose the latest predictions/confidence for an optional CONTROL resolver
        zr.last_predictions = predictions
        zr.last_confidence = confidence_results
        zr.last_preheat_plan = preheat_plan
        zr.last_decision_id = snapshot.decision_id

        if authoritative_change and self._persistence is not None:
            self._persistence.mark_dirty(inp.ts)

        return RuntimeCycleResult(
            zone_id=inp.zone_id, ts=inp.ts, mode=mode.value, decision_id=snapshot.decision_id,
            captured=True, shadow_predictions=shadow_predictions, preheat_plan=preheat_plan,
            comparisons=tuple(comparisons), confidence_results=confidence_results,
            model_errors=model_errors, warnings=tuple(warnings),
            dirty=self._persistence.dirty if self._persistence else False)

    def _baseline_heat_rate(self, zr: "_ZoneRuntime"):
        from ..contracts import PredictionType
        pred = (zr.last_predictions or {}).get(PredictionType.HEAT_RATE)
        try:
            return float(pred.values.get("heat_rate")) if pred else None
        except (AttributeError, TypeError, ValueError):
            return None

    def _runtime_confounders(self, zr, episode, dispatch, snapshot) -> tuple:
        """B2b-4b: derive runtime confounders from authoritative episode/dispatch evidence
        (multi-TRV per-device, supply-delay vs. expected onset, early-cutoff overlap,
        afterheat interaction). Returned as canonical flag strings; fed to the SAME central
        evaluator for both the boost outcome and the non-boost baseline. Never raises."""
        flags: list[str] = []
        try:
            from ..models import (BoostDeviceDispatchEvidence, evaluate_multi_trv,
                                  supply_delay_confounder, afterheat_attribution_severity,
                                  ConfounderSeverity, BoostConfounder)
            from ..contracts import PredictionType
            # ── multi-TRV from authoritative per-device effective setpoints ──
            if dispatch is not None:
                effs = list(getattr(dispatch, "effective_setpoints", ()) or ())
                base_sp = getattr(dispatch, "baseline_setpoint_c", None)
                dctype = getattr(dispatch, "device_control_type", "setpoint")
                applied = getattr(dispatch, "boost_applied_c", None)
                if effs and base_sp is not None:
                    devices = [BoostDeviceDispatchEvidence(
                        slot=f"trv{i}", control_type=dctype,
                        requested_offset_c=applied,
                        effective_offset_c=round(float(sp) - float(base_sp), 4),
                        clamp_applied_c=(abs((applied or 0.0) - (float(sp) - float(base_sp)))
                                         if applied is not None else 0.0))
                        for i, sp in enumerate(effs)]
                    total = int(getattr(dispatch, "targets_total", len(effs)) or len(effs))
                    failed = int(getattr(dispatch, "targets_failed", 0) or 0)
                    for j in range(max(0, total - len(effs)) + failed):
                        devices.append(BoostDeviceDispatchEvidence(
                            slot=f"trvF{j}", control_type=dctype, requested_offset_c=applied,
                            effective_offset_c=None, dispatch_succeeded=False))
                    mtv = evaluate_multi_trv(devices)
                    if mtv is not None:
                        flags.append(mtv.value)
            # ── supply-delay: expected onset (frozen) vs. actual effective onset ──
            try:
                preds = getattr(zr, "last_predictions", {}) or {}
                od = preds.get(PredictionType.ONSET_DELAY)
                expected_onset = (float(od.values.get("onset_delay_minutes"))
                                  if od is not None and not getattr(od, "fallback_used", True)
                                  else None)
                from ..models.baseline_comparison import _onset_and_target
                onset_ms, _tm = _onset_and_target(
                    episode.trajectory.points, episode.target, episode.start_temp,
                    getattr(episode, "comfort_tolerance_at_start", 0.5))
                actual_onset = onset_ms / 60000.0 if onset_ms is not None else None
                # Skip supply-delay assessment when no expected onset is available (cold start /
                # no onset-delay model yet): without a reference we cannot distinguish normal
                # heating from a supply-delay anomaly — treating None/None as HEATING_FAILURE
                # would confound all bootstrap outcomes.
                if expected_onset is not None:
                    sd = supply_delay_confounder(expected_onset, actual_onset)
                    if sd is not None:
                        flags.append(sd.value)
            except Exception:
                pass
            # ── early-cutoff overlap (same decision) ──
            ecs = getattr(snapshot, "early_cutoff_state", None) if snapshot is not None else None
            if ecs in ("cutoff_applied", "coasting_hold"):
                flags.append(BoostConfounder.EARLY_CUTOFF_INTERACTION.value)
            # ── afterheat interaction (high expected afterheat) ──
            try:
                ah = (getattr(zr, "last_predictions", {}) or {}).get(
                    PredictionType.EXPECTED_OVERSHOOT)
                exp_ah = float(ah.values.get("expected_overshoot")) if ah is not None else None
                if afterheat_attribution_severity(exp_ah, None) is ConfounderSeverity.DEGRADING:
                    flags.append(BoostConfounder.AFTERHEAT_INTERACTION.value)
            except Exception:
                pass
        except Exception as err:
            self._record_runtime_error("runtime_confounders", err)
        return tuple(flags)

    def _augment_confounders(self, episode, extra: tuple):
        if not extra:
            return episode
        merged = tuple(sorted(set(episode.confounder_flags or ()) | set(extra)))
        return replace(episode, confounder_flags=merged)

    # ── B2b-4d: deferred post-reach boost finalization ──────────────────────
    def _predicted_afterheat_duration_s(self, zr):
        from ..contracts import PredictionType
        pred = (zr.last_predictions or {}).get(PredictionType.AFTERHEAT_DURATION)
        try:
            v = pred.values.get("afterheat_duration") if pred else None
            return float(v) if v is not None else None
        except (AttributeError, TypeError, ValueError):
            return None

    def _device_states(self, snapshot) -> dict:
        return dict(getattr(snapshot, "device_states", {}) or {})

    def _control_type(self, snapshot) -> str:
        dispatch = getattr(snapshot, "boost_dispatch", None)
        return getattr(dispatch, "device_control_type", "setpoint") if dispatch else "setpoint"

    def _open_pending_boost(self, zr, episode, bctx, snapshot) -> None:
        """Open the single per-zone post-reach observer at thermal REACHED.

        Supersession (B2b-4e): a pre-existing pending for a DIFFERENT decision whose deadline
        has NOT passed is closed as BOOST_INTERRUPTED (superseded) — never silently
        overwritten and never as a positive outcome. An already-due pending is finalized
        regularly first."""
        from .boost_pending import (BoostPendingOutcome, _episode_to_dict,
                                    observation_window_s)
        from ..models import BoostOutcomeLifecycleState as _LS
        from datetime import timedelta
        try:
            old = zr.pending_boost_outcome
            if old is not None and old.decision_id != episode.decision_id:
                if old.is_due(snapshot.ts):
                    self._finalize_pending_boost(zr, old, reason="deadline")
                else:
                    old.add_confounder("superseded_by_new_decision")
                    old.state = _LS.INTERRUPTED.value
                    self._finalize_pending_boost(zr, old, reason="superseded_by_new_decision")
            elif old is not None and old.decision_id == episode.decision_id:
                return                          # same decision / retry -> no supersession
            end_ts = episode.end_ts
            window = observation_window_s(self._predicted_afterheat_duration_s(zr))
            deadline = (end_ts + timedelta(seconds=window)).isoformat()
            p = BoostPendingOutcome(
                zone_id=zr.zone_id, decision_id=episode.decision_id,
                episode_id=episode.episode_id,
                state=_LS.TARGET_REACHED_PENDING_OBSERVATION.value, target=episode.target,
                episode_end_ts=end_ts.isoformat(), observation_start_ts=end_ts.isoformat(),
                deadline_ts=deadline, last_valid_ts=end_ts.isoformat(),
                episode=_episode_to_dict(episode), boost_context=bctx.to_dict())
            p.seed_devices(end_ts.isoformat(), self._device_states(snapshot),
                           self._control_type(snapshot))
            zr.pending_boost_outcome = p
        except Exception as err:
            self._record_runtime_error("open_pending_boost", err)

    def _tick_pending_boost(self, zr, snapshot) -> bool:
        """Advance the pending observer with this cycle's REAL sample + device states;
        finalize when due or interrupted. Returns True iff a model update happened."""
        from ..models import BoostOutcomeLifecycleState as _LS
        from .boost_pending import OBS_MAX_GAP_S
        p = zr.pending_boost_outcome
        if p is None:
            return False
        now_ts = snapshot.ts
        gap = p.gap_s(now_ts)
        continuity_gap = gap is not None and gap > OBS_MAX_GAP_S
        # real cross-cycle device availability (continuity gap => unknown not a device fault)
        p.ingest_devices(now_ts, self._device_states(snapshot), self._control_type(snapshot),
                         continuity_gap=continuity_gap)
        # continuity / interruption checks (restart gap, safety, context change)
        interrupt_reason = None
        if continuity_gap:
            interrupt_reason = "restart_gap"
        elif snapshot.window_open:
            interrupt_reason = "window_open"
        elif snapshot.heating_failure:
            interrupt_reason = "heating_failure"
        elif snapshot.target_c is not None and abs(float(snapshot.target_c) - p.target) > 0.1:
            interrupt_reason = "target_change"
        if interrupt_reason is not None:
            p.add_confounder(interrupt_reason)
            p.state = _LS.INTERRUPTED.value
            return self._finalize_pending_boost(zr, p, reason=interrupt_reason)
        temp = snapshot.indoor_temp.value if getattr(snapshot, "indoor_temp", None) is not None \
            else None
        valid = (getattr(snapshot, "indoor_temp_quality", None) is None
                 or str(getattr(snapshot.indoor_temp_quality, "value", "ok")) == "ok")
        p.ingest(now_ts, temp, valid=valid)
        if p.is_due(now_ts):
            return self._finalize_pending_boost(zr, p, reason="deadline")
        return False

    def _finalize_pending_boost(self, zr, p, *, reason: str) -> bool:
        """Crash-atomic, EXACTLY-once finalization (Variant A — single atomic zone snapshot).

        Order: (1) persist the finalization INTENT into the pending (finalized=True), (2) run
        BoostModel.update which dedups by decision_id and itself records the processed
        decision_id, (3) mark update_applied and clear the pending. The whole _ZoneRuntime
        (BoostState incl. processed_decision_ids AND the pending) serializes in ONE snapshot,
        so any crash restores a consistent state: a re-finalize after restore is idempotent
        because the model already holds the processed decision_id."""
        from ..models import BoostOutcomeLifecycleState as _LS
        updated = False
        try:
            episode = p.build_final_episode()
            from ..models import BoostUpdateContext
            bctx = BoostUpdateContext.from_dict(p.boost_context)
            bctx = replace(bctx, source_episode_id=episode.episode_id)
            p.finalized = True                  # (1) intent
            p.finalization_reason = reason
            if p.state != _LS.INTERRUPTED.value:
                p.state = _LS.FINALIZED.value
            if zr.orchestrator.apply_update("boost", episode, bctx):   # (2) idempotent update
                updated = True
                zr.model_update_counts["boost"] = zr.model_update_counts.get("boost", 0) + 1
            p.update_applied = True             # (3) effective (model dedup is the authority)
        except Exception as err:
            self._record_runtime_error("finalize_pending_boost", err)
        finally:
            zr.pending_boost_outcome = None     # clear; model dedup prevents any double update
        return updated

    def _inject_baseline(self, zr: "_ZoneRuntime", bctx, episode, snapshot, dispatch=None):
        """B2b-3/3a: query comparable historical non-boost episodes -> expected duration.

        Causal: the cutoff is the boost DECISION time (bound episode start), and matching
        features are FROZEN at the decision (pending_baseline_ctx), never reconstructed from
        the episode end. Never raises; on any issue the context is unchanged
        (expected_non_boost_duration_s stays None => INSUFFICIENT_COMPARISON)."""
        try:
            from ..models import BaselineQuery
            frozen = zr.pending_baseline_ctx.pop(bctx.decision_id, {})
            q = BaselineQuery(
                learning_zone_id=zr.zone_id, decision_id=bctx.decision_id,
                episode_id=episode.episode_id,
                baseline_cutoff_ts=episode.start_ts.isoformat(),   # decision / bound start
                start_deficit_c=(bctx.start_deficit_c
                                 if bctx.start_deficit_c is not None
                                 else episode.target - episode.start_temp),
                target_temperature_c=episode.target,
                external_temperature_c=frozen.get("external_temperature_c"),
                heat_rate_used=frozen.get("heat_rate_used"), tpi_duty=None,
                time_of_day_bucket=episode.start_ts.hour,
                device_control_type=getattr(dispatch, "device_control_type", "setpoint")
                if dispatch is not None else frozen.get("device_control_type", "setpoint"),
                query_source="live_decision")
            est = zr.baseline_store.estimate(q)
            zr.last_baseline_estimate = est
            if est.expected_non_boost_duration_s is not None:
                return replace(
                    bctx, expected_non_boost_duration_s=est.expected_non_boost_duration_s,
                    baseline_reliability=est.reliability)
        except Exception as err:
            self._record_runtime_error("baseline_inject", err)
        return bctx

    def _maybe_store_baseline(self, zr: "_ZoneRuntime", episode, dispatch, snapshot) -> None:
        """B2b-3: build + store an authoritative non-boost baseline sample. Never raises."""
        if dispatch is None:
            return
        try:
            from ..models import build_non_boost_sample
            outdoor = snapshot.outdoor_temp.value if getattr(
                snapshot, "outdoor_temp", None) is not None else None
            reject_out: list = []
            sample = build_non_boost_sample(
                episode, dispatch, external_temperature_c=outdoor,
                heat_rate_used=self._baseline_heat_rate(zr),
                device_control_type=getattr(dispatch, "device_control_type", "setpoint"),
                sensor_reliability=1.0, reject_out=reject_out)
            if sample is not None:
                zr.baseline_store.add(sample)
            elif reject_out:
                # B2b-3b: count the diagnostic reason (e.g. boost_application_unknown).
                zr.baseline_store.record_rejection(reject_out[-1])
        except Exception as err:
            self._record_runtime_error("baseline_store", err)

    def _record_runtime_error(self, kind: str, err: Exception) -> None:
        # learning-internal errors must never reach control; counted only.
        self._runtime_errors = getattr(self, "_runtime_errors", 0) + 1

    def run_cycle_safe(self, inp: RuntimeCycleInput) -> RuntimeCycleResult:
        """Bridge entrypoint: NEVER raises; a failure cannot reach heating control."""
        try:
            return self.run_cycle(inp)
        except Exception as err:
            return RuntimeCycleResult(
                zone_id=getattr(inp, "zone_id", "?"), ts=getattr(inp, "ts", ""),
                mode=self._mode.value, decision_id=None, captured=False,
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
            mode=self._mode.value, zones=len(self._zones), initialized=self._initialized,
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
            prediction_snapshots=sum(z.ledger.size for z in self._zones.values()),
            model_update_counts=self._aggregate_model_update_counts(),
            control_policy_version=self._config.control_policy.policy_version,
            control_enabled=self.control_enabled,
            control_applied=sum(sum(z.control_ledger.summary()["applied_counts"].values())
                                for z in self._zones.values()),
            control_fallback=sum(z.control_ledger.summary()["fallback_count"]
                                 for z in self._zones.values()),
            control_rejections=self._aggregate_control_rejections(),
            reversion_count=self._reversion_count)

    def pending_attribution_summary(self, zone_id: str) -> dict:
        """Return a compact, privacy-safe summary of pending attribution for one zone.

        Returns a zero-state dict on any error so callers need no special handling.
        """
        try:
            return self._zones[zone_id].pending_attribution_summary()
        except Exception:
            return {"schema_version": 1, "pending_context_count": 0,
                    "pending_dispatch_count": 0}

    def _aggregate_model_update_counts(self) -> dict:
        out: dict[str, int] = {}
        for z in self._zones.values():
            for name, count in z.model_update_counts.items():
                out[name] = out.get(name, 0) + count
        return out

    def _aggregate_control_rejections(self) -> dict:
        out: dict[str, int] = {}
        for z in self._zones.values():
            for reason, count in z.control_ledger.summary()["rejection_counts"].items():
                out[reason] = out.get(reason, 0) + count
        return out


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
