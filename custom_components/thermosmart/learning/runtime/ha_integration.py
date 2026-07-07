"""Home Assistant integration layer for the LE2 runtime.

The ONLY glue between the live ThermoSmart coordinator and the LE2 runtime.
It builds a typed ``RuntimeCycleInput`` from values the coordinator has
already computed, runs a prediction-only observation cycle behind a hard
guard, and stores diagnostics. This layer also exposes bounded
adaptive-control adjustments (e.g. boost/preheat suggestions); it never
dispatches a Home Assistant service call itself and never lets an exception
reach the heating path — the coordinator is the sole place that decides,
per cycle via ``ControlAdaptationMode`` (coordinator.py), whether an
adjustment is ignored, shadowed, or actually applied.

This module imports Home Assistant (it is the integration shell, not the
pure core); it is deliberately NOT exported from ``runtime/__init__`` so the
pure runtime stays importable without HA.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping, Optional

from ...const import TPI_MAX_BOOST_CELSIUS
from ..clock import Clock
from ..contracts import DataQuality, Measurement
from ..confidence import ConfidencePurpose
from ..contracts import PredictionType
from .capture import ControllerDecisionInput, DecisionType, RuntimeCycleInput, ScheduleTarget
from .control import ControlContext, ControlFeature
from ..decision import DecisionMode, DecisionPipeline, FinalResolver
from ..decision.runtime_adapter import build_learning_prediction_set
from .ha_store import HomeAssistantStoreAdapter
from .lifecycle import (
    CoordinatorBridge,
    LearningRuntime,
    LearningRuntimeConfig,
    LearningRuntimeMode,
)

_LOGGER = logging.getLogger(__name__)
_ERROR_LOG_THROTTLE = 50  # log at most 1 of every N identical errors

# ── Adaptive Boost Control Resolver ──────────────────────────────────────────
ADAPTIVE_BOOST_RESOLVER_VERSION = 1


class BoostBlockReason(StrEnum):
    """Typed enum for boost gate blocking and lifecycle release reasons.

    Every outer-gate and dispatch-failure reason is a typed constant so control code
    never depends on raw string literals.  Lifecycle-internal reasons (window_open,
    heating_failure, etc.) are owned by adjust_recommendation_safe() and remain as-is.
    StrEnum: instances are both str and enum — serialization and direct string comparison
    both work without explicit .value access.
    """
    LEARNING_MODE_OFF = "learning_mode_off"
    ACTIVE_CONTROL_OFF = "active_control_off"
    MIXED_CONTROL_TYPES = "mixed_control_types_unsupported"
    DEVICE_UNAVAILABLE = "device_unavailable"
    FAILED_DISPATCH = "failed_dispatch"
    PARTIAL_DISPATCH = "partial_dispatch"
    READINESS_UNAVAILABLE = "readiness_unavailable"
    RESTORE_PENDING = "restore_pending"


@dataclass(frozen=True)
class AdaptiveBoostControlResult:
    """Typed output of the central adaptive boost resolver for one coordinator cycle.

    Field semantics:
      ``approved_boost_offset_c``  — 0.0 when any gate blocks; never None.
      ``applied_boost_offset_c``   — None when inner gates passed and the boost was written
                                     to trv_setpoint (pre-dispatch; not yet confirmed by
                                     service call); 0.0 when any gate blocked it (definitely
                                     not applied).  Post-dispatch truth lives in LiveDecisionRecord.
    """
    requested_boost_offset_c: Optional[float]   # raw LE2 proposal (None = no prediction)
    approved_boost_offset_c: float              # 0.0 when any gate blocks; never None
    applied_boost_offset_c: Optional[float]     # None = pre-dispatch; 0.0 = blocked
    boost_allowed: bool                         # True only when all outer gates pass
    selected_scope: Optional[str]               # bucket | general | general_fallback | prior
    eligibility_reason: str                     # BoostEligibilityReason or gate reason
    blocking_reason: Optional[str]              # first gate that blocked this cycle
    release_reason: Optional[str]               # lifecycle release reason, if triggered
    clamp_applied: bool                         # True when step/device clamp modified offset
    source_decision_id: Optional[str]           # LE2 decision_id that produced the boost
    # Trace fields — cycle-bound, never persisted
    authorization_source: Optional[str] = None  # "bootstrap_activation" when authorized_override
    bypassed_gates: tuple = ()                  # confidence gates bypassed by authorized_override
    component_version: int = ADAPTIVE_BOOST_RESOLVER_VERSION


def _zone_segment(zone_id: str) -> str:
    """Non-identifying, stable token for the store key (never an entity name)."""
    return hashlib.sha256(zone_id.encode("utf-8")).hexdigest()[:16]


def _is_celsius_unit(unit: str) -> bool:
    """Return True when the unit string represents Celsius temperature.

    Accepts only forms actually produced by LE2 model contracts:
      "C"       — AfterheatModel, HeatRate, HeatLoss (short form)
      "celsius" — ForecastModel, ManualCorrection, BoostOutcome (long form)

    Rejects "K", "%", "min", empty string, or any unknown unit.
    Never performs Kelvin conversion; missing unit → conservative False.
    """
    return unit in ("C", "celsius")


def _decision_type(recommendation: Mapping[str, Any]) -> DecisionType:
    try:
        if recommendation.get("preheat_minutes", 0) and recommendation.get("preheat_minutes", 0) > 0:
            return DecisionType.PREHEAT
        if recommendation.get("boost_active"):
            return DecisionType.BOOST
        if recommendation.get("window_open") or recommendation.get("is_summer"):
            return DecisionType.IDLE
        return DecisionType.NORMAL
    except Exception:
        return DecisionType.UNKNOWN


def _boost_dispatch_from_live_record(live_record: Any) -> Optional[Any]:
    """B2b-1: map the authoritative LiveDecisionRecord -> BoostDispatchRecord.

    This is the single source of truth for what boost was actually written. Returns
    None when no record is available (runtime then uses the marked legacy fallback).
    """
    if live_record is None:
        return None
    try:
        from .capture import BoostDispatchRecord
        did = getattr(live_record, "decision_id", None)
        if not did:
            return None
        # B2b-3c: device control type from the post-dispatch path ("setpoint"/"direct_valve").
        dpath = getattr(live_record, "dispatch_path", None)
        device_type = dpath if dpath in ("setpoint", "direct_valve") else "setpoint"
        return BoostDispatchRecord(
            decision_id=did,
            boost_candidate_c=getattr(live_record, "boost_candidate_c", None),
            # explicit identity: an authoritative 0.0 must NOT become None.
            boost_applied_c=getattr(live_record, "boost_applied_c", None),
            baseline_setpoint_c=getattr(live_record, "baseline_setpoint_c", None),
            final_setpoint_c=getattr(live_record, "final_setpoint_c", None),
            effective_setpoint_min_c=getattr(live_record, "dispatch_effective_setpoint_min_c", None),
            effective_setpoint_max_c=getattr(live_record, "dispatch_effective_setpoint_max_c", None),
            dispatch_status=getattr(live_record, "dispatch_status", "not_attempted"),
            outcome_eligible=bool(getattr(live_record, "outcome_eligible", False)),
            outcome_reliability=getattr(live_record, "outcome_reliability", "none"),
            boost_evaluation_status=getattr(live_record, "boost_evaluation_status", "unknown"),
            device_control_type=device_type,
            effective_setpoints=tuple(
                getattr(live_record, "dispatch_effective_setpoints", ()) or ()),
            targets_total=int(getattr(live_record, "dispatch_targets_total", 0) or 0),
            targets_failed=int(getattr(live_record, "dispatch_targets_failed", 0) or 0))
    except Exception:
        return None


def build_runtime_cycle_input(zone_id: str, recommendation: Mapping[str, Any], *,
                              weather: Optional[Mapping[str, Any]] = None,
                              schedule_comfort_time_utc: Optional[str] = None,
                              schedule_comfort_temperature_c: Optional[float] = None,
                              ts_iso: Optional[str] = None,
                              heating_failure: bool = False,
                              live_record: Any = None) -> RuntimeCycleInput:
    """Materialise a typed cycle input from already-computed coordinator values.

    Defensive: only reads present values, never invents a temperature, keeps 0.0,
    and uses Measurement/None for missing data. Pure given its inputs.
    """
    rec = recommendation or {}
    if ts_iso is None:
        raise ValueError(
            "build_runtime_cycle_input: ts_iso is required — "
            "pass self._utcnow_iso() from the LearningShadowController."
        )
    ts = ts_iso
    target = rec.get("effective_target")
    if target is None:
        target = rec.get("adjusted_target")
    setpoint = rec.get("trv_setpoint")
    current = rec.get("current_temp")
    indoor = Measurement(current, DataQuality.OK) if current is not None else None

    demand = None
    duty = rec.get("tpi_duty_cycle")
    if duty is not None:
        demand = duty > 0.0
    elif setpoint is not None and current is not None:
        demand = setpoint > current + 0.3

    boost_offset = None
    if setpoint is not None and target is not None and setpoint > target:
        boost_offset = min(setpoint - target, TPI_MAX_BOOST_CELSIUS)

    outdoor = None
    if weather:
        ot = weather.get("outdoor_temp") or weather.get("temperature")
        if ot is not None:
            outdoor = Measurement(ot, DataQuality.OK)

    decision_type = _decision_type(rec)
    # Read hvac_action from recommendation (set by coordinator from TRV entity attributes).
    # Enables ACTIVE_HEATING classification without relying solely on temperature slope,
    # which can fall below the detection threshold in high-loss or slow-response rooms.
    hvac_action = rec.get("_hvac_action")
    return RuntimeCycleInput(
        zone_id=zone_id, ts=ts, target_c=target, trv_setpoint_c=setpoint, indoor_temp=indoor,
        schedule=ScheduleTarget(comfort_time_utc=schedule_comfort_time_utc,
                                comfort_temperature_c=schedule_comfort_temperature_c)
        if schedule_comfort_time_utc and schedule_comfort_temperature_c is not None else None,
        mode=rec.get("mode"), controller_demand=demand,
        controller_decision=ControllerDecisionInput(
            decision_type=decision_type, target_c=target, trv_setpoint_c=setpoint,
            boost_offset_c=boost_offset, boost_active=bool(rec.get("boost_active")),
            preheat_active=bool(rec.get("preheat_minutes", 0)),
            mode=rec.get("mode")),
        window_open=rec.get("window_open"), heating_failure=heating_failure,
        outdoor_temp=outdoor, forecast_high=rec.get("forecast_high"),
        device_states=dict(rec.get("device_states") or {}),   # B2b-4e live availability
        hvac_action=hvac_action,
        boost_dispatch=_boost_dispatch_from_live_record(live_record))


class LearningShadowController:
    """Home Assistant bridge for LE2 observation and adaptive-control
    suggestions, for one config entry / zone.

    Owns the prediction-only runtime and exposes safe, bounded adjustment
    helpers (e.g. boost/preheat suggestions). It does not dispatch a Home
    Assistant service call directly — the coordinator gates and applies any
    adjustment according to the effective adaptation mode
    (``ControlAdaptationMode`` in coordinator.py).

    Every public method is guarded: a learning/store/model failure is counted and
    logged (throttled), never raised. A failure here never breaks the
    deterministic TPI control path the coordinator falls back to.
    """

    def __init__(self, hass: Any, zone_id: str, *, store: Any = None,
                 mode: LearningRuntimeMode = LearningRuntimeMode.SHADOW,
                 clock: Optional[Clock] = None) -> None:
        self._zone = zone_id
        self._enabled = True
        self._errors = 0
        self._error_signatures: dict[str, int] = {}
        self._last_result = None
        # Clock injection: explicit clock required — no hidden wall-clock fallback.
        if clock is None:
            raise ValueError(
                "LearningShadowController requires an explicit Clock — "
                "pass coordinator._clock or a FakeClock for tests."
            )
        _clk: Clock = clock
        self._utcnow_iso: Callable[[], str] = lambda: _clk.now_utc().isoformat()
        adapter = store
        if adapter is None:
            try:
                adapter = HomeAssistantStoreAdapter(hass, _zone_segment(zone_id))
            except Exception:  # store construction must never break setup
                adapter = None
        self._runtime = LearningRuntime(
            LearningRuntimeConfig(mode=mode), store=adapter, clock=self._utcnow_iso,
            # Synchronous, memory-only episode hand-off (see
            # record_completed_episode_safe()) — bound now, but only ever
            # invoked later during observe_safe(), by which point every
            # attribute it touches (_episode_history etc.) already exists.
            episode_sink=self.record_completed_episode_safe)
        # The new decision pipeline runs read-only every cycle (no sink => it can never
        # dispatch). The existing coordinator remains the single real dispatch path.
        self._pipeline = DecisionPipeline(
            resolver=FinalResolver(boost_runtime_limit=TPI_MAX_BOOST_CELSIUS))
        self._last_trace: Optional[dict] = None
        # Adaptation candidate history — in-memory with persistent backing store.
        self._adaptation_history: dict = {}
        self._last_outcome_ts: Optional[str] = None
        self._adaptation_last_error: Optional[str] = None
        self._adaptation_save_needed: bool = False
        self._adaptation_store = None
        try:
            from ..storage.stores import AdaptationHistoryStore, HomeAssistantStoreFactory
            self._adaptation_store = AdaptationHistoryStore(
                HomeAssistantStoreFactory(hass), zone_id,
            )
        except Exception:
            pass  # non-fatal: history stays in-memory only
        # Application lifecycle state — in-memory with persistent backing store.
        # No real application in this layer; entries are added by a future Application Layer.
        self._application_lifecycle_state = None  # ApplicationLifecycleState | None
        self._application_lifecycle_save_needed: bool = False
        self._application_lifecycle_last_error: Optional[str] = None
        self._application_lifecycle_store = None
        try:
            from ..storage.stores import ApplicationLifecycleStore, HomeAssistantStoreFactory
            self._application_lifecycle_store = ApplicationLifecycleStore(
                HomeAssistantStoreFactory(hass), zone_id,
            )
        except Exception:
            pass  # non-fatal: lifecycle state stays in-memory only
        # Raw/episode capture storage — live-reachable store-access facade.
        # Construction itself performs zero storage I/O: it only holds a
        # StoreFactory reference plus the canonical raw-track/episode-type
        # registries. The returned handles ARE actively saved/loaded below
        # (episodes_store(), support_critical_events_store(), research_daily_store()).
        self._capture_stores = None
        try:
            from ..storage.capture_stores import LearningCaptureStores
            from ..storage.stores import HomeAssistantStoreFactory as _HAStoreFactory
            self._capture_stores = LearningCaptureStores(
                _HAStoreFactory(hass), zone_id,
            )
        except Exception:
            pass  # non-fatal: capture storage foundation stays unavailable
        # Episode history — in-memory with persistent backing store, reached via the
        # LearningCaptureStores facade above (EpisodesStore). Live-wired: the append
        # hook is record_completed_episode_safe() below, bound as LearningRuntime's
        # episode_sink and invoked from run_cycle()'s completed-episode loop.
        self._episode_history: dict = {}  # episode_id -> flat serialized episode entry
        self._episode_save_needed: bool = False
        self._episode_last_error: Optional[str] = None
        # Support Critical Events — in-memory with persistent backing store, reached
        # via the same LearningCaptureStores facade (SupportCriticalEventStore).
        # Storage/setup landmark events (store loaded/empty/load-failed,
        # save-failed/save-recovered) are recorded from within the load/save
        # methods themselves. A coordinator-path producer also exists:
        # coordinator.py's _maybe_record_support_hold_events() calls
        # record_support_critical_event_safe() below every cycle.
        self._support_critical_events: dict = {}  # event_id -> flat serialized support event entry
        self._support_critical_events_save_needed: bool = False
        # First time this dirty streak began (ISO ts); gates the debounce in
        # _async_save_support_critical_events_safe() so a bare setup-time
        # landmark doesn't immediately hit real storage. Cleared on save.
        self._support_critical_events_first_dirty_ts: Optional[str] = None
        self._support_critical_events_last_error: Optional[str] = None
        # Tracks whether a storage_save_failed landmark has already been recorded
        # for the CURRENT failure streak — prevents re-recording one every
        # periodic save attempt while the underlying problem persists, and gates
        # the single storage_save_recovered landmark on the next success.
        self._support_critical_events_save_failure_notified: bool = False
        # Research Daily Buckets — in-memory with persistent backing store, reached
        # via the same LearningCaptureStores facade (ResearchDailyStore). No
        # runtime/coordinator aggregation hook exists — nothing in
        # run_cycle()/coordinator calls record_research_daily_observation_safe()
        # or record_support_critical_event_safe() directly. As of this step,
        # record_support_critical_event_safe() itself aggregates a genuinely
        # newly-appended Support Critical Event into the matching day's bucket
        # (see _maybe_aggregate_research_daily_from_support_event()) — pure
        # observation of an already-produced event, still no new producer and
        # still no runtime/control path involved.
        self._research_daily_buckets: dict = {}  # bucket_date -> flat serialized bucket entry
        self._research_daily_save_needed: bool = False
        # First time this dirty streak began (ISO ts); gates the debounce in
        # _async_save_research_daily_safe() — same rationale as the support
        # critical events counterpart above. Cleared on save.
        self._research_daily_first_dirty_ts: Optional[str] = None
        self._research_daily_last_error: Optional[str] = None
        # Cache: schedule target time from last observe_safe call (one cycle lag is acceptable
        # for deescalation checks — TPI sufficiency uses this to compute remaining_time_to_target).
        self._last_comfort_time_utc: Optional[str] = None
        # Context snapshot at boost-apply time: compared each cycle to detect schedule transitions.
        # A change in comfort_time (same target, new period) triggers a hard schedule_change release.
        self._boost_applied_comfort_time_utc: Optional[str] = None
        # Note: _boost_authorized_this_cycle mutable flag removed in favour of direct
        # authorized_override parameter on adjust_recommendation_safe() (structural refactor).

    @property
    def runtime(self) -> LearningRuntime:
        return self._runtime

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def capture_stores(self):
        """Return the LearningCaptureStores wiring foundation, or None.

        Write-nothing accessor object — see capture_stores.py. Present so a
        future capture/persist step has one clear, already-tested attachment
        point; nothing in this class calls it yet.
        """
        return self._capture_stores

    async def async_setup(self) -> bool:
        try:
            setup_ok = await self._runtime.async_setup()
        except Exception as err:  # learning setup failure must not fail the entry
            self._record_error("setup", err)
            self._enabled = False
            return False
        # Load persisted adaptation history (non-fatal; errors in _adaptation_last_error)
        await self._async_load_adaptation_history_safe()
        # Load persisted application lifecycle state (non-fatal; no real entries yet)
        await self._async_load_application_lifecycle_safe()
        # Load persisted episode history (non-fatal; no runtime writer yet)
        await self._async_load_episode_history_safe()
        # Load persisted support critical events (non-fatal; no runtime writer yet)
        await self._async_load_support_critical_events_safe()
        # Load persisted research daily buckets (non-fatal)
        await self._async_load_research_daily_safe()
        return setup_ok

    def observe_safe(self, recommendation: Mapping[str, Any], *,
                     weather: Optional[Mapping[str, Any]] = None,
                     schedule_comfort_time_utc: Optional[str] = None,
                     schedule_comfort_temperature_c: Optional[float] = None,
                     heating_failure: bool = False,
                     live_record: Any = None) -> None:
        """Run one prediction-only observation cycle. Never raises; does not
        dispatch a Home Assistant service call.

        ``live_record`` is the authoritative post-dispatch ``LiveDecisionRecord``;
        when provided, the boost outcome context is bound to its real dispatch result
        (B2b-1) rather than the setpoint-target heuristic.
        """
        if not self._enabled:
            return
        # Cache comfort time for TPI-sufficiency check in next adjust_recommendation_safe call.
        self._last_comfort_time_utc = schedule_comfort_time_utc
        try:
            inp = build_runtime_cycle_input(
                self._zone, recommendation, weather=weather,
                schedule_comfort_time_utc=schedule_comfort_time_utc,
                schedule_comfort_temperature_c=schedule_comfort_temperature_c,
                heating_failure=heating_failure, live_record=live_record,
                ts_iso=self._utcnow_iso())
            self._last_result = CoordinatorBridge(self._runtime).process(inp)
        except Exception as err:
            self._record_error("cycle", err)
        # Passive adaptation history — post-cycle, never raises, never affects heating.
        try:
            self._update_adaptation_history_safe(recommendation, weather)
        except Exception as err:
            try:
                self._adaptation_last_error = str(err)
            except Exception:
                pass

    def _update_adaptation_history_safe(
        self,
        recommendation: Any,
        weather: Optional[Any],
    ) -> None:
        """Update in-memory adaptation history from current outcome model state.

        Skips if outcome model has not advanced since last call (deduplication via
        last_update_ts). Never mutates control state. Never raises — all errors are
        captured in _adaptation_last_error.
        """
        try:
            zr = self._runtime._zone(self._zone)
            _om = zr.orchestrator.models.get("outcome")
            if _om is None:
                return
            diag = _om.diagnostics()
            new_ts = diag.last_update_ts
            if not new_ts or new_ts == self._last_outcome_ts:
                return

            # Build OutcomeSignal from diagnostics
            from ..adaptation.contracts import OutcomeSignal, SituationContext
            full_count, partial_count = diag.full_partial
            total_accepted = full_count + partial_count
            partial_ratio = (
                (partial_count / total_accepted) if total_accepted > 0 else 0.0
            )
            rej_counts = dict(diag.rejection_counts)
            total_rej = sum(rej_counts.values())
            signal = OutcomeSignal(
                sample_count=diag.sample_counts.get("general", 0),
                timeout_rate=diag.timeout_rate,
                overshoot_rate=diag.overshoot_rate,
                reached_rate=diag.reached_rate,
                general_data_quality=diag.general_data_quality,
                aggregate_reliability=getattr(diag, "confidence", 0.0),
                partial_ratio=partial_ratio,
                confounder_contamination=total_rej > 0,
            )

            # Build SituationContext — similar to export._adaptation_situation_context
            # but without coordinator/coord; reads from recommendation and weather args.
            _OUTDOOR_EDGES = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
            outdoor_bucket: Optional[str] = None
            if weather:
                try:
                    ot_raw = weather.get("temperature") if hasattr(weather, "get") else None
                    if ot_raw is not None:
                        ot = float(ot_raw)
                        idx = sum(1 for edge in _OUTDOOR_EDGES if ot >= edge)
                        outdoor_bucket = f"b{idx}"
                except Exception:
                    pass

            mode_context: Optional[str] = None
            preheat_was_active: Optional[bool] = None
            if recommendation:
                try:
                    mode_context = recommendation.get("mode") or None
                except Exception:
                    pass
                try:
                    pa = recommendation.get("preheat_active")
                    preheat_was_active = bool(pa) if pa is not None else None
                except Exception:
                    pass

            controller_kind: Optional[str] = None
            try:
                _state = getattr(_om, "_state", None)
                _samples = getattr(_state, "recent_samples", ()) or ()
                if _samples:
                    controller_kind = getattr(_samples[-1], "controller_kind", None)
            except Exception:
                pass

            time_of_day_bucket: Optional[int] = None
            weekday: Optional[int] = None
            is_weekend: Optional[bool] = None
            context_time_source = "unavailable"
            try:
                from datetime import datetime
                _ts = datetime.fromisoformat(new_ts.replace("Z", "+00:00"))
                time_of_day_bucket = _ts.hour
                weekday = _ts.weekday()
                is_weekend = weekday >= 5
                context_time_source = "model_last_update"
            except Exception:
                pass

            situation = SituationContext(
                controller_kind=controller_kind,
                outdoor_bucket=outdoor_bucket,
                mode_context=mode_context,
                time_of_day_bucket=time_of_day_bucket,
                weekday=weekday,
                is_weekend=is_weekend,
                preheat_was_active=preheat_was_active,
                boost_was_active=None,
                target_delta_c=None,
                heat_loss_c_per_h=None,
                preheat_minutes_used=None,
                active_control=None,
                learning_enabled=None,
                context_time_source=context_time_source,
            )

            from ..adaptation.history import update_adaptation_history
            self._adaptation_history = update_adaptation_history(
                self._adaptation_history,
                self._zone,
                signal,
                situation,
                new_ts,
            )
            self._last_outcome_ts = new_ts
            self._adaptation_last_error = None
            self._adaptation_save_needed = True
        except Exception as err:
            self._adaptation_last_error = str(err)

    def adaptation_history_snapshot(self) -> dict:
        """Return a shallow copy of the current in-memory adaptation history.

        Keys are candidate_key strings; values are CandidateHistoryEntry instances.
        Empty on first call (reset on HA restart). Never raises.
        """
        try:
            return dict(self._adaptation_history)
        except Exception:
            return {}

    def adaptation_last_error(self) -> Optional[str]:
        """Return the last error message from adaptation history accumulation, or None."""
        return self._adaptation_last_error

    async def _async_load_adaptation_history_safe(self) -> None:
        """Load persisted adaptation history. Non-fatal on missing or corrupt store."""
        if self._adaptation_store is None:
            return
        try:
            raw = await self._adaptation_store.load()
            if raw is None:
                return
            from ..adaptation.history_store_schema import deserialize_history_state
            state = deserialize_history_state(raw, self._zone)
            if state.entries:
                self._adaptation_history = dict(state.entries)
        except Exception as err:
            self._adaptation_last_error = str(err)

    async def _async_save_adaptation_history_safe(self) -> None:
        """Persist adaptation history best-effort. Non-fatal on error."""
        if self._adaptation_store is None or not self._adaptation_save_needed:
            return
        try:
            from ..adaptation.history_store_schema import (
                AdaptationHistoryState, prune_history_entries, serialize_history_state,
            )
            now_ts = self._utcnow_iso()
            entries = prune_history_entries(self._adaptation_history, now_ts=now_ts)
            state = AdaptationHistoryState(
                learning_zone_id=self._zone,
                updated_at=now_ts,
                entries=entries,
            )
            await self._adaptation_store.save(serialize_history_state(state))
            self._adaptation_save_needed = False
        except Exception as err:
            self._adaptation_last_error = str(err)

    async def _async_load_application_lifecycle_safe(self) -> None:
        """Load persisted application lifecycle state. Non-fatal on missing or corrupt store."""
        if self._application_lifecycle_store is None:
            return
        try:
            from ..adaptation.application_state import (
                ApplicationLifecycleState, deserialize_application_lifecycle_state,
            )
            raw = await self._application_lifecycle_store.load()
            if raw is None:
                # No persisted data yet — initialize empty state so the Application Layer
                # can add entries without having to guard against None first.
                self._application_lifecycle_state = ApplicationLifecycleState(
                    learning_zone_id=self._zone, updated_at="", entries={},
                )
                return
            state = deserialize_application_lifecycle_state(raw, self._zone)
            self._application_lifecycle_state = state
        except Exception as err:
            self._application_lifecycle_last_error = str(err)

    async def _async_save_application_lifecycle_safe(self) -> None:
        """Persist application lifecycle state best-effort. Non-fatal on error.

        Dirty flag remains True when save fails, ensuring the next opportunity retries.
        """
        if (self._application_lifecycle_store is None
                or not self._application_lifecycle_save_needed):
            return
        try:
            if self._application_lifecycle_state is None:
                return
            from ..adaptation.application_state import serialize_application_lifecycle_state
            await self._application_lifecycle_store.save(
                serialize_application_lifecycle_state(self._application_lifecycle_state)
            )
            self._application_lifecycle_save_needed = False
        except Exception as err:
            self._application_lifecycle_last_error = str(err)
            # dirty flag remains — will retry on next save opportunity

    def application_lifecycle_snapshot(self) -> Optional[dict]:
        """Return a summary dict of the current application lifecycle state, or None.

        Returns None when no state has been loaded yet (e.g. first run, no entries).
        Never raises. Read-only — does not modify state.
        """
        try:
            if self._application_lifecycle_state is None:
                return None
            from ..adaptation.application_state import summarize_application_lifecycle_state
            return summarize_application_lifecycle_state(
                self._application_lifecycle_state
            ).to_dict()
        except Exception:
            return None

    async def _async_load_episode_history_safe(self) -> None:
        """Load persisted episode history via LearningCaptureStores.episodes_store().

        Non-fatal on missing/corrupt store. Each stored entry is validated with
        deserialize_episode() (from episode_serialization.py) before being kept —
        a malformed or schema-mismatched entry is silently dropped, never crashes
        the load. No runtime writer populates this store yet in this step.
        """
        if self._capture_stores is None:
            return
        try:
            store = self._capture_stores.episodes_store()
            raw = await store.load()
            if raw is None:
                return
            raw_entries = raw.get("episodes") if isinstance(raw, Mapping) else None
            if not isinstance(raw_entries, Mapping):
                return
            from ..storage.episode_serialization import deserialize_episode
            rebuilt: dict = {}
            for eid, entry in raw_entries.items():
                if not isinstance(entry, Mapping):
                    continue  # malformed -> skip
                if deserialize_episode(entry) is None:
                    continue  # unknown type / schema mismatch / malformed -> skip
                rebuilt[eid] = dict(entry)
            self._episode_history = rebuilt
        except Exception as err:
            self._episode_last_error = str(err)

    async def _async_save_episode_history_safe(self) -> None:
        """Persist episode history best-effort. Non-fatal on error.

        Dirty flag remains True when save fails, ensuring the next opportunity
        retries. Nothing in this step ever sets _episode_save_needed — this
        method exists so a future runtime append hook can rely on an
        already-tested save path via the existing periodic save trigger.
        """
        if self._capture_stores is None or not self._episode_save_needed:
            return
        try:
            store = self._capture_stores.episodes_store()
            await store.save({"episodes": dict(self._episode_history)})
            self._episode_save_needed = False
        except Exception as err:
            self._episode_last_error = str(err)
            # dirty flag remains — will retry on next save opportunity

    def episode_history_snapshot(self) -> dict:
        """Return a shallow copy of the current in-memory episode history.

        Keys are episode_id strings; values are flat, versioned serialized
        episode entry dicts (see episode_serialization.py). Empty until a
        future runtime append hook and/or a successful store load populate it.
        Never raises.
        """
        try:
            return dict(self._episode_history)
        except Exception:
            return {}

    def episode_last_error(self) -> Optional[str]:
        """Return the last error message from episode history load/save, or None."""
        return self._episode_last_error

    def record_completed_episode_safe(self, episode: Any) -> None:
        """Synchronous, memory-only hand-off for one completed episode.

        Bound as LearningRuntime's ``episode_sink`` — called from
        ``run_cycle()``'s completed-episode loop. Never performs storage I/O
        itself: appends into the in-memory ``_episode_history`` via the pure
        ``append_completed_episode()`` (which also applies that episode
        type's own RetentionPolicy immediately, from
        ``self._capture_stores.episode_registry`` — no new/arbitrary
        retention numbers). ``_episode_save_needed`` is set only when an
        entry was actually appended and/or pruned — a duplicate
        ``episode_id`` alone does not mark it dirty. The next
        ``async_save_if_due()`` call flushes it; nothing here saves directly.

        Also observes (never influences) whether this was a genuinely new,
        completed OutcomeEpisode and, if so, records ONE Support Critical
        Event via ``_maybe_record_outcome_resolved_event()`` — see that
        method's own docstring.

        Never raises: a missing capture-store foundation, an unrecognised/
        malformed episode, or any internal failure is captured in
        ``_episode_last_error`` and swallowed — never propagated, never
        touches ``_enabled``.
        """
        if self._capture_stores is None:
            return
        try:
            from datetime import datetime
            from ..storage.episode_persistence import append_completed_episode
            now_utc = datetime.fromisoformat(self._utcnow_iso())
            result = append_completed_episode(
                {"episodes": self._episode_history},
                episode,
                episode_registry=self._capture_stores.episode_registry,
                now_utc=now_utc,
            )
            if result.appended_episode_ids or result.pruned_episode_ids:
                self._episode_history = result.updated_payload["episodes"]
                self._episode_save_needed = True
            self._maybe_record_outcome_resolved_event(episode, result)
        except Exception as err:
            self._episode_last_error = str(err)

    def _maybe_record_outcome_resolved_event(self, episode: Any, result: Any) -> None:
        """Record ONE outcome_resolved Support Critical Event when a
        completed OutcomeEpisode was genuinely newly appended this call —
        pure observation of the ALREADY-BUILT episode object handed to
        ``record_completed_episode_safe()`` (for outcome episodes this is
        the confounder-augmented ``bound_episode`` lifecycle.py's
        ``run_cycle()`` already constructs — see that module's own comments;
        never the raw episode). Never influences episode construction,
        outcome scoring, retention, or Learning Progress — read-only access
        to fields the ``OutcomeEpisode`` dataclass already carries.

        Not produced for non-outcome episode types (heating/afterheat/
        passive_cooling/window_cooling) — checked via ``isinstance``, not a
        new classification.

        Deduped by reusing the SAME episode_id-based dedup
        ``append_completed_episode()`` already performs: fires only when
        ``episode.episode_id`` is present in ``result.appended_episode_ids``
        (a genuinely new append this call) — a duplicate resubmission of the
        same completed outcome is a no-op here too, via already-tested
        infrastructure rather than a new dedup mechanism.
        ``episode.episode_id``/``episode.learning_zone_id``/
        ``episode.decision_id`` are used only for this internal membership
        check (or not at all) — never placed into the event itself.

        Never raises: any failure here is swallowed — a missing landmark is
        strictly less important than the episode persistence it decorates.
        """
        try:
            from ..episode_schemas import OutcomeEpisode
            if not isinstance(episode, OutcomeEpisode):
                return
            if episode.episode_id not in result.appended_episode_ids:
                return  # duplicate resubmission, or not newly appended this call

            from datetime import datetime
            from ..support_event_schemas import (
                SupportCriticalEvent, SupportEventSeverity, SupportEventType,
            )
            from ..storage.support_event_serialization import SUPPORT_EVENT_SCHEMA_VERSION

            confounder_count = len(episode.confounder_flags)
            details: dict = {
                "regime": episode.regime.value,
                "target_reached": episode.reason.value == "reached",
                "reason": episode.reason.value,
                "confounded": confounder_count > 0,
                "confounder_count": confounder_count,
            }
            try:
                delta = float(episode.end_temp) - float(episode.target)
                details["overshoot_c"] = round(max(0.0, delta), 2)
                details["undershoot_c"] = round(max(0.0, -delta), 2)
                details["comfort_error_c"] = round(abs(delta), 2)
            except (TypeError, ValueError):
                pass  # optional derived fields only; core details above remain valid

            now_utc = datetime.fromisoformat(self._utcnow_iso())
            # Hashed (not raw) episode_id: guarantees a distinct, deterministic
            # support-event id per underlying outcome episode — a plain
            # timestamp alone can collide when two outcomes resolve within the
            # same millisecond (or under a frozen test clock), which would
            # make append_support_event()'s own event_id dedup incorrectly
            # treat two DIFFERENT outcomes as the same event. Never exposed
            # raw or otherwise — only this one-way hash leaves the method.
            _episode_id_hash = hashlib.sha256(episode.episode_id.encode()).hexdigest()[:16]
            event_id = f"outcome_resolved_{_episode_id_hash}"
            event = SupportCriticalEvent(
                schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id=event_id,
                event_type=SupportEventType.OUTCOME_RESOLVED, ts=now_utc,
                severity=SupportEventSeverity.INFO,
                reason=episode.reason.value, summary="Outcome resolved", details=details,
            )
            self.record_support_critical_event_safe(event)
        except Exception:
            pass  # event production is best-effort; must never affect episode persistence

    # 6h — avoids flooding the 48h critical-event timeline with a repeated
    # "store loaded"/"store empty"/"load failed" landmark on every HA restart
    # when restarts happen frequently (e.g. during setup/troubleshooting).
    _STORAGE_RESTORE_LANDMARK_DEDUPE_WINDOW_S = 6 * 3600

    # Debounce for support-critical-event / research-daily saves, mirroring
    # PersistenceOrchestrator's default debounce_s. Without this, a pure
    # setup-time "store loaded (empty)" landmark would go straight to real
    # `.storage` on the very first coordinator refresh — an empty fresh
    # setup must not write LE2 store files. `force=True` (unload/shutdown)
    # always bypasses this so genuinely accumulated data is never lost.
    _SUPPORT_RESEARCH_SAVE_DEBOUNCE_S = 30.0

    def _save_debounce_elapsed(self, first_dirty_ts: Optional[str]) -> bool:
        """True once ``first_dirty_ts`` is at least the debounce window old.

        Fails open (returns True) on a missing/unparseable timestamp so a
        genuinely dirty state is never stuck unsaved due to a clock glitch.
        """
        if first_dirty_ts is None:
            return True
        try:
            from datetime import datetime
            now_utc = datetime.fromisoformat(self._utcnow_iso())
            first_utc = datetime.fromisoformat(first_dirty_ts)
        except Exception:
            return True
        return (now_utc - first_utc).total_seconds() >= self._SUPPORT_RESEARCH_SAVE_DEBOUNCE_S

    def _record_storage_landmark_event_safe(
        self, *, event_type: Any, severity: Any, reason: str, summary: str,
        details: Mapping[str, Any], dedupe_window_s: Optional[float] = None,
    ) -> None:
        """Build and record ONE storage/setup landmark Support Critical Event.

        Pure hand-off to record_support_critical_event_safe() — never touches
        storage itself (no save triggered here), so this can never recurse
        into the save path it may be called from. When ``dedupe_window_s`` is
        given, skips recording if an event with the same (event_type, reason)
        already exists in the current in-memory snapshot within that window
        — used for the load-outcome landmarks, which would otherwise repeat
        once per HA restart. Save-failure/-recovery landmarks pass no window
        here; they are already deduped by the caller's own state flag
        (``_support_critical_events_save_failure_notified``), which is a
        precise state-transition signal rather than a time window. Never
        raises: any failure here is swallowed — a missing landmark is
        strictly less important than the load/save operation it decorates.
        """
        try:
            from datetime import datetime
            from .. import support_event_schemas as _schemas
            from ..storage.support_event_serialization import SUPPORT_EVENT_SCHEMA_VERSION

            now_utc = datetime.fromisoformat(self._utcnow_iso())
            event_id = f"storage_landmark_{event_type.value}_{int(now_utc.timestamp() * 1000)}"
            candidate = _schemas.SupportCriticalEvent(
                schema_version=SUPPORT_EVENT_SCHEMA_VERSION, event_id=event_id,
                event_type=event_type, ts=now_utc, severity=severity,
                reason=reason, summary=summary, details=dict(details),
            )
            if dedupe_window_s is not None:
                from ..storage.support_event_persistence import is_recent_duplicate
                recent = list(self._support_critical_events.values())
                if is_recent_duplicate(candidate, recent, window_s=dedupe_window_s):
                    return
            self.record_support_critical_event_safe(candidate)
        except Exception:
            pass  # landmark recording is best-effort; never affects load/save itself

    async def _async_load_support_critical_events_safe(self) -> None:
        """Load persisted support critical events via
        LearningCaptureStores.support_critical_events_store().

        Non-fatal on missing/corrupt store. Each stored entry is validated with
        deserialize_support_event() (from support_event_serialization.py)
        before being kept — a malformed or schema-mismatched entry is silently
        dropped (counted), never crashes the load.

        Records a storage/setup landmark event only for a genuine restore of
        existing data or a load failure — never for an empty/nonexistent
        store, so a fresh setup (nothing ever saved yet) does not itself
        create a new dirty Support Critical Event that a subsequent
        force-flushed unload would persist. Deduped against the just-loaded
        snapshot so frequent HA restarts don't flood the 48h timeline with
        repeated identical landmarks.
        """
        if self._capture_stores is None:
            return
        from .. import support_event_schemas as _schemas

        records_loaded = 0
        malformed_skipped = 0
        load_error: Optional[Exception] = None
        try:
            store = self._capture_stores.support_critical_events_store()
            raw = await store.load()
            if raw is not None:
                raw_entries = raw.get("events") if isinstance(raw, Mapping) else None
                if isinstance(raw_entries, Mapping):
                    from ..storage.support_event_serialization import deserialize_support_event
                    rebuilt: dict = {}
                    for eid, entry in raw_entries.items():
                        if not isinstance(entry, Mapping):
                            malformed_skipped += 1
                            continue  # malformed -> skip
                        if deserialize_support_event(entry) is None:
                            malformed_skipped += 1
                            continue  # unknown type / schema mismatch / malformed -> skip
                        rebuilt[eid] = dict(entry)
                    self._support_critical_events = rebuilt
                    records_loaded = len(rebuilt)
        except Exception as err:
            self._support_critical_events_last_error = str(err)
            load_error = err

        if load_error is not None:
            self._record_storage_landmark_event_safe(
                event_type=_schemas.SupportEventType.STORAGE_RESTORE_FAILED,
                severity=_schemas.SupportEventSeverity.WARNING,
                reason="support_events_load_failed",
                summary="Support critical event store failed to load",
                details={"error_type": type(load_error).__name__},
                dedupe_window_s=self._STORAGE_RESTORE_LANDMARK_DEDUPE_WINDOW_S,
            )
        elif records_loaded > 0:
            self._record_storage_landmark_event_safe(
                event_type=_schemas.SupportEventType.STORAGE_RESTORE,
                severity=_schemas.SupportEventSeverity.INFO,
                reason="support_events_loaded",
                summary="Support critical event store loaded",
                details={"records_loaded": records_loaded, "malformed_skipped": malformed_skipped},
                dedupe_window_s=self._STORAGE_RESTORE_LANDMARK_DEDUPE_WINDOW_S,
            )

    async def _async_save_support_critical_events_safe(self, *, force: bool = False) -> None:
        """Persist support critical events best-effort. Non-fatal on error.

        Debounced like PersistenceOrchestrator (``_SUPPORT_RESEARCH_SAVE_DEBOUNCE_S``):
        a fresh dirty streak only reaches real storage once it has been
        pending for at least the debounce window, so a bare setup-time
        "store loaded (empty)" landmark does not itself write a `.storage`
        file on the very first coordinator refresh. ``force=True``
        (unload/shutdown) always bypasses the debounce so genuinely
        accumulated data is never lost.

        Dirty flag remains True when save fails, ensuring the next opportunity
        retries. On the FIRST failure of a failure streak, records one
        storage_save_failed landmark and sets
        ``_support_critical_events_save_failure_notified`` — subsequent
        failures in the same streak record nothing further (no per-cycle
        spam). On the next successful save after a notified failure, records
        one storage_save_recovered landmark and clears the flag. Recording
        a landmark here never triggers a second save within this call — it
        only marks ``_support_critical_events_save_needed`` dirty again for
        the NEXT periodic save opportunity, exactly like any other append.
        """
        if self._capture_stores is None or not self._support_critical_events_save_needed:
            return
        if not force and not self._save_debounce_elapsed(
                self._support_critical_events_first_dirty_ts):
            return
        from .. import support_event_schemas as _schemas
        try:
            store = self._capture_stores.support_critical_events_store()
            await store.save({"events": dict(self._support_critical_events)})
            self._support_critical_events_save_needed = False
            self._support_critical_events_first_dirty_ts = None
            if self._support_critical_events_save_failure_notified:
                self._support_critical_events_save_failure_notified = False
                self._record_storage_landmark_event_safe(
                    event_type=_schemas.SupportEventType.STORAGE_SAVE_RECOVERED,
                    severity=_schemas.SupportEventSeverity.INFO,
                    reason="support_events_save_recovered",
                    summary="Support critical event store save recovered",
                    details={},
                )
        except Exception as err:
            self._support_critical_events_last_error = str(err)
            # dirty flag remains — will retry on next save opportunity
            if not self._support_critical_events_save_failure_notified:
                self._support_critical_events_save_failure_notified = True
                self._record_storage_landmark_event_safe(
                    event_type=_schemas.SupportEventType.STORAGE_SAVE_FAILED,
                    severity=_schemas.SupportEventSeverity.WARNING,
                    reason="support_events_save_failed",
                    summary="Support critical event store failed to save",
                    details={"error_type": type(err).__name__},
                )

    def support_critical_events_snapshot(self) -> dict:
        """Return a shallow copy of the current in-memory support critical events.

        Keys are event_id strings; values are flat, versioned serialized
        support event entry dicts (see support_event_serialization.py). Empty
        until a future runtime producer and/or a successful store load
        populate it. Never raises.
        """
        try:
            return dict(self._support_critical_events)
        except Exception:
            return {}

    def support_critical_events_last_error(self) -> Optional[str]:
        """Return the last error message from support event load/save, or None."""
        return self._support_critical_events_last_error

    def record_support_critical_event_safe(self, event: Any) -> None:
        """Synchronous, memory-only hand-off for one support critical event.

        NOT bound anywhere yet — no runtime/coordinator call site invokes this
        in this step (see support_event_persistence.py's module docstring for
        the exact reasoning and next-step hook points). Never performs
        storage I/O itself: appends into the in-memory
        ``_support_critical_events`` via the pure ``append_support_event()``
        (which also applies the shared SUPPORT_EVENT_RETENTION_DEFAULT policy
        immediately — no new/arbitrary retention numbers).
        ``_support_critical_events_save_needed`` is set only when an entry
        was actually appended and/or pruned — a duplicate ``event_id`` alone
        does not mark it dirty. The next ``async_save_if_due()`` call flushes
        it; nothing here saves directly.

        Never raises: a missing capture-store foundation, an unrecognised/
        malformed event, or any internal failure is captured in
        ``_support_critical_events_last_error`` and swallowed — never
        propagated, never touches ``_enabled``.
        """
        if self._capture_stores is None:
            return
        try:
            from datetime import datetime
            from ..storage.support_event_persistence import append_support_event
            now_utc = datetime.fromisoformat(self._utcnow_iso())
            result = append_support_event(
                {"events": self._support_critical_events},
                event,
                now_utc=now_utc,
            )
            if result.appended_event_ids or result.pruned_event_ids:
                self._support_critical_events = result.updated_payload["events"]
                if not self._support_critical_events_save_needed:
                    self._support_critical_events_first_dirty_ts = self._utcnow_iso()
                self._support_critical_events_save_needed = True
            if event.event_id in result.appended_event_ids:
                # Only aggregate a GENUINELY newly appended event — a
                # duplicate (deduped by append_support_event()'s own
                # event_id check) or a rejected/malformed event (never
                # reaches here; append_support_event() would have skipped
                # it) must never inflate the daily counts. Failure here must
                # never fail Support Event recording itself — fully isolated
                # try/except, in addition to the one already inside
                # _maybe_aggregate_research_daily_from_support_event().
                try:
                    self._maybe_aggregate_research_daily_from_support_event(event)
                except Exception as err:
                    self._research_daily_last_error = str(err)
        except Exception as err:
            self._support_critical_events_last_error = str(err)

    # SupportEventType -> ResearchDailyBucket counter field, for the simple
    # "one event -> +1 one field" cases. OUTCOME_RESOLVED is handled
    # separately below (needs multiple fields from its details).
    #
    # Deliberately NOT mapped in this step (documented, not an oversight):
    #   - Storage/setup landmark events (STORAGE_RESTORE(_FAILED),
    #     STORAGE_SAVE_FAILED, STORAGE_SAVE_RECOVERED, RESTART_RESTORE):
    #     these measure store health, not heating/learning quality — kept
    #     out of Research Daily v1 per explicit product decision.
    #   - LEARNING_RECOMMENDATION_ONLY: semantically distinct from
    #     TRV_COMMAND_BLOCKED (no field exists for it, and folding it into
    #     trv_command_blocked_count would conflate "no command was even
    #     attempted because adaptive control wasn't authorized" with "a
    #     command was attempted and vetoed") — deliberately left unmapped
    #     rather than misclassified.
    #   - HEATING_DECISION, WINDOW_CLOSED_RELEASE, MANUAL_OVERRIDE_END,
    #     SCHEDULE_CHANGE, PRESENCE_HOLD, TEMPERATURE_INVALID,
    #     MIN_INTERVAL_BLOCK, LEARNING_ADAPTIVE_APPLIED: no live producer
    #     exists yet for these types (see support_event_schemas.py) or no
    #     corresponding daily field exists — nothing to map.
    #   - decision_count/heating_allowed_count/heating_blocked_count: real
    #     per-decision counts should come from a future Coordinator/
    #     Decision-Record source, not be estimated from Support Events.
    _SUPPORT_EVENT_TYPE_TO_DAILY_FIELD = {
        "trv_command_sent": "trv_command_sent_count",
        "trv_command_blocked": "trv_command_blocked_count",
        "same_setpoint_block": "same_setpoint_block_count",
        "trv_unavailable": "trv_unavailable_count",
        "window_open_hold": "window_hold_count",
        "summer_mode_hold": "summer_hold_count",
        "manual_override_start": "manual_override_count",
        "boost_started": "boost_started_count",
        "boost_blocked": "boost_blocked_count",
        "boost_ended": "boost_ended_count",
        "sensor_unavailable": "sensor_unavailable_count",
        "sensor_restored": "sensor_restored_count",
        "fallback_used": "fallback_used_count",
    }

    def _maybe_aggregate_research_daily_from_support_event(self, event: Any) -> None:
        """Aggregate ONE genuinely newly-appended Support Critical Event into
        the matching day's Research Daily Bucket.

        Called ONLY from ``record_support_critical_event_safe()``, and only
        for an event whose ``event_id`` is in that call's
        ``result.appended_event_ids`` — never for a duplicate/rejected
        event. This is pure aggregation of an already-produced, already
        public-safe event; it creates NO new Support Event, changes no
        Support Event semantics, and never touches the Support/Research
        Export layout.

        Bucket date is derived from ``event.ts`` (``YYYY-MM-DD``, UTC) — a
        malformed/non-datetime ``ts`` silently skips aggregation (no system-
        clock fallback, never raises). Only fixed, pre-defined scalar
        aggregate fields are ever read from ``event.details`` (never the
        whole dict, never entity/zone/episode ids, never raw events) — see
        ``research_daily_schemas.py``'s module docstring for why Research
        Daily Buckets need no free-form details mapping at all.

        Never raises: any failure is left for the caller to catch (this
        method itself, and ``record_research_daily_observation_safe()``
        beneath it, are both already exception-safe, but the caller wraps
        this call in its own try/except as well, so a genuinely unexpected
        error here can never fail Support Event recording).
        """
        from ..research_daily_schemas import ResearchDailyObservation

        try:
            from datetime import timezone
            bucket_date = event.ts.astimezone(timezone.utc).date().isoformat()
        except (AttributeError, ValueError, OverflowError):
            return  # malformed/non-datetime ts -> no aggregation, no system-clock fallback

        event_type_value = getattr(event.event_type, "value", None)
        details = event.details if isinstance(event.details, Mapping) else {}

        kwargs: dict = {"bucket_date": bucket_date}
        field_name = self._SUPPORT_EVENT_TYPE_TO_DAILY_FIELD.get(event_type_value)
        if field_name is not None:
            kwargs[field_name] = 1
        elif event_type_value == "outcome_resolved":
            kwargs["outcome_resolved_count"] = 1
            target_reached = details.get("target_reached")
            if target_reached is True:
                kwargs["outcome_success_count"] = 1
            elif target_reached is False:
                kwargs["outcome_failed_count"] = 1
            if details.get("confounded") is True:
                kwargs["outcome_confounded_count"] = 1
            for detail_key in ("overshoot_c", "undershoot_c", "comfort_error_c"):
                value = details.get(detail_key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    kwargs[detail_key] = float(value)
        else:
            return  # not mapped in this step (see the field-table comment above)

        observation = ResearchDailyObservation(**kwargs)
        self.record_research_daily_observation_safe(observation)

    async def _async_load_research_daily_safe(self) -> None:
        """Load persisted research daily buckets via
        LearningCaptureStores.research_daily_store().

        Non-fatal on missing/corrupt store. Each stored entry is validated
        with deserialize_research_daily_bucket() (from
        research_daily_serialization.py) before being kept — a malformed or
        schema-mismatched entry is silently dropped, never crashes the load.
        On a store-load exception, the error is recorded in
        ``_research_daily_last_error`` and the state stays empty — no
        exception ever propagates, ``_enabled`` is never touched.
        """
        if self._capture_stores is None:
            return
        try:
            store = self._capture_stores.research_daily_store()
            raw = await store.load()
            if raw is None:
                return
            raw_entries = raw.get("buckets") if isinstance(raw, Mapping) else None
            if not isinstance(raw_entries, Mapping):
                return
            from ..storage.research_daily_serialization import deserialize_research_daily_bucket
            rebuilt: dict = {}
            for bucket_date, entry in raw_entries.items():
                if not isinstance(entry, Mapping):
                    continue  # malformed -> skip
                if deserialize_research_daily_bucket(entry) is None:
                    continue  # unknown schema version / malformed -> skip
                rebuilt[bucket_date] = dict(entry)
            self._research_daily_buckets = rebuilt
        except Exception as err:
            self._research_daily_last_error = str(err)

    async def _async_save_research_daily_safe(self, *, force: bool = False) -> None:
        """Persist research daily buckets best-effort. Non-fatal on error.

        Only writes when a capture-store foundation exists AND
        ``_research_daily_save_needed`` is True. Debounced like
        ``_async_save_support_critical_events_safe()`` (same
        ``_SUPPORT_RESEARCH_SAVE_DEBOUNCE_S`` window) so a bare setup-time
        aggregation does not itself write a `.storage` file on the very
        first coordinator refresh; ``force=True`` (unload/shutdown) always
        bypasses this. Dirty flag remains True when save fails, ensuring
        the next periodic save opportunity retries; success clears it.
        Never raises.
        """
        if self._capture_stores is None or not self._research_daily_save_needed:
            return
        if not force and not self._save_debounce_elapsed(self._research_daily_first_dirty_ts):
            return
        try:
            store = self._capture_stores.research_daily_store()
            await store.save({"buckets": dict(self._research_daily_buckets)})
            self._research_daily_save_needed = False
            self._research_daily_first_dirty_ts = None
        except Exception as err:
            self._research_daily_last_error = str(err)
            # dirty flag remains — will retry on next save opportunity

    def research_daily_snapshot(self) -> dict:
        """Return a shallow copy of the current in-memory research daily buckets.

        Keys are bucket_date strings (YYYY-MM-DD); values are flat, versioned
        serialized bucket entry dicts (see research_daily_serialization.py).
        Empty until a future runtime aggregation hook and/or a successful
        store load populate it. Never raises.
        """
        try:
            return dict(self._research_daily_buckets)
        except Exception:
            return {}

    def research_daily_last_error(self) -> Optional[str]:
        """Return the last error message from research daily load/save, or None."""
        return self._research_daily_last_error

    def record_research_daily_observation_safe(self, observation: Any) -> None:
        """Synchronous, memory-only hand-off for one research daily observation.

        Called from ``_maybe_aggregate_research_daily_from_support_event()``
        (itself only reached from ``record_support_critical_event_safe()``,
        for a genuinely newly-appended Support Critical Event) — still no
        direct runtime/coordinator call site invokes this method itself (see
        research_daily_persistence.py's module docstring for the exact
        reasoning and further next-step hook points, e.g. Episode-based
        aggregation). Never performs storage I/O itself: merges into the in-memory
        ``_research_daily_buckets`` via the pure
        ``append_or_update_research_daily_bucket()`` (which also applies the
        shared RESEARCH_DAILY_RETENTION_DEFAULT policy immediately — no
        new/arbitrary retention numbers).

        ``_research_daily_save_needed`` is set only when the resulting
        payload actually differs from the payload before this call (a
        genuinely no-op observation, e.g. one whose bucket_date is malformed
        and gets silently ignored, does not mark it dirty).

        Never raises: a missing capture-store foundation, a malformed
        observation, or any internal failure is captured in
        ``_research_daily_last_error`` and swallowed — never propagated,
        never touches ``_enabled``.
        """
        if self._capture_stores is None:
            return
        try:
            from datetime import datetime
            from ..storage.research_daily_persistence import (
                append_or_update_research_daily_bucket,
            )
            bucket_date = getattr(observation, "bucket_date", None)
            if not isinstance(bucket_date, str):
                return
            before = dict(self._research_daily_buckets)
            now_utc = datetime.fromisoformat(self._utcnow_iso())
            result = append_or_update_research_daily_bucket(
                {"buckets": self._research_daily_buckets},
                bucket_date,
                observation,
                now_utc=now_utc,
            )
            after = result.updated_payload["buckets"]
            if after != before:
                self._research_daily_buckets = after
                if not self._research_daily_save_needed:
                    self._research_daily_first_dirty_ts = self._utcnow_iso()
                self._research_daily_save_needed = True
        except Exception as err:
            self._research_daily_last_error = str(err)

    def invalidate_boost_after_failed_dispatch_safe(self, dispatch_status: str) -> None:
        """Hard-release boost lifecycle when the coordinator reports a failed/partial dispatch.

        Called post-dispatch when the service call did not fully succeed for all setpoint
        TRVs.  Prevents a stale APPLIED lifecycle from carrying into the next cycle and
        re-applying a boost offset that was never confirmed by the hardware.  Never raises.
        """
        try:
            reason = (BoostBlockReason.PARTIAL_DISPATCH
                      if dispatch_status == "partially_succeeded"
                      else BoostBlockReason.FAILED_DISPATCH)
            self._release_boost_lifecycle_safe(reason)
        except Exception as err:
            self._record_error("invalidate_dispatch", err)

    @property
    def control_enabled(self) -> bool:
        return self._enabled and self._runtime.control_enabled

    def _boost_model(self):
        """Return the BoostModel for this zone, or None if unavailable."""
        try:
            zr = self._runtime._zone(self._zone)
            return zr.orchestrator.models.get("boost")
        except Exception:
            return None

    def _release_boost_lifecycle_safe(self, reason: str) -> None:
        """Release active boost lifecycle with the given reason. Never raises."""
        try:
            model = self._boost_model()
            if model is None:
                return
            from ..models.boost import BoostLifecycle
            lc = model._state.lifecycle
            if lc in (BoostLifecycle.APPLIED, BoostLifecycle.ACTIVE):
                model.release_lifecycle(reason, self._utcnow_iso())
        except Exception as err:
            self._record_error("lifecycle_release", err)

    def _check_lifecycle_timeout_safe(self) -> None:
        """Release lifecycle if max_boost_duration_s has elapsed since apply. Never raises."""
        try:
            model = self._boost_model()
            if model is None:
                return
            from ..models.boost import BoostLifecycle
            s = model._state
            if s.lifecycle not in (BoostLifecycle.APPLIED, BoostLifecycle.ACTIVE):
                return
            if not s.lifecycle_start_ts:
                return
            from datetime import datetime
            start = datetime.fromisoformat(s.lifecycle_start_ts.replace("Z", "+00:00"))
            now = datetime.fromisoformat(self._utcnow_iso())
            elapsed = (now - start).total_seconds()
            if elapsed >= s.lifecycle_max_duration_s:
                model.release_lifecycle("timeout", self._utcnow_iso())
        except Exception as err:
            self._record_error("lifecycle_timeout", err)

    def _get_le2_predictions_for_zone(self) -> dict:
        """Return last_predictions dict for the current zone. Isolated for testing."""
        try:
            zr = self._runtime._zone(self._zone)
            return getattr(zr, "last_predictions", {}) or {}
        except Exception:
            return {}

    def _check_lifecycle_deescalation_safe(self, recommendation: dict) -> None:
        """Release active boost early when real-time data shows heating is self-sufficient.

        Three ordered checks (early return on first match):
        1. released_tpi_sufficient: LE2 HEAT_RATE projection shows TPI closes gap alone,
           within the actual schedule remaining time (or fallback horizon when no schedule).
           High TPI duty alone does NOT prove sufficiency — high duty proves high demand.
        2. released_overshoot_risk: LE2 EXPECTED_OVERSHOOT (AfterheatModel residual rise)
           covers remaining gap near target (≤ 0.3°C). Safety heuristic (remaining ≤ 0.3°C
           AND slope > 0 AND active ≥ 600s) fires as fallback when prediction unavailable.
        3. released_afterheat_sufficient: LE2 EXPECTED_OVERSHOOT (AfterheatModel residual
           rise = learned physical afterheat) covers remaining deficit with safety margin.
           BOOST_OUTCOME must NOT be used here — it measures historical boost quality,
           not physical residual rise after heating stops.

        Authority map (no parallel thermal logic):
          TPI sufficient    → HEAT_RATE + remaining deficit + schedule remaining time
          Overshoot risk    → EXPECTED_OVERSHOOT (AfterheatModel) primary + heuristic fallback
          Afterheat suff.   → EXPECTED_OVERSHOOT (AfterheatModel), authoritative

        The 3600s timeout in _check_lifecycle_timeout_safe remains the absolute safety cap.
        Never raises.
        """
        try:
            model = self._boost_model()
            if model is None:
                return
            from ..models.boost import BoostLifecycle
            s = model._state
            if s.lifecycle not in (BoostLifecycle.APPLIED, BoostLifecycle.ACTIVE):
                return
            # Coordinator writes current_temp (not current_temperature) in the rec dict
            current_temp = recommendation.get("current_temp")
            target = (recommendation.get("effective_target")
                      or recommendation.get("adjusted_target"))
            if current_temp is None or target is None:
                return
            try:
                current_f = float(current_temp)
                target_f = float(target)
            except (TypeError, ValueError):
                return
            remaining = target_f - current_f
            if remaining <= 0:
                return  # target already reached; handled by target_reached check
            slope = recommendation.get("temp_slope")  # °C/min from coordinator
            ts = self._utcnow_iso()
            params = model._params

            # Compute active duration since lifecycle start
            active_dur_s = 0.0
            if s.lifecycle_start_ts:
                try:
                    from datetime import datetime
                    t0 = datetime.fromisoformat(s.lifecycle_start_ts.replace("Z", "+00:00"))
                    now_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    active_dur_s = max(0.0, (now_dt - t0).total_seconds())
                except Exception:
                    pass

            # Compute remaining time to comfort schedule target (for TPI sufficiency gate)
            remaining_time_to_target_min: Optional[float] = None
            if self._last_comfort_time_utc:
                try:
                    from datetime import datetime
                    comfort_dt = datetime.fromisoformat(
                        self._last_comfort_time_utc.replace("Z", "+00:00"))
                    now_dt2 = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    rem_s = (comfort_dt - now_dt2).total_seconds()
                    if rem_s > 0:
                        remaining_time_to_target_min = rem_s / 60.0
                except Exception:
                    pass

            # Read LE2 predictions for thermal authority (never raises)
            heat_rate_c_per_h: Optional[float] = None
            # AfterheatModel residual rise: authoritative for overshoot risk AND afterheat
            afterheat_rise_c: Optional[float] = None
            afterheat_confidence: float = 0.0
            afterheat_valid: bool = False
            try:
                from ..contracts import PredictionType
                le2_preds = self._get_le2_predictions_for_zone()

                # HEAT_RATE: authoritative for TPI sufficiency projection
                hr_pred = le2_preds.get(PredictionType.HEAT_RATE)
                if hr_pred is not None and not getattr(hr_pred, "fallback_used", True):
                    hr_conf = float(getattr(hr_pred, "confidence", 0.0) or 0.0)
                    if hr_conf >= params.deescalation_tpi_heat_rate_min_confidence:
                        hr_val = hr_pred.values.get("heat_rate", 0.0)
                        if hr_val is not None:
                            hr_float = float(hr_val)
                            if hr_float > 0.0:
                                heat_rate_c_per_h = hr_float

                # EXPECTED_OVERSHOOT (from AfterheatModel): authoritative for overshoot risk
                # and afterheat sufficiency. Values key: "expected_overshoot" (unit: °C).
                # BOOST_OUTCOME is NOT used here — it is historical boost outcome quality.
                # Validity gates (all must pass before prediction is used for control):
                #   1. fallback_used=False  — non-cold-start learned prediction only
                #   2. unit must be "C" or "CELSIUS"  — physical sanity check
                #   3. no "stale"/"superseded" in warnings  — prediction currency
                #   4. source_episode_id matches current episode (if both known)
                #   5. confidence >= minimum threshold (checked in release checks below)
                eo_pred = le2_preds.get(PredictionType.EXPECTED_OVERSHOOT)
                if eo_pred is not None and not getattr(eo_pred, "fallback_used", True):
                    # Gate 2: unit — use central contract normalizer (see _is_celsius_unit)
                    _eo_units = getattr(eo_pred, "units", {}) or {}
                    _eo_unit = _eo_units.get("expected_overshoot", "")
                    if _is_celsius_unit(_eo_unit):
                        # Gate 3: stale / superseded
                        _eo_warns = getattr(eo_pred, "warnings", ()) or ()
                        if "stale" not in _eo_warns and "superseded" not in _eo_warns:
                            # Gate 4: episode binding
                            _eo_episode = getattr(eo_pred, "source_episode_id", None)
                            _cur_episode = s.current_episode_id
                            _episode_ok = (
                                not _eo_episode or not _cur_episode
                                or _eo_episode == _cur_episode)
                            if _episode_ok:
                                eo_conf = float(getattr(eo_pred, "confidence", 0.0) or 0.0)
                                eo_val = eo_pred.values.get("expected_overshoot")
                                if eo_val is not None and eo_conf > 0.0:
                                    afterheat_rise_c = float(max(0.0, eo_val))
                                    afterheat_confidence = eo_conf
                                    afterheat_valid = True
            except Exception:
                pass

            # 1. TPI sufficient: LE2 heat-rate shows TPI closes gap within the effective horizon.
            # Effective horizon: remaining_time_to_target - safety_margin (when schedule known),
            # or fallback fixed horizon (when no schedule available — conservative).
            if (heat_rate_c_per_h is not None
                    and remaining <= params.deescalation_tpi_max_remaining_c):
                try:
                    heat_rate_per_min = heat_rate_c_per_h / 60.0
                    predicted_min = remaining / heat_rate_per_min
                    if remaining_time_to_target_min is not None:
                        # Use actual remaining time minus safety margin
                        effective_horizon = (remaining_time_to_target_min
                                             - params.deescalation_tpi_safety_margin_min)
                    else:
                        # No schedule available: use conservative fixed fallback horizon
                        effective_horizon = params.deescalation_tpi_safety_horizon_min
                    slope_ok = (slope is None or float(slope) > 0.0)
                    if predicted_min <= effective_horizon and effective_horizon > 0 and slope_ok:
                        model.release_lifecycle("released_tpi_sufficient", ts)
                        return
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

            # 2. Overshoot risk: LE2 EXPECTED_OVERSHOOT (AfterheatModel residual rise) as primary.
            # Only fires near-target (remaining ≤ overshoot_deficit_c = 0.3°C) to ensure
            # separation from the wider-gap afterheat check (Check 3).
            if (afterheat_valid
                    and afterheat_confidence >= params.deescalation_overshoot_min_confidence
                    and afterheat_rise_c is not None
                    and remaining <= params.deescalation_overshoot_deficit_c):
                try:
                    if (remaining <= afterheat_rise_c + params.deescalation_overshoot_safety_buffer_c
                            and (slope is None or float(slope) > 0.0)):
                        model.release_lifecycle("released_overshoot_risk", ts)
                        return
                except (TypeError, ValueError):
                    pass
            # Heuristic fallback: only after min active time and rising slope. No prediction needed.
            if slope is not None and active_dur_s >= params.deescalation_overshoot_min_active_s:
                try:
                    if (remaining <= params.deescalation_overshoot_deficit_c
                            and float(slope) > 0.0):
                        model.release_lifecycle("released_overshoot_risk", ts)
                        return
                except (TypeError, ValueError):
                    pass

            # 3. Afterheat sufficient: AfterheatModel residual rise covers remaining gap.
            # Authority: PredictionType.EXPECTED_OVERSHOOT (not BOOST_OUTCOME).
            # No valid non-cold-start prediction → no release (slope alone is not afterheat).
            if (afterheat_valid
                    and afterheat_confidence >= params.deescalation_afterheat_min_confidence
                    and afterheat_rise_c is not None):
                try:
                    covers_gap = (
                        afterheat_rise_c >= remaining + params.deescalation_afterheat_safety_margin_c)
                    slope_supporting = (slope is None or float(slope) > 0.0)
                    if covers_gap and slope_supporting:
                        model.release_lifecycle("released_afterheat_sufficient", ts)
                        return
                except (TypeError, ValueError):
                    pass
        except Exception as err:
            self._record_error("lifecycle_deescalation", err)

    def adjust_recommendation_safe(self, recommendation: dict, *,
                                   boost_runtime_limit: Optional[float] = None,
                                   authorized_override: bool = False) -> None:
        """In CONTROL mode only, apply LE 2.0 boost authority to the dispatched recommendation.

        Full authority: LE 2.0 determines the final boost offset (both increases and
        decreases). The existing TPI setpoint is the baseline; LE 2.0 may increase or
        reduce it, subject to confidence gate and device clamps. Never raises; a no-op
        in any non-CONTROL mode (SHADOW stays byte-identical).

        ``authorized_override=True`` is passed by resolve_adaptive_boost_control() to bypass
        the runtime-mode gate and confidence gate; all safety gates remain active.

        Side-effects on ``recommendation`` (Phase B1 provenance tracking):
          ``_boost_rejection_reason`` — gate that blocked boost, or None when applied.
          ``_boost_candidate_c``     — raw LE2 °C proposal (pre-guard); None = no proposal.
          ``_boost_applied_c``       — °C offset actually written; None when not applied.
        """
        recommendation["_boost_rejection_reason"] = None
        recommendation["_boost_candidate_c"] = None
        recommendation["_boost_applied_c"] = None
        # Mode gate: non-bypassable by authorized_override. Production shadows run in
        # LearningRuntimeMode.CONTROL (control_enabled=True). SHADOW-mode instances are
        # read-only capture; they must never apply boost regardless of authorization.
        if not self.control_enabled:
            return
        try:
            target = recommendation.get("effective_target")
            if target is None:
                target = recommendation.get("adjusted_target")
            setpoint = recommendation.get("trv_setpoint")
            if target is None or setpoint is None:
                recommendation["_boost_rejection_reason"] = "no_target"
                return
            # Direct valve TRVs: setpoint is always target; heating authority is valve %.
            # An additive °C offset does not apply — skip boost authority for these.
            if recommendation.get("tpi_valve_direct"):
                recommendation["_boost_rejection_reason"] = "direct_valve"
                return
            # Block during safety conditions; release lifecycle if currently applied
            if recommendation.get("window_open"):
                self._release_boost_lifecycle_safe("window_open")
                recommendation["_boost_rejection_reason"] = "window_open"
                return
            if recommendation.get("heating_failure"):
                self._release_boost_lifecycle_safe("heating_failure")
                recommendation["_boost_rejection_reason"] = "heating_failure"
                return
            # Release on mode change (non-heating mode ends boost context)
            mode = recommendation.get("mode") or recommendation.get("hvac_mode")
            if mode is not None and str(mode).lower() not in ("heat", "auto", "heat_cool"):
                self._release_boost_lifecycle_safe("mode_change")
                recommendation["_boost_rejection_reason"] = "mode_change"
                return
            # Hard release: manual override — user explicitly changed temperature during boost.
            # Outcome is marked as manually influenced; no automatic counter-correction against user.
            if recommendation.get("override_active"):
                self._release_boost_lifecycle_safe("manual_override")
                recommendation["_boost_rejection_reason"] = "manual_override"
                return
            # Hard release: effective target changed — old episode no longer matches context.
            # Prevents carrying a stale boost offset into a new schedule slot or target.
            try:
                _bm_tc = self._boost_model()
                if _bm_tc is not None and _bm_tc._state.lifecycle_base_target_c is not None:
                    if abs(float(_bm_tc._state.lifecycle_base_target_c) - float(target)) > 0.1:
                        self._release_boost_lifecycle_safe("target_change")
                        recommendation["_boost_rejection_reason"] = "target_change"
                        return
            except Exception:
                pass
            # Hard release: schedule/comfort-time context changed with target unchanged.
            # comfort_time_utc is the clearest stable context identifier: it changes when
            # the schedule period transitions (new slot, preheat→comfort, setback shift, etc.).
            # One cycle lag (from observe_safe) is acceptable — same lag as TPI sufficiency.
            try:
                _bm_ctx = self._boost_model()
                if _bm_ctx is not None:
                    _lc_ctx = (_bm_ctx._state.lifecycle.value
                               if hasattr(_bm_ctx._state.lifecycle, "value")
                               else str(_bm_ctx._state.lifecycle))
                    if _lc_ctx in ("applied", "active"):
                        # At least one side must be non-None (both None = no schedule, no change)
                        if (self._boost_applied_comfort_time_utc is not None
                                or self._last_comfort_time_utc is not None):
                            if self._last_comfort_time_utc != self._boost_applied_comfort_time_utc:
                                self._release_boost_lifecycle_safe("schedule_change")
                                recommendation["_boost_rejection_reason"] = "schedule_change"
                                return
            except Exception:
                pass
            # Release on lifecycle timeout (max_boost_duration_s elapsed)
            self._check_lifecycle_timeout_safe()
            # Hard releases must run BEFORE soft deescalation so they always take priority.
            # Hard release: early cutoff / coasting hold — Boost + EarlyCutoff must not overlap.
            early_cutoff_state = recommendation.get("early_cutoff_state")
            if early_cutoff_state in ("cutoff_applied", "coasting_hold"):
                self._release_boost_lifecycle_safe("early_cutoff")
                recommendation["_boost_rejection_reason"] = "early_cutoff"
                return  # Hard release: no boost applied this cycle
            # Hard release: target already reached — no boost benefit remaining.
            # Coordinator key is "current_temp" (not "current_temperature")
            current_temp = recommendation.get("current_temp")
            if current_temp is not None:
                try:
                    if float(current_temp) >= float(target):
                        self._release_boost_lifecycle_safe("target_reached")
                        recommendation["_boost_rejection_reason"] = "target_reached"
                        return
                except (TypeError, ValueError):
                    pass
            # Soft deescalation: release active boost mid-episode when TPI or thermal evidence
            # shows the boost is no longer needed. Only fires after all hard releases above.
            self._check_lifecycle_deescalation_safe(recommendation)
            # Deescalation dispatch: hard or soft, depending on the reason.
            # Hard deesc (overshoot/afterheat): boost is physically overfluous — immediate 0,
            #   same cycle. No step-down; TPI takes over as sole heating authority.
            # Soft deesc (TPI sufficient): controlled step-down per cycle; abrupt removal
            #   not required because TPI *will* close the gap within the schedule window.
            try:
                _bm_soft = self._boost_model()
                if _bm_soft is not None:
                    _slc = (_bm_soft._state.lifecycle.value
                            if hasattr(_bm_soft._state.lifecycle, "value")
                            else str(_bm_soft._state.lifecycle))
                    _HARD_DEESC = frozenset({
                        "released_overshoot_risk",
                        "released_afterheat_sufficient",
                    })
                    _SOFT_DEESC = frozenset({"released_tpi_sufficient"})
                    if _slc in _HARD_DEESC:
                        recommendation["_boost_rejection_reason"] = "deescalation_hard"
                        return  # hard: boost fully removed in this cycle, no step-down
                    if _slc in _SOFT_DEESC:
                        _cur_off = _bm_soft._state.applied_offset_c or 0.0
                        _step = getattr(_bm_soft._params, "deescalation_soft_step_c", 0.5)
                        _new_off = max(0.0, _cur_off - _step)
                        if _new_off > 0.0:
                            _bm_soft.update_deescalation_offset(_new_off)
                            recommendation["tpi_baseline_setpoint"] = float(setpoint)
                            recommendation["trv_setpoint"] = round(
                                float(setpoint) + _new_off, 4)
                            recommendation["_boost_applied_c"] = _new_off
                            recommendation["le2_boost_adjusted"] = True
                        else:
                            recommendation["_boost_rejection_reason"] = "deescalation_soft"
                        return  # no further boost during soft deescalation
            except Exception as err:
                self._record_error("soft_deescalation", err)
            # Cooldown guard: skip boost application during active cooldown period
            _model_cooldown = self._boost_model()
            if _model_cooldown is not None and _model_cooldown.cooldown_active(self._utcnow_iso()):
                recommendation["_boost_rejection_reason"] = "cooldown_active"
                return
            # Baseline = currently applied le2 boost (not TPI offset reconstruction).
            # Anti-chatter compares against this: 0.0 when no boost active.
            baseline_offset = 0.0
            try:
                _bm = self._boost_model()
                if _bm is not None:
                    _lc = _bm._state.lifecycle.value if hasattr(
                        _bm._state.lifecycle, "value") else str(_bm._state.lifecycle)
                    if _lc in ("applied", "active"):
                        baseline_offset = float(_bm._state.applied_offset_c)
            except Exception:
                baseline_offset = max(0.0, float(setpoint) - float(target))
            zr = self._runtime._zone(self._zone)
            boost_pred = zr.last_predictions.get(PredictionType.BOOST_FACTOR)
            proposed = boost_pred.values.get("boost_factor") if boost_pred else None
            # B2b-bootstrap: the orchestrator calls predict() with a default context that
            # lacks device_prior_offset_c, so last_predictions holds the neutral 0.0 default.
            # When the activation readiness (already computed by _read_boost_activation_readiness_safe)
            # reports BOOTSTRAP state with a non-zero factor, use that authoritative factor instead
            # so the bootstrap prior reaches the control decision.
            # B2b-bootstrap: the orchestrator calls predict() with a default context that
            # lacks device_prior_offset_c, so last_predictions holds the neutral 0.0 default.
            # When the activation readiness (already computed by _read_boost_activation_readiness_safe)
            # reports BOOTSTRAP state with a non-zero factor, use that authoritative factor instead
            # so the bootstrap prior reaches the control decision.
            if proposed is None or proposed <= 0.0:
                _br = recommendation.get("_boost_readiness")
                if _br and _br.get("learning_readiness") == "bootstrap":
                    _cf = _br.get("current_factor_c")
                    if _cf is not None and float(_cf) > 0.0:
                        proposed = float(_cf)
            if proposed is None:
                recommendation["_boost_rejection_reason"] = "no_prediction"
                return
            recommendation["_boost_candidate_c"] = float(proposed)
            ctx = ControlContext(
                window_open=bool(recommendation.get("window_open")),
                heating_failure=bool(recommendation.get("heating_failure")),
                boost_runtime_limit=boost_runtime_limit,
                # Forward the boost authorization so ControlPolicy bypasses the shadow-mode
                # runtime gate and the confidence gate (BoostActivationReadiness is the authority).
                authorized_override=authorized_override)
            decision = self._runtime.control_decision(
                self._zone, ControlFeature.BOOST_OFFSET, baseline_offset, proposed,
                ConfidencePurpose.BOOST, context=ctx, unit="celsius_offset",
                model_version=getattr(boost_pred, "model_version", None),
                parameter_version=getattr(boost_pred, "parameter_version", None))
            # Full authority: apply any LE 2.0 decision (increase or decrease)
            if decision.applied and decision.final_value is not None:
                # Wire lifecycle: transition to APPLIED for episode tracking.
                # apply_lifecycle() returns False when episode binding blocks retry
                # (same episode that previously failed). In that case, skip the setpoint
                # update — the failed episode must not boost again in this context.
                _lifecycle_applied = False
                try:
                    model = self._boost_model()
                    if model is not None:
                        episode_id = str(getattr(zr, "last_decision_id", None) or "")
                        _lifecycle_applied = model.apply_lifecycle(
                            episode_id=episode_id,
                            applied_offset_c=decision.final_value,
                            base_target_c=float(target),
                            ts=self._utcnow_iso())
                    else:
                        _lifecycle_applied = True  # no model → no binding check → allow
                except Exception as err:
                    self._record_error("lifecycle_apply", err)
                    _lifecycle_applied = True  # error path → allow (fail open for heating)
                if _lifecycle_applied:
                    # Record context at boost-apply time for schedule-change detection.
                    # Updated every cycle boost is applied; compared next cycle to detect transitions.
                    self._boost_applied_comfort_time_utc = self._last_comfort_time_utc
                    # TPI authority: final = tpi_baseline_setpoint + le2_boost_offset.
                    # Preserve original TPI baseline for audit before overwriting.
                    recommendation["tpi_baseline_setpoint"] = float(setpoint)
                    recommendation["trv_setpoint"] = round(
                        float(setpoint) + decision.final_value, 4)
                    recommendation["_boost_applied_c"] = decision.final_value
                    recommendation["le2_boost_adjusted"] = True
                else:
                    recommendation["_boost_rejection_reason"] = "lifecycle_blocked"
            else:
                recommendation["_boost_rejection_reason"] = "guard_rejected"
        except Exception as err:
            self._record_error("control", err)

    def build_live_decision_pre_dispatch(
        self,
        recommendation: Mapping[str, Any],
        *,
        ts: Optional[str] = None,
    ) -> Optional[Any]:
        """Create pre-dispatch LiveDecisionRecord from the current recommendation.

        Called after ``adjust_recommendation_safe()`` but before actual HA dispatch.
        Returns an immutable record; the caller must add dispatch fields via
        ``dataclasses.replace()`` after dispatch and then pass the completed record
        to ``compute_decision_trace_safe()``.  Returns None if the shadow is disabled
        or an internal error occurs — never raises.
        """
        if not self._enabled:
            return None
        try:
            from ..decision.contracts import LiveDecisionRecord
            from ..contracts import PredictionType as _PT
            zr = self._runtime._zone(self._zone)
            decision_id = getattr(zr, "last_decision_id", None)
            _tpi_diag = recommendation.get("tpi_coef_diag") or {}
            boost_applied = bool(recommendation.get("le2_boost_adjusted"))
            baseline_sp = (
                recommendation.get("tpi_baseline_setpoint")
                if boost_applied
                else recommendation.get("trv_setpoint")
            )
            mode_str = "control" if self._runtime.control_enabled else "shadow"
            # Onset delay — read from current predictions (same source as resolver path)
            onset_delay_min: Optional[float] = None
            onset_delay_source: Optional[str] = None
            try:
                _last_preds = getattr(zr, "last_predictions", {}) or {}
                _od_pred = _last_preds.get(_PT.ONSET_DELAY)
                if _od_pred is not None:
                    _od_fb = getattr(_od_pred, "fallback_used", True)
                    _od_conf = float(getattr(_od_pred, "confidence", 0.0) or 0.0)
                    if not _od_fb and _od_conf >= self._PREHEAT_MIN_CONFIDENCE:
                        onset_delay_min = float(
                            _od_pred.values.get("onset_delay_minutes", 0.0) or 0.0)
                        onset_delay_source = "valid"
                    else:
                        onset_delay_min = self._ONSET_DELAY_PRIOR_MIN
                        onset_delay_source = "cold_start_prior"
            except Exception:
                pass
            return LiveDecisionRecord(
                decision_id=decision_id,
                zone_id=self._zone,
                ts=ts or self._utcnow_iso(),
                mode=mode_str,
                baseline_setpoint_c=(float(baseline_sp) if baseline_sp is not None else None),
                final_setpoint_c=(float(recommendation["trv_setpoint"])
                                  if recommendation.get("trv_setpoint") is not None else None),
                boost_applied=boost_applied,
                boost_candidate_c=(float(recommendation["_boost_candidate_c"])
                                   if recommendation.get("_boost_candidate_c") is not None
                                   else None),
                boost_applied_c=(float(recommendation["_boost_applied_c"])
                                 if recommendation.get("_boost_applied_c") is not None
                                 else None),
                boost_rejected_reason=recommendation.get("_boost_rejection_reason"),
                preheat_minutes=float(recommendation.get("preheat_minutes") or 0.0),
                preheat_status=recommendation.get("preheat_status"),
                early_cutoff_state=recommendation.get("early_cutoff_state"),
                heat_rate_c_per_h=_tpi_diag.get("heat_rate_c_per_h"),
                heat_rate_confidence=_tpi_diag.get("hr_confidence"),
                heat_rate_source=recommendation.get("tpi_coef_source"),
                onset_delay_min=onset_delay_min,
                onset_delay_source=onset_delay_source,
            )
        except Exception as err:
            self._record_error("live_decision_record", err)
            return None

    def enrich_live_decision_pre_dispatch(
        self,
        record: Any,
        recommendation: Mapping[str, Any],
    ) -> Optional[Any]:
        """Enrich a coordinator-built baseline LiveDecisionRecord with LE2-specific fields.

        Adds ``decision_id`` from the zone lifecycle and ``onset_delay`` from LE2
        predictions.  Returns the enriched record (via ``dataclasses.replace``), or
        the original record unchanged if enrichment fails.  Never raises.
        """
        if not self._enabled or record is None:
            return record
        try:
            import dataclasses as _dc
            from ..contracts import PredictionType as _PT
            zr = self._runtime._zone(self._zone)
            _le2_decision_id = getattr(zr, "last_decision_id", None)
            # Only override the coordinator's baseline decision_id when LE2 has generated one.
            # Overwriting with None would lose the coordinator-generated ID.
            decision_id = _le2_decision_id if _le2_decision_id is not None else record.decision_id
            onset_delay_min: Optional[float] = None
            onset_delay_source: Optional[str] = None
            try:
                _last_preds = getattr(zr, "last_predictions", {}) or {}
                _od_pred = _last_preds.get(_PT.ONSET_DELAY)
                if _od_pred is not None:
                    _od_fb = getattr(_od_pred, "fallback_used", True)
                    _od_conf = float(getattr(_od_pred, "confidence", 0.0) or 0.0)
                    if not _od_fb and _od_conf >= self._PREHEAT_MIN_CONFIDENCE:
                        onset_delay_min = float(
                            _od_pred.values.get("onset_delay_minutes", 0.0) or 0.0)
                        onset_delay_source = "valid"
                    else:
                        onset_delay_min = self._ONSET_DELAY_PRIOR_MIN
                        onset_delay_source = "cold_start_prior"
            except Exception:
                pass
            return _dc.replace(
                record,
                decision_id=decision_id,
                onset_delay_min=onset_delay_min,
                onset_delay_source=onset_delay_source,
            )
        except Exception as err:
            self._record_error("enrich_live_decision", err)
            return record  # return original record, not None

    def _build_trace_from_live_record(
        self,
        record: Any,
        recommendation: Mapping[str, Any],
        *,
        ts: Optional[str] = None,
    ) -> Any:
        """Derive DecisionTrace from the authoritative live record — no resolver re-run."""
        from ..decision.contracts import (
            DecisionTrace, DecisionTraceEntry,
            DecisionReason, FallbackReason, DecisionMode,
        )

        boost_applied = record.boost_applied
        boost_applied_c = record.boost_applied_c or 0.0
        boost_candidate = record.boost_candidate_c
        boost_rejected = record.boost_rejected_reason

        boost_entry = DecisionTraceEntry(
            feature="boost_offset",
            baseline_value=0.0,
            le2_value=boost_candidate,
            final_value=boost_applied_c if boost_applied else 0.0,
            applied=boost_applied,
            reason=(DecisionReason.LE2_APPLIED.value if boost_applied
                    else (boost_rejected or DecisionReason.BASELINE.value)),
            fallback_reason=(FallbackReason.NONE.value if boost_applied
                             else FallbackReason.GUARD_BLOCKED.value),
            confidence=None,
            clamp_applied=0.0,
        )

        preheat_min = record.preheat_minutes
        preheat_status = record.preheat_status
        preheat_is_le2 = preheat_status in ("valid", "valid_hl_prior")
        preheat_entry = DecisionTraceEntry(
            feature="preheat_start",
            baseline_value=recommendation.get("preheat_baseline_minutes"),
            le2_value=preheat_min if preheat_is_le2 else None,
            final_value=preheat_min,
            applied=preheat_min > 0.0,
            reason=("le2_applied" if preheat_is_le2 else "deterministic_baseline"),
            fallback_reason=("none" if preheat_is_le2 else "shadow_mode"),
            confidence=record.heat_rate_confidence,
            clamp_applied=0.0,
        )

        ec_state = record.early_cutoff_state
        ec_active = ec_state in ("cutoff_applied", "coasting_hold")
        ec_contribution = recommendation.get("early_cutoff_contribution_c")
        ec_entry = DecisionTraceEntry(
            feature="early_cutoff",
            baseline_value=None,
            le2_value=ec_contribution,
            final_value=ec_contribution if ec_active else None,
            applied=ec_active,
            reason=("le2_applied" if ec_active else DecisionReason.BASELINE.value),
            fallback_reason=(FallbackReason.NONE.value if ec_active
                             else FallbackReason.GUARD_BLOCKED.value),
            confidence=None,
            clamp_applied=0.0,
        )

        reason_codes: list[str] = []
        if boost_applied:
            reason_codes.append(DecisionReason.LE2_APPLIED.value)
        if boost_rejected:
            reason_codes.append(boost_rejected)
        if not boost_applied and not boost_rejected:
            reason_codes.append(DecisionReason.BASELINE.value)

        mode_str = (DecisionMode.CONTROL.value if self._runtime.control_enabled
                    else DecisionMode.SHADOW.value)

        onset_delay = record.onset_delay_min
        room_heat_dur: Optional[float] = None
        if onset_delay is not None:
            room_heat_dur = max(0.0, preheat_min - onset_delay)

        return DecisionTrace(
            zone_id=record.zone_id,
            ts=ts or record.ts,
            mode=mode_str,
            decision_id=record.decision_id,
            baseline_setpoint_c=record.baseline_setpoint_c,
            final_setpoint_c=record.final_setpoint_c,
            applied_any=boost_applied or ec_active,
            entries=(boost_entry, preheat_entry, ec_entry),
            reason_codes=tuple(sorted(set(reason_codes))),
            preheat_minutes_le2=preheat_min if preheat_is_le2 else None,
            preheat_status=preheat_status,
            preheat_baseline_minutes=recommendation.get("preheat_baseline_minutes"),
            selected_preheat_min=preheat_min,
            heat_rate_c_per_h=record.heat_rate_c_per_h,
            heat_rate_confidence=record.heat_rate_confidence,
            heat_rate_status=record.heat_rate_source,
            effective_onset_delay_min=onset_delay or 0.0,
            onset_delay_source=record.onset_delay_source,
            onset_delay_status=record.onset_delay_source,
            effective_room_heating_duration_min=room_heat_dur,
            preheat_command_lead_time_min=preheat_min,
            early_cutoff_state=ec_state,
            early_cutoff_hold_active=ec_active,
            boost_offset_c_applied=boost_applied_c if boost_applied else None,
            boost_offset_requested_c=boost_candidate,
            boost_rejected_reason=boost_rejected,
            final_device_setpoint_c=(record.dispatch_setpoint_c
                                      if record.dispatch_setpoint_c is not None
                                      else record.final_setpoint_c),
        )

    def emit_boost_authority_status_safe(self, recommendation: dict) -> None:
        """B2b-3c: authoritatively emit the three-state boost evaluation for THIS decision.

        Runs every cycle (SHADOW and CONTROL), after ``adjust_recommendation_safe``. The
        single source of truth for whether a boost was applied / evaluated-not-applied /
        not-evaluated. ``0.0`` (evaluated, authoritatively no boost) is never collapsed to
        ``None`` (not evaluated). Never raises; never changes a dispatched value.

        Sets ``_boost_applied_c`` and ``_boost_evaluation_status``; preserves any rejection
        reason already recorded by the CONTROL applier (e.g. ``direct_valve``).
        """
        from ..decision.contracts import BoostEvaluationStatus as _BES
        try:
            # LE2 disabled / authority not reachable -> boost was NOT evaluated.
            if not self._enabled:
                recommendation["_boost_applied_c"] = None
                recommendation["_boost_evaluation_status"] = _BES.NOT_EVALUATED.value
                return
            applied = recommendation.get("_boost_applied_c")
            # An applied boost (CONTROL path wrote a positive offset) -> APPLIED.
            if applied is not None and float(applied) > 0.0:
                recommendation["_boost_evaluation_status"] = _BES.APPLIED.value
                return
            target = recommendation.get("effective_target")
            if target is None:
                target = recommendation.get("adjusted_target")
            setpoint = recommendation.get("trv_setpoint")
            # Required runtime data missing -> no authoritative statement -> NOT_EVALUATED.
            if target is None or setpoint is None:
                recommendation["_boost_applied_c"] = None
                recommendation["_boost_evaluation_status"] = _BES.NOT_EVALUATED.value
                return
            # Direct-valve: an additive °C boost is not applicable; the authority evaluated
            # it and authoritatively applied no boost (§7).
            if recommendation.get("tpi_valve_direct"):
                recommendation["_boost_applied_c"] = 0.0
                recommendation["_boost_evaluation_status"] = _BES.EVALUATED_NOT_APPLIED.value
                if not recommendation.get("_boost_rejection_reason"):
                    recommendation["_boost_rejection_reason"] = "direct_valve"
                return
            # Heating decision in scope, no boost applied -> authoritatively 0.0.
            recommendation["_boost_applied_c"] = 0.0
            recommendation["_boost_evaluation_status"] = _BES.EVALUATED_NOT_APPLIED.value
        except Exception as err:
            self._record_error("boost_status", err)

    # ── Adaptive Boost Control Resolver ──────────────────────────────────────

    def _read_boost_activation_readiness_safe(
        self, recommendation: Mapping[str, Any]
    ) -> Optional["BoostActivationReadiness"]:  # type: ignore[name-defined]
        """Read the BoostActivationReadiness from the BoostModel for the current context.

        Pure-on-model-state: derived entirely from persisted model state plus current
        recommendation context.  Returns None on any error (never raises).
        """
        try:
            from ..models.boost import BoostModel, BoostPredictionContext
            from ..models.boost_activation import BoostActivationReadiness
            model = self._boost_model()
            if model is None:
                return None
            target = recommendation.get("effective_target")
            if target is None:
                target = recommendation.get("adjusted_target")
            current = recommendation.get("current_temp")
            deficit = None
            if target is not None and current is not None:
                try:
                    deficit = max(0.0, float(target) - float(current))
                except (TypeError, ValueError):
                    pass
            # Control type from dispatch path
            ctrl_type = "direct_valve" if recommendation.get("tpi_valve_direct") else "setpoint"
            # Collect runtime blocking confounders
            blocking: list[str] = []
            if recommendation.get("window_open"):
                blocking.append("window_open")
            if recommendation.get("heating_failure"):
                blocking.append("heating_failure")
            preheat_active = bool(recommendation.get("preheat_minutes", 0))
            decision_type_str = "preheat" if preheat_active else "normal"
            # B2b-bootstrap: read zone-configured device prior from recommendation so
            # the bootstrap path in BoostModel can allow initial controlled trials.
            _bsp = recommendation.get("boost_bootstrap_prior_c")
            _device_prior = float(_bsp) if _bsp and float(_bsp) > 0.0 else None
            ctx = BoostPredictionContext(
                start_deficit_c=deficit,
                decision_type=decision_type_str,
                control_type=ctrl_type,
                blocking_reasons=tuple(blocking),
                now_ts=self._utcnow_iso(),
                device_prior_offset_c=_device_prior,
            )
            return model.activation_readiness(ctx)
        except Exception as err:
            self._record_error("activation_readiness", err)
            return None

    def resolve_adaptive_boost_control(
        self,
        recommendation: dict,
        *,
        learning_mode_on: bool,
        active_control_on: bool,
        restore_pending: bool = False,
        boost_runtime_limit: Optional[float] = None,
    ) -> "AdaptiveBoostControlResult":
        """Central adaptive boost resolver. Restore barrier → double gate → readiness → inner safety gates.

        The ONLY production entry point for real boost control.  Returns a fully typed
        result for provenance; sets diagnostic keys on ``recommendation``.  Never raises.

        Gate order (strict):
          0. Restore barrier  (active_control RestoreEntity initialized)
          1. Learning mode ON  (cfg learning_enabled + LE2 enabled)
          2. Active Control ON (coordinator _active_control)
          3. Mixed-control zone check
          3.5. Pre-dispatch device availability
          4. BoostActivationReadiness.eligibility == True AND factor_usable == True
          5. Inner safety gates (adjust_recommendation_safe) — hard/soft/cooldown/policy
        """
        _ZERO = AdaptiveBoostControlResult(
            requested_boost_offset_c=None, approved_boost_offset_c=0.0,
            applied_boost_offset_c=0.0, boost_allowed=False,
            selected_scope=None, eligibility_reason="not_evaluated",
            blocking_reason=None, release_reason=None, clamp_applied=False,
            source_decision_id=None,
        )

        def _blocked(reason: str, *, scope: Optional[str] = None,
                     release: bool = False) -> AdaptiveBoostControlResult:
            if release:
                self._release_boost_lifecycle_safe(reason)
            recommendation["_boost_rejection_reason"] = reason
            return AdaptiveBoostControlResult(
                requested_boost_offset_c=None, approved_boost_offset_c=0.0,
                applied_boost_offset_c=0.0, boost_allowed=False,
                selected_scope=scope, eligibility_reason=reason,
                blocking_reason=reason,
                release_reason=reason if release else None,
                clamp_applied=False, source_decision_id=None,
            )

        try:
            # Gate 0: Restore initialization barrier.
            # Active Control RestoreEntity must have completed at least one restore/set
            # call.  Until then boost is blocked so no partial-restore cycle can
            # accidentally activate boost.  Learning mode comes from config (not a
            # RestoreEntity) and is therefore not in the barrier.
            if restore_pending:
                return _blocked(BoostBlockReason.RESTORE_PENDING)

            # Gate 1: Learning mode (coordinator-side flag + LE2 health)
            if not learning_mode_on or not self._enabled:
                return _blocked(BoostBlockReason.LEARNING_MODE_OFF, release=True)

            # Gate 2: Active Control
            if not active_control_on:
                return _blocked(BoostBlockReason.ACTIVE_CONTROL_OFF, release=True)

            # Gate 3: Mixed control zone — adaptive boost blocked conservatively.
            # A zone mixing setpoint-TRVs and direct-valve-TRVs cannot safely receive
            # a per-device split; the entire zone is blocked with a clear typed reason.
            if recommendation.get("zone_control_type") == "mixed":
                return _blocked(BoostBlockReason.MIXED_CONTROL_TYPES, release=True)

            # Gate 3.5: Pre-dispatch device availability.
            # If any setpoint TRV was unavailable when the coordinator sampled states
            # before the resolver, boost is blocked for this cycle.  Prevents a partial
            # dispatch where some TRVs received the boosted setpoint and others did not.
            if recommendation.get("_setpoint_device_unavailable"):
                return _blocked(BoostBlockReason.DEVICE_UNAVAILABLE, release=True)

            # Gate 4: BoostActivationReadiness
            readiness = self._read_boost_activation_readiness_safe(recommendation)
            selected_scope = readiness.selected_scope if readiness is not None else "prior"
            if readiness is None or not readiness.eligibility or not readiness.factor_usable:
                reason = readiness.eligibility_reason if readiness is not None else "readiness_unavailable"
                self._release_boost_lifecycle_safe(reason)
                recommendation["_boost_rejection_reason"] = reason
                return AdaptiveBoostControlResult(
                    requested_boost_offset_c=None, approved_boost_offset_c=0.0,
                    applied_boost_offset_c=0.0, boost_allowed=False,
                    selected_scope=selected_scope, eligibility_reason=reason,
                    blocking_reason=reason, release_reason=reason,
                    clamp_applied=False, source_decision_id=None,
                )

            # Gate 4 passed — emit readiness diagnostics on recommendation
            try:
                recommendation["_boost_readiness"] = readiness.to_dict()
            except Exception:
                pass

            # Inner gates: invoke adjust_recommendation_safe() with authorization.
            # authorized_override=True passes directly — no mutable flag lifecycle needed.
            self.adjust_recommendation_safe(
                recommendation, boost_runtime_limit=boost_runtime_limit,
                authorized_override=True)

            # Build typed result from what the inner method wrote
            inner_rejection = recommendation.get("_boost_rejection_reason")
            applied_c: float = float(recommendation.get("_boost_applied_c") or 0.0)
            candidate_c = recommendation.get("_boost_candidate_c")
            boost_applied = bool(recommendation.get("le2_boost_adjusted", False))
            approved_c: float = applied_c if boost_applied else 0.0

            clamp_applied = False
            if candidate_c is not None and applied_c > 0.0:
                try:
                    clamp_applied = abs(float(candidate_c) - applied_c) > 0.001
                except (TypeError, ValueError):
                    pass

            source_did: Optional[str] = None
            try:
                zr = self._runtime._zone(self._zone)
                _did = getattr(zr, "last_decision_id", None)
                source_did = str(_did) if _did is not None else None
            except Exception:
                pass

            return AdaptiveBoostControlResult(
                requested_boost_offset_c=(float(candidate_c)
                                          if candidate_c is not None else None),
                approved_boost_offset_c=approved_c,
                # None = pre-dispatch; actual applied truth lives in LiveDecisionRecord.
                # 0.0 when inner gates blocked the offset (definitely not applied).
                applied_boost_offset_c=None if boost_applied else 0.0,
                boost_allowed=True,
                selected_scope=selected_scope,
                eligibility_reason=readiness.eligibility_reason,
                blocking_reason=inner_rejection if not boost_applied else None,
                release_reason=None,
                clamp_applied=clamp_applied,
                source_decision_id=source_did,
                authorization_source="bootstrap_activation",
                bypassed_gates=(
                    "confidence", "reliability", "fallback_dominant", "prior_dominant"
                ),
            )
        except Exception as err:
            self._record_error("resolve_adaptive_boost", err)
            return _ZERO

    def compute_decision_trace_safe(self, recommendation: Mapping[str, Any], *,
                                    active_control: bool = False,
                                    ts: Optional[str] = None,
                                    live_record: Optional[Any] = None) -> None:
        """Produce a DecisionTrace for this coordinator cycle.

        When ``live_record`` (a ``LiveDecisionRecord``) is provided, the trace is
        derived directly from the authoritative live path without re-running the
        resolver.  When omitted, the full pipeline is re-run for backward
        compatibility (test-only path). Never raises into the heating cycle.
        """
        if not self._enabled:
            return
        try:
            if live_record is not None:
                trace = self._build_trace_from_live_record(
                    live_record, recommendation, ts=ts or self._utcnow_iso())
            else:
                # LEGACY / TEST-ONLY: backward-compat path for unit tests that call
                # compute_decision_trace_safe() without a live_record.  The production
                # coordinator always provides live_record; this branch must never run
                # in a live coordinator cycle (enforced by test_production_live_record_always_provided).
                zr = self._runtime._zone(self._zone)
                preds = build_learning_prediction_set(
                    self._zone, getattr(zr, "last_predictions", {}) or {},
                    getattr(zr, "last_confidence", {}) or {},
                    decision_id=getattr(zr, "last_decision_id", None))
                mode = (DecisionMode.CONTROL if self._runtime.control_enabled
                        else DecisionMode.SHADOW)
                trace, _ = self._pipeline.run(
                    self._zone, ts or self._utcnow_iso(), recommendation, preds, mode=mode,
                    active_control=active_control,
                    decision_id=getattr(zr, "last_decision_id", None))
            trace_dict = trace.support_dict()
            # Enrich trace with TPI authority chain fields (for audit/debug)
            # These allow verification that LE 2.0 setpoint and TPI baseline are not mixed.
            # tpi_baseline_setpoint_c: written by adjust_recommendation_safe before boost.
            # Falls back to trv_setpoint in shadow mode (no boost → not overwritten).
            _tpi_diag = recommendation.get("tpi_coef_diag") or {}
            trace_dict.update({
                # decision_id: links this trace to the lifecycle capture and any outcome record.
                "decision_id": trace.decision_id,
                "comfort_target_c": recommendation.get("effective_target"),
                "tpi_duty_c": recommendation.get("tpi_duty_cycle"),
                "tpi_baseline_setpoint_c": (recommendation.get("tpi_baseline_setpoint")
                                            or recommendation.get("trv_setpoint")),
                "tpi_valve_direct": bool(recommendation.get("tpi_valve_direct", False)),
                # HeatLoss authority trace (Phase 19D — Variante B bounded adaptive heuristic)
                "tpi_coef_source": recommendation.get("tpi_coef_source"),
                # Coefficient authority chain: default → learned → blend → used
                "tpi_coef_int_default": _tpi_diag.get("coef_int_default"),
                "tpi_coef_int_learned": _tpi_diag.get("coef_int_learned"),
                "tpi_coef_blend_weight": _tpi_diag.get("blend_weight"),
                "tpi_coef_int_used": recommendation.get("tpi_coef_int"),
                "tpi_coef_ext_used": recommendation.get("tpi_coef_ext"),
                # Prediction values feeding the ratio
                "heat_rate_prediction_c_per_h": _tpi_diag.get("heat_rate_c_per_h"),
                "heat_loss_prediction_c_per_h": recommendation.get("tpi_hl_rate"),
                "heat_rate_confidence": _tpi_diag.get("hr_confidence"),
                "heat_loss_confidence": _tpi_diag.get("hl_confidence"),
                # Outdoor/context availability
                "tpi_outdoor_context_available": _tpi_diag.get("outdoor_context_available"),
                "tpi_relative_only": _tpi_diag.get("relative_only"),
                # Backward-compat alias (used by parity tests)
                "tpi_heat_loss_prediction_c_per_h": recommendation.get("tpi_hl_rate"),
                # Step-limiting transition audit fields (Phase 19D)
                "tpi_coef_int_candidate": _tpi_diag.get("coef_int_candidate"),
                "tpi_coef_int_previous": _tpi_diag.get("coef_int_previous"),
                "tpi_coef_transition_applied": _tpi_diag.get("transition_applied"),
                "tpi_coef_transition_reason": _tpi_diag.get("transition_reason"),
                "tpi_coef_max_step_up": _tpi_diag.get("coef_int_max_step_up"),
                "tpi_coef_max_step_down": _tpi_diag.get("coef_int_max_step_down"),
                # Duty-level transition audit
                "tpi_duty_previous": _tpi_diag.get("duty_previous"),
                "tpi_duty_candidate": _tpi_diag.get("duty_candidate"),
                "tpi_duty_used": _tpi_diag.get("duty_used"),
            })
            # Enrich trace with lifecycle diagnostics from BoostModel (not available in resolver)
            try:
                boost_model = self._boost_model()
                if boost_model is not None:
                    s = boost_model._state
                    ts_now = ts or self._utcnow_iso()
                    active_dur: Optional[float] = None
                    cooldown_rem: Optional[float] = None
                    from ..models.boost import BoostLifecycle
                    from datetime import datetime, timezone, timedelta
                    if s.lifecycle in (BoostLifecycle.APPLIED, BoostLifecycle.ACTIVE) \
                            and s.lifecycle_start_ts:
                        try:
                            start = datetime.fromisoformat(
                                s.lifecycle_start_ts.replace("Z", "+00:00"))
                            now_dt = datetime.fromisoformat(ts_now.replace("Z", "+00:00"))
                            active_dur = round((now_dt - start).total_seconds(), 1)
                        except Exception:
                            pass
                    if s.cooldown_until_ts:
                        try:
                            until_dt = datetime.fromisoformat(
                                s.cooldown_until_ts.replace("Z", "+00:00"))
                            now_dt2 = datetime.fromisoformat(ts_now.replace("Z", "+00:00"))
                            rem = (until_dt - now_dt2).total_seconds()
                            cooldown_rem = round(max(0.0, rem), 1)
                        except Exception:
                            pass
                    # Classify deescalation type from release reason.
                    # Overshoot/afterheat deescalation are HARD (immediate 0, same cycle).
                    # Only TPI sufficient is soft (step-down per cycle).
                    _release_reason = s.lifecycle_release_reason
                    _hard_reasons = frozenset({
                        "window_open", "heating_failure", "mode_change",
                        "early_cutoff", "target_reached", "manual_override",
                        "schedule_change", "timeout", "no_response", "overshoot",
                        "target_change",
                        "released_overshoot_risk", "released_afterheat_sufficient"})
                    _soft_reasons = frozenset({"released_tpi_sufficient"})
                    _deesc_type = None
                    if _release_reason in _hard_reasons:
                        _deesc_type = "hard"
                    elif _release_reason in _soft_reasons:
                        _deesc_type = "soft"
                    # Read afterheat prediction for trace enrichment
                    _afterheat_c: Optional[float] = None
                    _afterheat_status: Optional[str] = None
                    _afterheat_conf: Optional[float] = None
                    _boost_outcome_risk: Optional[bool] = None
                    try:
                        from ..contracts import PredictionType as _PT
                        _le2p = self._get_le2_predictions_for_zone()
                        _eo_pred = _le2p.get(_PT.EXPECTED_OVERSHOOT)
                        if _eo_pred is not None:
                            _afterheat_c = float(max(0.0,
                                _eo_pred.values.get("expected_overshoot", 0.0) or 0.0))
                            _afterheat_conf = float(getattr(_eo_pred, "confidence", 0.0) or 0.0)
                            _afterheat_status = (
                                "cold_start" if getattr(_eo_pred, "fallback_used", True)
                                else "learned")
                        else:
                            _afterheat_status = "unavailable"
                        _bo_pred = _le2p.get(_PT.BOOST_OUTCOME)
                        if _bo_pred is not None and not getattr(_bo_pred, "fallback_used", True):
                            _bo_ov = _bo_pred.values.get("expected_overshoot")
                            if _bo_ov is not None:
                                _boost_outcome_risk = float(_bo_ov) > 0.0
                    except Exception:
                        pass
                    # Remaining time to target from cached comfort_time_utc
                    _remaining_tgt_min: Optional[float] = None
                    if self._last_comfort_time_utc:
                        try:
                            from datetime import datetime
                            _c_dt = datetime.fromisoformat(
                                self._last_comfort_time_utc.replace("Z", "+00:00"))
                            _n_dt = datetime.fromisoformat(ts_now.replace("Z", "+00:00"))
                            _rem_s2 = (_c_dt - _n_dt).total_seconds()
                            if _rem_s2 > 0:
                                _remaining_tgt_min = round(_rem_s2 / 60.0, 1)
                        except Exception:
                            pass
                    # Current and new offset for trace
                    _new_off_c = float(recommendation.get("trv_setpoint", 0.0) or 0.0) - \
                        float(recommendation.get("tpi_baseline_setpoint") or
                              recommendation.get("trv_setpoint") or 0.0)
                    _params_soft = boost_model._params
                    trace_dict.update({
                        "boost_lifecycle_state": s.lifecycle.value,
                        "boost_episode_id": s.current_episode_id,
                        "boost_active_duration_s": active_dur,
                        "boost_max_duration_s": s.lifecycle_max_duration_s,
                        "boost_cooldown_remaining_s": cooldown_rem,
                        "boost_release_reason": _release_reason,
                        "boost_applied_offset_c": s.applied_offset_c,
                        "boost_failed_episode_id": s.last_failed_episode_id,
                        "boost_completed_episode_id": s.last_completed_episode_id,
                        "boost_deescalation_type": _deesc_type,
                        "boost_previous_offset_c": s.applied_offset_c,
                        "boost_new_offset_c": max(0.0, _new_off_c) if _new_off_c > 0 else 0.0,
                        "final_device_setpoint_c": recommendation.get("trv_setpoint"),
                        # Afterheat authority (EXPECTED_OVERSHOOT from AfterheatModel)
                        "afterheat_prediction_c": _afterheat_c,
                        "afterheat_prediction_status": _afterheat_status,
                        "afterheat_confidence": _afterheat_conf,
                        "expected_afterheat_c": _afterheat_c,  # alias
                        "boost_outcome_overshoot_risk": _boost_outcome_risk,
                        # TPI sufficiency time fields
                        "remaining_time_to_target_min": _remaining_tgt_min,
                        "tpi_sufficiency_safety_margin_min": getattr(
                            _params_soft, "deescalation_tpi_safety_margin_min", 5.0),
                    })
            except Exception:
                pass
            self._last_trace = trace_dict
        except Exception as err:
            self._record_error("decision_trace", err)

    def clear_last_trace(self) -> None:
        """Clear the cached decision trace.

        Called by the coordinator when no authoritative live record is available
        for this cycle — prevents a stale trace from a prior cycle being read as
        current-cycle provenance.
        """
        self._last_trace = None

    @property
    def last_decision_trace(self) -> Optional[dict]:
        return self._last_trace

    def confidence_display(self) -> float:
        """Combined learning confidence [0,1] from the LE-2.0 aggregator (display only)."""
        from ..decision.confidence_adapter import combined_confidence
        try:
            zr = self._runtime._zone(self._zone)
            return combined_confidence(getattr(zr, "last_confidence", {}) or {})
        except Exception as err:
            self._record_error("confidence_display", err)
            return 0.0

    def confidence_display_attributes(self) -> dict:
        """Legacy-shaped confidence breakdown sourced purely from LE-2.0 (display only)."""
        from ..decision.confidence_adapter import confidence_breakdown
        try:
            zr = self._runtime._zone(self._zone)
            counts = self._runtime.health().model_update_counts
            return confidence_breakdown(getattr(zr, "last_confidence", {}) or {},
                                        model_update_counts=counts)
        except Exception as err:
            self._record_error("confidence_attrs", err)
            return {}

    # Maps Read Gate rejection codes to user-facing forecast trust status strings.
    _TRUST_STATUS_MAP: dict = {
        "missing": "missing",
        "stale": "stale",
        "superseded": "superseded",
    }

    def read_forecast_trust_safe(self) -> tuple:
        """Read LE 2.0 FORECAST_TRUST (dimensionless reliability [0,1]) via validated Read Gate.

        Returns ``(trust, status)`` where:
        - ``trust``: 0.0 when unavailable / cold-start / invalid; float in [0,1] when valid.
        - ``status``: one of "valid", "not_available", "missing", "cold_start", "stale",
          "superseded", "invalid".

        Status "not_available" means the shadow is disabled or an internal error occurred.
        Status "cold_start" means a prior prediction exists but has no real evidence
        (``fallback_used=True``).

        In both unavailable and cold_start cases ``trust=0.0`` so the suppression formula
        ``1 – (1–raw)·trust = 1.0`` leaves heating fully unaffected — the local thermal
        path takes over without any invented high forecast confidence.

        Semantically distinct from FORECAST_BIAS – never mixed.
        """
        if not self._enabled:
            return 0.0, "not_available"
        try:
            from ..decision.runtime_adapter import validated_prediction_read, READ_OK
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            trust_pred = preds.get(PredictionType.FORECAST_TRUST)
            if trust_pred is not None and getattr(trust_pred, "fallback_used", True):
                return 0.0, "cold_start"
            value, read_status = validated_prediction_read(
                preds,
                "forecast_trust",
                zone_id=self._zone,
                prediction_zone_id=self._zone,
                expected_unit="score",
            )
            if read_status == READ_OK and value is not None:
                return float(max(0.0, min(1.0, value))), "valid"
            return 0.0, self._TRUST_STATUS_MAP.get(read_status, "invalid")
        except Exception as err:
            self._record_error("forecast_trust_read", err)
        return 0.0, "not_available"

    def read_forecast_bias_safe(self) -> float:
        """Read LE 2.0 FORECAST_BIAS (signed °C error correction) via validated Read Gate.

        Semantically distinct from FORECAST_TRUST – never used as a reliability score.
        Returns 0.0°C when cold-start (fallback_used=True) or gate rejects.
        """
        if not self._enabled:
            return 0.0
        try:
            from ..decision.runtime_adapter import validated_prediction_read, READ_OK
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            bias_pred = preds.get(PredictionType.FORECAST_BIAS)
            if bias_pred is not None and getattr(bias_pred, "fallback_used", True):
                return 0.0
            value, status = validated_prediction_read(
                preds,
                "forecast_bias",
                zone_id=self._zone,
                prediction_zone_id=self._zone,
                expected_unit="celsius",
            )
            if status == READ_OK and value is not None:
                return float(value)
        except Exception as err:
            self._record_error("forecast_bias_read", err)
        return 0.0

    # Minimum confidence threshold for early cutoff prediction to be acted on.
    _EARLY_CUTOFF_MIN_CONFIDENCE: float = 0.35
    # Absolute maximum residual rise the model may predict (mirrors AfterheatParameters cap).
    _EARLY_CUTOFF_MAX_C: float = 3.0
    # Allowed cutoff at minimum admissible confidence: at conf=0.35 only 0.5°C is safe
    # (worst-case room deficit limited to 0.5°C if afterheat fails).  At conf=1.0 the
    # full model cap (3.0°C) applies.  Linear interpolation in between.
    _EARLY_CUTOFF_MAX_C_AT_MIN_CONF: float = 0.5

    def read_early_cutoff_safe(self) -> tuple:
        """Read LE 2.0 RECOMMENDED_EARLY_CUTOFF (expected residual rise °C) via Read Gate.

        Returns ``(residual_rise_c, status)`` where:
        - ``residual_rise_c``: 0.0 when unavailable / cold-start / low-confidence / gate-rejected;
          non-negative float capped by a confidence-proportional limit otherwise.
        - ``status``: one of "valid", "not_available", "cold_start", "low_confidence",
          "missing", "stale", "superseded", "invalid".

        A versionised prior (``fallback_used=True``) always returns ``(0.0, "cold_start")``.
        Low evidence-quality (``confidence < 0.35``) always returns ``(0.0, "low_confidence")``.
        The returned value is capped proportionally to confidence so that a marginal-confidence
        prediction can only reduce the drive target by a small, low-risk amount.
        """
        if not self._enabled:
            return 0.0, "not_available"
        try:
            from ..decision.runtime_adapter import validated_prediction_read, READ_OK
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            cutoff_pred = preds.get(PredictionType.RECOMMENDED_EARLY_CUTOFF)
            if cutoff_pred is not None and getattr(cutoff_pred, "fallback_used", True):
                return 0.0, "cold_start"
            confidence = 0.0
            if cutoff_pred is not None:
                confidence = float(getattr(cutoff_pred, "confidence", 0.0) or 0.0)
                if confidence < self._EARLY_CUTOFF_MIN_CONFIDENCE:
                    return 0.0, "low_confidence"
            value, read_status = validated_prediction_read(
                preds,
                "early_cutoff",
                zone_id=self._zone,
                prediction_zone_id=self._zone,
                expected_unit="celsius",
            )
            if read_status == READ_OK and value is not None:
                value = float(max(0.0, value))
                # Confidence-proportional cap: limit the allowed cutoff magnitude to
                # reduce control risk at lower confidence.  At minimum admissible
                # confidence (0.35) only 0.5°C is permitted; at confidence=1.0 the
                # full model cap (3.0°C) applies.  Linear interpolation in between.
                conf_range = max(0.001, 1.0 - self._EARLY_CUTOFF_MIN_CONFIDENCE)
                t = max(0.0, min(1.0, (confidence - self._EARLY_CUTOFF_MIN_CONFIDENCE) / conf_range))
                conf_cap = (self._EARLY_CUTOFF_MAX_C_AT_MIN_CONF
                            + t * (self._EARLY_CUTOFF_MAX_C - self._EARLY_CUTOFF_MAX_C_AT_MIN_CONF))
                return min(value, conf_cap), "valid"
            return 0.0, self._TRUST_STATUS_MAP.get(read_status, "invalid")
        except Exception as err:
            self._record_error("early_cutoff_read", err)
        return 0.0, "not_available"

    def read_regime_safe(self) -> Optional[str]:
        """Return the regime string from the last shadow pipeline cycle, or None.

        Returns one of the ``Regime`` enum values (e.g. "active_heating",
        "afterheat", "passive_cooling", "stable", "disturbed", "unknown") or
        ``None`` when the shadow has not yet completed a cycle.

        Used by the coordinator to gate Early Cutoff hold starts: only
        ``"active_heating"`` allows a new hold; ``None`` / ``"unknown"`` / etc.
        preserve the deterministic baseline and prevent an invented regime.
        """
        if not self._enabled:
            return None
        try:
            zr = self._runtime._zone(self._zone)
            return getattr(zr, "last_regime", None)
        except Exception as err:
            self._record_error("regime_read", err)
        return None

    # ── Preheat / HeatRate read ──────────────────────────────────────────────
    # Minimum confidence for a HEAT_RATE prediction to contribute to preheat.
    _PREHEAT_MIN_CONFIDENCE: float = 0.35
    # Onset delay cold-start PRIOR: TRV command → measurable room heating response.
    # This is a versionable prior, not a permanent truth.  Once ONSET_DELAY learning
    # accumulates evidence, read_onset_delay_safe() will return a learned value.
    # Context factors that WILL influence the learned delay (future implementation):
    #   - time bucket (morning vs afternoon)
    #   - prior setback depth and duration (deep overnight setback → longer lag)
    #   - outdoor temperature and wind (cold system → longer warm-up of pipes)
    #   - device type / zone context
    #   - schedule transition type (from night to morning vs from eco to comfort)
    _ONSET_DELAY_PRIOR_MIN: float = 5.0
    # Maximum onset delay admitted as valid (sanity gate for learned values).
    # Raised from 30 → 45 min to match OnsetDelayParameters.max_onset_delay_min:
    # real central-heating supply lags can reach 40+ min after deep overnight setbacks.
    _ONSET_DELAY_MAX_MIN: float = 45.0
    # Conservative room heating rate for the deterministic baseline (no LE needed).
    # 1.5 °C/h is intentionally slow: errs toward starting earlier, not later.
    # Real rooms typically heat at 2–4 °C/h; this under-estimates so the room is
    # warm ON TIME even for sluggish systems.  More aggressive than 0 (LE v1 cold
    # start), but NEVER invented — it's a principled physical prior.
    _DETERMINISTIC_HEAT_RATE_C_PER_H: float = 1.5
    # Safe heat-loss prior used when HEAT_LOSS_RATE prediction is absent, stale,
    # fallback-only, or low-confidence.  Choosing 0.0 would silently assume perfect
    # insulation (net_rate = heat_rate), systematically underestimating preheat
    # duration.  0.3 °C/h is a lower-bound physical prior: even well-insulated rooms
    # lose heat; using it makes the preheat estimate safer while remaining sub-linear
    # relative to the deterministic baseline net-rate (1.5 °C/h).
    _HEAT_LOSS_PRIOR_C_PER_H: float = 0.3
    # Absolute maximum preheat allowed from LE2 (prevents runaway early starts).
    _PREHEAT_MAX_MIN: float = 180.0
    # Minimum temperature deficit that justifies a preheat command (°C).
    _PREHEAT_MIN_DELTA_C: float = 0.5
    # Legacy alias kept for any external references; remove in v1.2
    _PREHEAT_FALLBACK_MIN: float = 60.0

    @classmethod
    def compute_deterministic_preheat_baseline(
        cls, current_temp: Optional[float], comfort_temp: float
    ) -> tuple:
        """Deterministic preheat baseline — no learning required.

        Inputs: ``current_temp`` (°C, may be None), ``comfort_temp`` (°C).

        Behaviour:
        - ``current_temp is None`` → ``(0.0, "unavailable")`` (cannot compute without sensor).
        - ``deficit ≤ _PREHEAT_MIN_DELTA_C`` → ``(0.0, "target_reached")``.
        - Otherwise: ``command_lead = room_heating + onset_prior`` where
          ``room_heating = deficit / _DETERMINISTIC_HEAT_RATE_C_PER_H × 60``.

        Conservative rate (1.5 °C/h) ensures the room is warm ON TIME for
        slow systems; fast rooms finish slightly early, which is safe.
        Result is capped at ``_PREHEAT_MAX_MIN``.

        TRV-only / no outdoor sensor: works identically (no outdoor input needed).
        """
        if current_temp is None:
            return 0.0, "unavailable"
        deficit = comfort_temp - current_temp
        if deficit <= cls._PREHEAT_MIN_DELTA_C:
            return 0.0, "target_reached"
        room_heating = (deficit / cls._DETERMINISTIC_HEAT_RATE_C_PER_H) * 60.0
        total = min(room_heating + cls._ONSET_DELAY_PRIOR_MIN, cls._PREHEAT_MAX_MIN)
        return round(total, 1), "deterministic_baseline"

    def read_onset_delay_safe(self) -> tuple:
        """Read LE 2.0 ONSET_DELAY (minutes): TRV command → measurable room response.

        Returns ``(delay_min, status)`` where status is:
        - ``"valid"``: learned value, confidence-gated.
        - ``"cold_start_prior"``: no evidence yet; returns ``_ONSET_DELAY_PRIOR_MIN``.
        - ``"not_available"``: shadow disabled.

        Context factors registered for future learning (not yet implemented):
        ``time_bucket``, ``setback_depth_c``, ``setback_duration_h``,
        ``outdoor_temp_c``, ``schedule_transition``, ``device_type``.
        These are tracked per episode and stored alongside onset_delay measurements
        so the model can learn situation-specific delays (e.g. morning after deep
        overnight setback is systematically longer than afternoon re-heat).
        """
        if not self._enabled:
            return self._ONSET_DELAY_PRIOR_MIN, "not_available"
        try:
            from ..contracts import PredictionType
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            od_pred = preds.get(PredictionType.ONSET_DELAY)
            if od_pred is None:
                return self._ONSET_DELAY_PRIOR_MIN, "cold_start_prior"
            fallback_used = getattr(od_pred, "fallback_used", True)
            if fallback_used:
                return self._ONSET_DELAY_PRIOR_MIN, "cold_start_prior"
            confidence = float(getattr(od_pred, "confidence", 0.0) or 0.0)
            if confidence < self._PREHEAT_MIN_CONFIDENCE:
                return self._ONSET_DELAY_PRIOR_MIN, "cold_start_prior"
            delay = float(od_pred.values.get("onset_delay", self._ONSET_DELAY_PRIOR_MIN) or self._ONSET_DELAY_PRIOR_MIN)
            delay = max(0.0, min(delay, self._ONSET_DELAY_MAX_MIN))
            return round(delay, 1), "valid"
        except Exception as err:
            self._record_error("onset_delay_read", err)
        return self._ONSET_DELAY_PRIOR_MIN, "cold_start_prior"

    def read_preheat_minutes_safe(
        self,
        current_temp: Optional[float],
        comfort_temp: float,
        *,
        outdoor_temp: Optional[float] = None,
    ) -> tuple:
        """Read LE 2.0 preheat duration (command lead time in minutes).

        Returns ``(minutes, status)`` where:
        - ``minutes``: command lead time ≥ 0 (always the right value for the status).
        - ``status``: one of "valid", "low_confidence", "deterministic_baseline",
          "target_reached", "unavailable", "not_available".

        Three semantically distinct durations are kept separate (A = B + C):
        - **A Command Lead Time**: returned ``minutes``
        - **B Effective Heating Onset Delay**: from ``read_onset_delay_safe()``
          (cold-start prior = 5 min; adaptive once evidence accumulates)
        - **C Effective Room Heating Duration**: ``deficit / net_heat_rate × 60``

        Fallback hierarchy (no LE v1 reads at any level):
        1. LE 2.0 valid prediction (status "valid") — adaptive, confidence-gated
        2. LE 2.0 low-confidence → deterministic baseline (status "deterministic_baseline")
        3. No LE 2.0 evidence → deterministic baseline (status "deterministic_baseline")
        4. No temperature data → (0.0, "unavailable")
        5. Target already reached → (0.0, "target_reached")
        6. Shadow disabled → deterministic baseline (status "deterministic_baseline")
        """
        if not self._enabled:
            return self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)
        try:
            from ..contracts import PredictionType
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}

            hr_pred = preds.get(PredictionType.HEAT_RATE)
            hl_pred = preds.get(PredictionType.HEAT_LOSS_RATE)
            af_pred = preds.get(PredictionType.EXPECTED_OVERSHOOT)

            if hr_pred is None:
                return self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)

            fallback_used = getattr(hr_pred, "fallback_used", True)
            confidence = float(getattr(hr_pred, "confidence", 0.0) or 0.0)
            heat_rate_c_per_h = float(hr_pred.values.get("heat_rate", 0.0) or 0.0)

            if fallback_used or heat_rate_c_per_h <= 0:
                return self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)

            if current_temp is None:
                return 0.0, "unavailable"

            deficit = comfort_temp - current_temp
            if deficit <= self._PREHEAT_MIN_DELTA_C:
                return 0.0, "target_reached"

            if confidence < self._PREHEAT_MIN_CONFIDENCE:
                return self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)

            # HEAT_LOSS_RATE gate: absent / fallback / stale / low-confidence prediction
            # must NOT be treated as 0.0 (which would imply perfect insulation).
            # Use the versioned safe prior instead so preheat is never under-estimated.
            _hl_valid = (
                hl_pred is not None
                and not getattr(hl_pred, "fallback_used", True)
                and float(getattr(hl_pred, "confidence", 0.0) or 0.0) >= self._PREHEAT_MIN_CONFIDENCE
            )
            heat_loss_c_per_h = (
                float(hl_pred.values.get("heat_loss_rate", 0.0) or 0.0)
                if _hl_valid
                else self._HEAT_LOSS_PRIOR_C_PER_H
            )
            afterheat_c = float(
                af_pred.values.get("expected_overshoot", 0.0) if af_pred else 0.0) or 0.0

            # Effective Room Heating Duration (C):
            #   net_rate = heat_rate − heat_loss  (min 0.2 C/h floor)
            #   effective_deficit = deficit − afterheat (early-cutoff contribution reduces this)
            #   C = effective_deficit / net_rate × 60
            net_rate = max(heat_rate_c_per_h - heat_loss_c_per_h, 0.2)
            effective_deficit = max(0.0, deficit - max(0.0, afterheat_c))
            room_heating_min = (effective_deficit / net_rate) * 60.0

            # Confidence-proportional cap (0.35–0.6 band): interpolate between
            # deterministic_baseline and PREHEAT_MAX_MIN as confidence rises.
            if confidence < 0.6:
                conf_range = max(0.001, 0.6 - self._PREHEAT_MIN_CONFIDENCE)
                t = max(0.0, min(1.0, (confidence - self._PREHEAT_MIN_CONFIDENCE) / conf_range))
                baseline_min, _ = self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)
                cap = baseline_min + t * (self._PREHEAT_MAX_MIN - baseline_min)
                room_heating_min = min(room_heating_min, cap)

            room_heating_min = max(0.0, min(room_heating_min, self._PREHEAT_MAX_MIN))

            # Command Lead Time (A) = Room Heating Duration (C) + Onset Delay (B).
            # Onset delay comes from read_onset_delay_safe() — adaptive when learned,
            # otherwise returns the cold-start prior.  Never folded into heat rate.
            onset_delay, _ = self.read_onset_delay_safe()
            total_min = room_heating_min + onset_delay
            total_min = min(total_min, self._PREHEAT_MAX_MIN)

            # "valid" requires both HR and HL from learned LE 2.0 predictions.
            # "valid_hl_prior" signals that HR is learned but HL uses the safe prior.
            # A fallback must never be labelled as "valid".
            _status = "valid" if _hl_valid else "valid_hl_prior"
            return round(total_min, 1), _status
        except Exception as err:
            self._record_error("preheat_read", err)
        return self.compute_deterministic_preheat_baseline(current_temp, comfort_temp)

    def read_heat_rate_safe(self) -> tuple:
        """Read LE 2.0 HEAT_RATE (°C/h) for display and diagnostics.

        Returns ``(rate_c_per_h, status)`` where:
        - ``rate_c_per_h``: 0.0 when unavailable / cold-start / invalid.
        - ``status``: one of "valid", "not_available", "cold_start", "low_confidence".

        Unit is always °C/h.  Callers needing °C/min must divide by 60.
        Only the effective room heating rate is returned; onset delay is NOT
        included in this rate and must never be included in heat-rate learning.
        """
        if not self._enabled:
            return 0.0, "not_available"
        try:
            from ..contracts import PredictionType
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            hr_pred = preds.get(PredictionType.HEAT_RATE)
            if hr_pred is None:
                return 0.0, "cold_start"
            fallback_used = getattr(hr_pred, "fallback_used", True)
            if fallback_used:
                return 0.0, "cold_start"
            confidence = float(getattr(hr_pred, "confidence", 0.0) or 0.0)
            if confidence < self._PREHEAT_MIN_CONFIDENCE:
                return 0.0, "low_confidence"
            rate = float(hr_pred.values.get("heat_rate", 0.0) or 0.0)
            if rate <= 0:
                return 0.0, "cold_start"
            return rate, "valid"
        except Exception as err:
            self._record_error("heat_rate_read", err)
        return 0.0, "not_available"

    def read_heat_loss_rate_safe(self) -> tuple:
        """Read LE 2.0 HEAT_LOSS_RATE (°C/h) for display.

        Returns ``(rate_c_per_h, status)``.
        """
        if not self._enabled:
            return 0.0, "not_available"
        try:
            from ..contracts import PredictionType
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            hl_pred = preds.get(PredictionType.HEAT_LOSS_RATE)
            if hl_pred is None:
                return 0.0, "cold_start"
            if getattr(hl_pred, "fallback_used", True):
                return 0.0, "cold_start"
            rate = float(hl_pred.values.get("heat_loss_rate", 0.0) or 0.0)
            if rate < 0:
                return 0.0, "cold_start"
            return rate, "valid"
        except Exception as err:
            self._record_error("heat_loss_rate_read", err)
        return 0.0, "not_available"

    # Confidence thresholds for the coef_int blend ramp:
    #   BLEND_MIN  = gate-6 threshold (blend weight = 0 → fully default)
    #   BLEND_FULL = confidence at which blend weight reaches 1 → fully learned
    _BLEND_FULL_CONFIDENCE: float = 0.80
    # Max blend weight when HeatLoss has no outdoor context (relative_only).
    # Without outdoor context the general rate averages all episodes across ΔT — it may
    # understate losses in cold weather and overstate them in mild weather.  Capping the
    # blend at 0.5 ensures the default coef_int always contributes at least 50 %.
    _BLEND_WEIGHT_RELATIVE_ONLY_CAP: float = 0.50

    def read_tpi_coefficients_safe(self) -> tuple:
        """Return LE2-derived TPI coefficients with blend diagnostics (5-tuple).

        Returns (coef_int_used, coef_ext, heat_loss_pred_or_none, status, diag) where:
          coef_int_used  — blended coef_int applied to compute_tpi()
          coef_ext       — TPI_COEF_EXT_DEFAULT (always static)
          heat_loss_pred — learned rate in °C/h, or None for all non-"valid_le2" statuses
          status         — gate result string (see below)
          diag           — dict with fields for trace/audit (never raises)

        Classification: Variante B — Bounded Adaptive Heuristic
        -------------------------------------------------------
        coef_int is derived as heat_loss_rate / heat_rate.  This is NOT a rigorous
        derivation of a proportional control gain from first principles.

        The steady-state analysis shows: to hold the room at target against a passive
        cooling rate of heat_loss_rate, the required duty is heat_loss_rate / heat_rate.
        Using this as coef_int implies "the duty needed per °C error scales with the
        same ratio as steady-state maintenance duty."  This is directionally correct —
        lossier rooms need more aggressive control — but conflates the steady-state
        operating point with the error-correction gain.  The two are related but not
        identical.

        Safeguards that make Variante B acceptable:
          - coef_int is clamped to [0.15, 1.2] (estimate_coefficients)
          - only applied when both models exceed confidence 0.35
          - linearly blended with TPI_COEF_INT_DEFAULT over confidence [0.35 → 0.80]
          - blend is capped at 0.50 when HeatLoss has no outdoor context (relative_only)
          - reverts to deterministic defaults on stale / invalid / error

        HeatLoss outdoor context
        ------------------------
        HEAT_LOSS_RATE may be learned from episodes at varying outdoor temperatures.
        When no outdoor sensor is available (relative_only = True in reason_codes),
        the general bucket averages episodes across all observed ΔT.  This rate is a
        valid local observation but may not generalize to the current outdoor condition.
        The blend cap (_BLEND_WEIGHT_RELATIVE_ONLY_CAP) limits TPI influence of
        context-unaware heat loss estimates.

        Physical units
        --------------
        heat_rate      [°C/h] — UNIT_C_PER_H, verified by Gate 4
        heat_loss_rate [°C/h] — UNIT_C_PER_H, verified by Gate 4
        coef_int       [dimensionless = °C/h / °C/h], acts as 1/°C in TPI formula
        coef_ext       [dimensionless], static TPI_COEF_EXT_DEFAULT

        coef_ext rationale (Variante A, static)
        ----------------------------------------
        HEAT_LOSS_RATE is not normalised by the indoor-outdoor delta at episode time.
        Deriving coef_ext = coef_int / 50 and then multiplying by (target − outdoor_now)
        would implicitly double-scale the outdoor effect.  The static default avoids this.
        For TRV-only setups without an outdoor sensor, compute_tpi() skips the ext term.

        Validity gates (sequential; first failure returns deterministic defaults + diag):
          1. shadow enabled                          → le2_disabled
          2. both predictions present                → prediction_missing
          3. neither prediction has fallback_used    → cold_start
          4. UNIT_C_PER_H for both quantities        → invalid_unit
          5. no "stale"/"superseded" in warnings     → stale_or_superseded
          6. min(hr_conf, hl_conf) >= 0.35           → low_confidence
          7. both rates finite and > 0               → invalid_value

        After all gates pass, coef_int_used is computed as:
          blend       = clamp((conf − 0.35) / (0.80 − 0.35), 0, 1)
          [if relative_only: blend = min(blend, 0.50)]
          coef_int_used = blend × coef_int_raw + (1 − blend) × TPI_COEF_INT_DEFAULT

        status values:
          "valid_le2"           — all gates passed; coef_int is LE2-adaptive (possibly blended)
          "prediction_missing"  — one or both predictions absent (gate 2)
          "cold_start"          — fallback_used == True; prior-based, not learned (gate 3)
          "invalid_unit"        — unit is not UNIT_C_PER_H (gate 4)
          "stale_or_superseded" — stale or superseded warning present (gate 5)
          "low_confidence"      — confidence below 0.35 (gate 6)
          "invalid_value"       — rate ≤ 0 or non-finite (gate 7)
          "le2_disabled"        — shadow disabled
          "not_available"       — unexpected runtime error

        Never reads LE v1 _heat_loss_ema.
        """
        from ...const import TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, UNIT_C_PER_H
        _default_diag: dict = {
            "coef_int_default": TPI_COEF_INT_DEFAULT,
            "coef_int_candidate": TPI_COEF_INT_DEFAULT,  # gate failures: candidate = default
            "coef_int_learned": None, "blend_weight": 0.0,
            "relative_only": False, "outdoor_context_available": False,
            "heat_rate_c_per_h": None, "heat_loss_c_per_h": None,
            "hr_confidence": None, "hl_confidence": None,
        }
        if not self._enabled:
            return TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None, "le2_disabled", _default_diag
        try:
            from ..contracts import PredictionType
            zr = self._runtime._zone(self._zone)
            preds = getattr(zr, "last_predictions", {}) or {}
            hr_pred = preds.get(PredictionType.HEAT_RATE)
            hl_pred = preds.get(PredictionType.HEAT_LOSS_RATE)
            # Gate 2: both predictions present
            if hr_pred is None or hl_pred is None:
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "prediction_missing", _default_diag)
            # Gate 3: not a fallback/cold-start prior
            if getattr(hr_pred, "fallback_used", True) or getattr(hl_pred, "fallback_used", True):
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "cold_start", _default_diag)
            # Gate 4: unit must be exactly UNIT_C_PER_H (central contract)
            _hr_units = getattr(hr_pred, "units", {}) or {}
            _hl_units = getattr(hl_pred, "units", {}) or {}
            if _hr_units.get("heat_rate") != UNIT_C_PER_H \
                    or _hl_units.get("heat_loss_rate") != UNIT_C_PER_H:
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "invalid_unit", _default_diag)
            # Gate 5: not stale or superseded
            _hr_warns = getattr(hr_pred, "warnings", ()) or ()
            _hl_warns = getattr(hl_pred, "warnings", ()) or ()
            if ("stale" in _hr_warns or "superseded" in _hr_warns
                    or "stale" in _hl_warns or "superseded" in _hl_warns):
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "stale_or_superseded", _default_diag)
            # Gate 6: confidence above threshold (min of both)
            hr_conf = float(getattr(hr_pred, "confidence", 0.0) or 0.0)
            hl_conf = float(getattr(hl_pred, "confidence", 0.0) or 0.0)
            conf = min(hr_conf, hl_conf)
            if conf < self._PREHEAT_MIN_CONFIDENCE:
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "low_confidence", {**_default_diag, "hr_confidence": hr_conf,
                                           "hl_confidence": hl_conf})
            # Gate 7: physically plausible, finite rates (nan/inf bypass <= 0 in Python)
            heat_rate = float(hr_pred.values.get("heat_rate", 0.0) or 0.0)
            heat_loss = float(hl_pred.values.get("heat_loss_rate", 0.0) or 0.0)
            if not (math.isfinite(heat_rate) and heat_rate > 0
                    and math.isfinite(heat_loss) and heat_loss > 0):
                return (TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None,
                        "invalid_value", {**_default_diag, "hr_confidence": hr_conf,
                                          "hl_confidence": hl_conf,
                                          "heat_rate_c_per_h": heat_rate,
                                          "heat_loss_c_per_h": heat_loss})
            # Outdoor context: HeatLoss learned without outdoor sensor cannot claim
            # full authority because the general rate mixes all episode ΔT values.
            _hl_reasons = getattr(hl_pred, "reason_codes", ()) or ()
            relative_only = "relative_only" in _hl_reasons \
                or "outdoor_temp" in (getattr(hl_pred, "missing_evidence", ()) or ())
            # Blend: linearly ramp from 0 (at gate-6 threshold) to 1 (at full confidence).
            # This prevents abrupt step change from default→learned at the threshold.
            blend = max(0.0, min(1.0,
                (conf - self._PREHEAT_MIN_CONFIDENCE)
                / max(0.001, self._BLEND_FULL_CONFIDENCE - self._PREHEAT_MIN_CONFIDENCE)))
            if relative_only:
                blend = min(blend, self._BLEND_WEIGHT_RELATIVE_ONLY_CAP)
            from ...tpi import estimate_coefficients
            coef_int_raw, _ = estimate_coefficients(heat_rate, heat_loss)
            coef_int_used = blend * coef_int_raw + (1.0 - blend) * TPI_COEF_INT_DEFAULT
            coef_ext = TPI_COEF_EXT_DEFAULT
            diag: dict = {
                "coef_int_default": TPI_COEF_INT_DEFAULT,
                "coef_int_candidate": round(coef_int_used, 4),  # blended; before coordinator smoothing
                "coef_int_learned": round(coef_int_raw, 4),
                "blend_weight": round(blend, 4),
                "relative_only": relative_only,
                "outdoor_context_available": not relative_only,
                "heat_rate_c_per_h": round(heat_rate, 4),
                "heat_loss_c_per_h": round(heat_loss, 4),
                "hr_confidence": round(hr_conf, 4),
                "hl_confidence": round(hl_conf, 4),
            }
            return coef_int_used, coef_ext, round(heat_loss, 4), "valid_le2", diag
        except Exception as err:
            self._record_error("tpi_coefficients_read", err)
        return TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT, None, "not_available", _default_diag

    async def async_save_if_due(self) -> None:
        if not self._enabled:
            return
        try:
            await self._runtime.async_save_if_due()
        except Exception as err:
            self._record_error("save", err)
        await self._async_save_adaptation_history_safe()
        await self._async_save_application_lifecycle_safe()
        await self._async_save_episode_history_safe()
        await self._async_save_support_critical_events_safe()
        await self._async_save_research_daily_safe()

    async def async_flush(self) -> None:
        try:
            await self._runtime.async_flush()
        except Exception as err:
            self._record_error("flush", err)

    async def async_unload(self) -> None:
        self._enabled = False
        try:
            await self._runtime.async_unload()
        except Exception as err:
            self._record_error("unload", err)
        # Flush adaptation history (best-effort; saves if dirty)
        await self._async_save_adaptation_history_safe()
        # Flush application lifecycle state (best-effort; saves if dirty)
        await self._async_save_application_lifecycle_safe()
        # Flush episode history (best-effort; saves if dirty)
        await self._async_save_episode_history_safe()
        # Flush support critical events (best-effort; saves if dirty).
        # force=True: unload must not lose genuinely accumulated data
        # just because the debounce window hasn't elapsed yet.
        await self._async_save_support_critical_events_safe(force=True)
        # Flush research daily buckets (best-effort; saves if dirty).
        # force=True — same rationale as above.
        await self._async_save_research_daily_safe(force=True)

    def read_outcome_score_safe(self) -> tuple:
        """Read LE 2.0 outcome quality score (0–100 %) for display.

        Returns ``(score_pct, status)`` where:
        - ``score_pct``: float 0–100 when enough accepted outcome samples exist, else ``None``.
        - ``status``: "valid", "collecting_data", "cold_start", or "not_available".

        Requires at least ``_MIN_OUTCOME_DISPLAY_SAMPLES`` accepted samples before returning a
        score; fewer samples produce an unreliable average that must not be shown as a fact.
        Uses ``OutcomeModel.predict()`` — pure read, no side effects, no state mutation.
        """
        if not self._enabled:
            return None, "not_available"
        try:
            from ..models.outcome import OutcomePredictionContext
            zr = self._runtime._zone(self._zone)
            model = zr.orchestrator.models.get("outcome")
            if model is None:
                return None, "not_available"
            pred = model.predict(OutcomePredictionContext())
            if getattr(pred, "fallback_used", True):
                return None, "cold_start"
            evidence = getattr(pred, "evidence_count", 0) or 0
            if evidence < self._MIN_OUTCOME_DISPLAY_SAMPLES:
                return None, "collecting_data"
            quality = pred.values.get("outcome_quality")
            if quality is None:
                return None, "cold_start"
            return round(float(quality) * 100, 1), "valid"
        except Exception as err:
            self._record_error("outcome_score_read", err)
        return None, "not_available"

    def read_outcome_attributes_safe(self) -> dict:
        """Read LE 2.0 outcome diagnostics for entity extra_state_attributes.

        Pure read — no side effects, no state mutation.
        """
        if not self._enabled:
            return {"data_status": "not_available", "data_source": "current_learning_engine"}
        try:
            zr = self._runtime._zone(self._zone)
            model = zr.orchestrator.models.get("outcome")
            if model is None:
                return {"data_status": "not_available", "data_source": "current_learning_engine"}
            diag = model.diagnostics()
            sample_count = diag.sample_counts.get("general", 0)
            _min = self._MIN_OUTCOME_DISPLAY_SAMPLES
            if sample_count == 0:
                outcome_status = "cold_start"
            elif sample_count < _min:
                outcome_status = "collecting_data"
            elif sample_count < 5:
                outcome_status = "learning"
            else:
                outcome_status = "learned"
            attrs: dict = {
                "data_source": "current_learning_engine",
                "outcome_status": outcome_status,
                "sample_count": sample_count,
                "required_samples": _min,
                "reached_rate": diag.reached_rate,
                "timeout_rate": diag.timeout_rate,
                "overshoot_rate": diag.overshoot_rate,
                "last_update": diag.last_update_ts,
            }
            if sample_count < _min:
                attrs["reason"] = "insufficient_outcome_samples"
            return attrs
        except Exception as err:
            self._record_error("outcome_attrs_read", err)
        return {"data_status": "not_available", "data_source": "current_learning_engine"}

    # Minimum accepted outcome samples before the outcome score is displayed.
    # Fewer samples produce an unreliable average that must not be shown as a fact.
    _MIN_OUTCOME_DISPLAY_SAMPLES: int = 3

    def learning_progress_safe(self) -> tuple[float, dict]:
        """Learning progress from zone-specific real model updates (0.0-100.0 %, attributes).

        Delegates to the pure, calibrated ``compute_learning_progress()``
        (learning_progress.py) — see that module's docstring for the full
        rationale and the six weighted components (data volume, thermal
        regime coverage, situation diversity, clean-episode ratio, outcome
        validation, model confidence) plus the confounder penalty. This
        method's only job is to gather each model's REAL, already-tracked
        diagnostics/confidence into a ``ModelSignal`` per model — no new
        data is invented here.

        Only episode-driven model updates count toward ``accepted_updates``;
        bootstrap/prior-only state never contributes (a model with zero
        accepted updates is excluded from the confidence-average component).

        Returns ``(progress_pct, attributes)`` where ``progress_pct`` is 0.0
        when no real heating data has been accumulated yet.
        """
        if not self._enabled:
            from .learning_progress import cold_learning_progress_result
            return 0.0, {**cold_learning_progress_result("no_completed_episodes_yet"), "reason": "learning_disabled"}
        try:
            from .learning_progress import ModelSignal, compute_learning_progress

            zr = self._runtime._zone(self._zone)
            counts = dict(zr.model_update_counts)

            signals: dict[str, Any] = {}
            for model_name in ("heat_rate", "heat_loss", "afterheat", "outcome", "onset_delay"):
                accepted = counts.get(model_name, 0)
                sample_counts: dict = {}
                rejection_counts: dict = {}
                outlier_counts: dict = {}
                confidence_value: Optional[float] = None
                general_data_quality: Optional[float] = None
                timeout_rate: Optional[float] = None
                overshoot_rate: Optional[float] = None
                partial_ratio: Optional[float] = None
                try:
                    model = zr.orchestrator.models.get(model_name)
                    if model is not None:
                        diag = model.diagnostics()
                        sample_counts = dict(getattr(diag, "sample_counts", None) or {})
                        rejection_counts = dict(getattr(diag, "rejection_counts", None) or {})
                        outlier_counts = dict(getattr(diag, "outlier_counts", None) or {})
                        if model_name == "outcome":
                            general_data_quality = getattr(diag, "general_data_quality", None)
                            timeout_rate = getattr(diag, "timeout_rate", None)
                            overshoot_rate = getattr(diag, "overshoot_rate", None)
                            fc, pc = getattr(diag, "full_partial", (0, 0))
                            partial_ratio = (pc / (fc + pc)) if (fc + pc) > 0 else 0.0
                        if accepted > 0:
                            confidence_value = model.confidence().value
                except Exception:
                    pass  # a single model's diagnostics failure must not block the others
                signals[model_name] = ModelSignal(
                    accepted_updates=accepted,
                    sample_counts=sample_counts,
                    rejection_counts=rejection_counts,
                    outlier_counts=outlier_counts,
                    confidence_value=confidence_value,
                    general_data_quality=general_data_quality,
                    timeout_rate=timeout_rate,
                    overshoot_rate=overshoot_rate,
                    partial_ratio=partial_ratio,
                )

            progress_pct, attrs = compute_learning_progress(signals)
            self._maybe_record_research_daily_progress_sample_safe(progress_pct, attrs)
            return progress_pct, attrs
        except Exception as err:
            self._record_error("learning_progress", err)
            from .learning_progress import cold_learning_progress_result
            return 0.0, cold_learning_progress_result("calculation_error")

    def _maybe_record_research_daily_progress_sample_safe(self, progress_pct: Any, attrs: Any) -> None:
        """Aggregate ONE learning-progress/confidence sample into the
        current day's Research Daily Bucket.

        Called ONLY from the successful ``compute_learning_progress()``
        branch of ``learning_progress_safe()`` above — never for the
        disabled-shadow early return or the calculation-error except
        branch, since those are "we could not compute this" guards, not
        genuine progress readings; aggregating their hard-coded 0.0 would
        corrupt the daily min with a value that was never actually
        measured. Reads exactly two already-computed, already public-safe
        scalar values: ``progress_pct`` (0-100 %) and
        ``attrs["model_confidence_score"]`` (the REAL averaged per-model
        confidence score, 0.0-1.0 — never ``attrs["confidence_level"]``,
        which is a string band label like "medium", not a number, and is
        deliberately never guessed into one).

        This is the ONLY call site for this method — no coordinator/sensor/
        export change was made to call it more than once per
        ``learning_progress_safe()`` invocation. Repeated calls with an
        unchanged value are effectively free: unlike a naive counter, this
        never spams storage, because
        ``record_research_daily_observation_safe()`` only marks the bucket
        dirty when the resulting merged payload genuinely differs (see its
        own docstring) — an unchanged last_pct/confidence_last produces a
        byte-identical serialized bucket, so a redundant call with the same
        reading never triggers an extra save.

        Bucket date is derived from this controller's own injected Clock
        (``self._utcnow_iso()``, timezone-aware, deterministic under
        tests) — never a bare wall-clock current-time helper. Never raises: a
        malformed/out-of-range value or any internal failure is captured
        in ``_research_daily_last_error`` and swallowed — never
        propagated, never touches ``_enabled``.
        """
        try:
            from datetime import datetime
            from ..research_daily_schemas import ResearchDailyObservation

            if not isinstance(progress_pct, (int, float)) or isinstance(progress_pct, bool):
                return
            if not math.isfinite(progress_pct) or not (0.0 <= progress_pct <= 100.0):
                return  # out-of-range/invalid -> not a real percent, do not guess/clamp

            confidence: Optional[float] = None
            if isinstance(attrs, Mapping):
                raw_confidence = attrs.get("model_confidence_score")
                if (
                    isinstance(raw_confidence, (int, float))
                    and not isinstance(raw_confidence, bool)
                    and math.isfinite(raw_confidence)
                    and 0.0 <= raw_confidence <= 1.0
                ):
                    confidence = float(raw_confidence)

            now_utc = datetime.fromisoformat(self._utcnow_iso())
            bucket_date = now_utc.date().isoformat()
            observation = ResearchDailyObservation(
                bucket_date=bucket_date,
                learning_progress_pct=float(progress_pct),
                confidence=confidence,
            )
            self.record_research_daily_observation_safe(observation)
        except Exception as err:
            self._research_daily_last_error = str(err)

    def diagnostics(self) -> dict:
        h = self._runtime.health()
        return {
            "mode": h.mode, "initialized": h.initialized, "dirty": h.dirty,
            "last_cycle_ts": h.last_cycle_ts, "last_save": h.last_save,
            "open_decisions": h.open_decisions, "open_comparisons": h.open_comparisons,
            "model_update_total": h.model_update_total,
            "model_update_counts": dict(h.model_update_counts),
            "prediction_snapshots": h.prediction_snapshots,
            "learning_errors": self._errors, "enabled": self._enabled,
            "control_enabled": h.control_enabled,
            "control_policy_version": h.control_policy_version,
            "control_applied": h.control_applied, "control_fallback": h.control_fallback,
            "control_rejections": dict(h.control_rejections),
            "reversion_count": h.reversion_count,
            "decision_trace": self._last_trace,
        }

    def _record_error(self, kind: str, err: Exception) -> None:
        self._errors += 1
        sig = f"{kind}:{type(err).__name__}"
        n = self._error_signatures.get(sig, 0)
        self._error_signatures[sig] = n + 1
        if n % _ERROR_LOG_THROTTLE == 0:  # throttle identical errors
            _LOGGER.warning("ThermoSmart Learning Engine %s error (zone hidden): %s", kind,
                            type(err).__name__)
