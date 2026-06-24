"""Runtime capture for LE 2.0 shadow foundation (pure Python).

Materialises the actually-available ThermoSmart signals into typed snapshots and
a stable per-zone decision identity, plus a bounded command ledger used to tell
ThermoSmart's own setpoint writes (and device echoes) apart from real user
corrections. No HA imports, no service calls, no control effect — all inputs are
injected as typed values; timestamps are passed in.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping, Optional, Sequence

from ..contracts import DataQuality, DecisionId, Measurement


class DecisionType(Enum):
    NORMAL = "normal"
    PREHEAT = "preheat"
    BOOST = "boost"
    IDLE = "idle"
    UNKNOWN = "unknown"


class CommandType(Enum):
    SETPOINT = "setpoint"
    MODE = "mode"


class CommandStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    EXPIRED = "expired"


# -- typed runtime input ------------------------------------------------------

@dataclass(frozen=True)
class ScheduleTarget:
    comfort_time_utc: str          # next planned comfort-achievement time (UTC iso)
    comfort_temperature_c: float


@dataclass(frozen=True)
class ControllerDecisionInput:
    """The existing ThermoSmart decision as observed (never mutated)."""
    decision_type: DecisionType = DecisionType.UNKNOWN
    target_c: Optional[float] = None
    trv_setpoint_c: Optional[float] = None
    boost_offset_c: Optional[float] = None
    boost_active: bool = False
    preheat_active: bool = False
    fallback_active: bool = False
    mode: Optional[str] = None
    superseded: bool = False
    decision_token: Optional[str] = None   # opaque controller token if available


@dataclass(frozen=True)
class SentCommand:
    command_id: str
    command_type: CommandType
    trv_binding_id: str
    value: float
    sent_ts: str
    echo_window_s: float = 120.0


@dataclass(frozen=True)
class ManualCorrectionObservation:
    event_id: str
    prior_target_c: float
    new_target_c: float
    source: str                    # raw CorrectionSource value
    action_type: str               # raw CorrectionActionType value
    ts: str
    trv_binding_id: Optional[str] = None


@dataclass(frozen=True)
class BoostDispatchRecord:
    """Authoritative boost dispatch result for one decision, sourced from the live
    ``LiveDecisionRecord`` (B2b-1). The single source of truth for what boost was
    actually written — ``boost_applied_c == 0.0`` is distinct from "not dispatched"
    (``boost_applied_c is None`` / ``dispatch_status == 'not_attempted'``)."""
    decision_id: str
    boost_candidate_c: Optional[float] = None
    boost_applied_c: Optional[float] = None
    baseline_setpoint_c: Optional[float] = None
    final_setpoint_c: Optional[float] = None
    effective_setpoint_min_c: Optional[float] = None
    effective_setpoint_max_c: Optional[float] = None
    dispatch_status: str = "not_attempted"
    outcome_eligible: bool = False
    outcome_reliability: str = "none"
    start_deficit_c: Optional[float] = None
    expected_non_boost_duration_s: Optional[float] = None
    # B2b-3c: authoritative boost evaluation state + control type (for baseline authority).
    boost_evaluation_status: str = "unknown"
    device_control_type: str = "setpoint"      # "setpoint" | "direct_valve"
    # B2b-4b: per-device effective setpoints written + target counts (multi-TRV attribution).
    effective_setpoints: tuple = ()
    targets_total: int = 0
    targets_failed: int = 0


@dataclass(frozen=True)
class RuntimeCycleInput:
    """Everything one passive runtime cycle needs, typed and injected."""
    zone_id: str
    ts: str                                   # UTC iso timestamp of this cycle
    target_c: Optional[float] = None
    trv_setpoint_c: Optional[float] = None
    indoor_temp: Optional[Measurement] = None
    schedule: Optional[ScheduleTarget] = None
    mode: Optional[str] = None
    controller_demand: Optional[bool] = None     # controller calling for heat (if known)
    tpi_on: Optional[bool] = None
    controller_decision: ControllerDecisionInput = field(
        default_factory=ControllerDecisionInput)
    trv_setpoints: Mapping[str, float] = field(default_factory=dict)  # binding -> setpoint
    # B2b-4e: live per-device availability (binding -> "available"|"unavailable"|"unknown")
    device_states: Mapping[str, str] = field(default_factory=dict)
    outdoor_temp: Optional[Measurement] = None
    valve_opening: Optional[Measurement] = None
    hvac_action: Optional[str] = None
    window_open: Optional[bool] = None
    forecast_high: Optional[float] = None
    heating_failure: bool = False
    sent_commands: tuple[SentCommand, ...] = ()
    manual_corrections: tuple[ManualCorrectionObservation, ...] = ()
    legacy_preheat_start_utc: Optional[str] = None
    legacy_boost_offset_c: Optional[float] = None
    legacy_early_cutoff_utc: Optional[float] = None
    boost_dispatch: Optional[BoostDispatchRecord] = None   # B2b-1 authoritative binding

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ValueError("zone_id must be non-empty")
        if not self.ts:
            raise ValueError("ts must be provided")


# -- captured records ---------------------------------------------------------

@dataclass(frozen=True)
class CapturedDecision:
    decision_id: DecisionId
    zone_id: str
    decision_type: DecisionType
    opened_ts: str
    target_c: Optional[float]
    trv_setpoint_c: Optional[float]
    boost_offset_c: Optional[float]
    superseded: bool = False
    closed_ts: Optional[str] = None


@dataclass(frozen=True)
class CapturedSnapshot:
    zone_id: str
    ts: str
    decision_id: DecisionId
    target_c: Optional[float]
    trv_setpoint_c: Optional[float]
    indoor_temp: Optional[Measurement]
    indoor_temp_quality: DataQuality
    decision_type: DecisionType
    controller_demand: Optional[bool]
    tpi_on: Optional[bool]
    boost_active: bool
    preheat_active: bool
    fallback_active: bool
    window_open: Optional[bool]
    heating_failure: bool
    schedule_comfort_time_utc: Optional[str]
    schedule_comfort_temperature_c: Optional[float]
    outdoor_temp: Optional[Measurement]
    valve_opening: Optional[Measurement]
    hvac_action: Optional[str]
    forecast_high: Optional[float]
    accepted_manual_corrections: tuple[ManualCorrectionObservation, ...]
    rejected_echo_corrections: tuple[str, ...]   # event ids excluded as command echoes
    boost_dispatch: Optional[BoostDispatchRecord] = None   # B2b-1 authoritative binding
    device_states: Mapping[str, str] = field(default_factory=dict)   # B2b-4e live availability


# -- command ledger -----------------------------------------------------------

class CommandLedger:
    """Bounded record of ThermoSmart-sent setpoint/mode commands per zone+TRV.

    Lets capture exclude our own writes (and device echoes) from being learned as
    user corrections. Auto-pruned; never an unbounded history.
    """

    def __init__(self, max_entries: int = 200) -> None:
        self._max = max_entries
        self._commands: dict[str, SentCommand] = {}

    def record(self, command: SentCommand) -> None:
        self._commands[command.command_id] = command
        if len(self._commands) > self._max:
            ordered = sorted(self._commands.values(), key=lambda c: (c.sent_ts, c.command_id))
            for victim in ordered[: len(self._commands) - self._max]:
                self._commands.pop(victim.command_id, None)

    def ingest(self, commands: Sequence[SentCommand]) -> None:
        for c in commands:
            self.record(c)

    def is_echo(self, trv_binding_id: str, value: float, ts_iso: str) -> bool:
        ts = _parse(ts_iso)
        for c in self._commands.values():
            if c.command_type is not CommandType.SETPOINT:
                continue
            if c.trv_binding_id != trv_binding_id:
                continue
            if abs(c.value - value) > 1e-6:
                continue
            sent = _parse(c.sent_ts)
            if sent is None or ts is None:
                continue
            delta = (ts - sent).total_seconds()
            if 0 <= delta <= c.echo_window_s:
                return True
        return False

    def prune(self, now_iso: str, max_age_s: float = 3600.0) -> int:
        now = _parse(now_iso)
        if now is None:
            return 0
        removed = 0
        for cid, c in list(self._commands.items()):
            sent = _parse(c.sent_ts)
            if sent is not None and (now - sent).total_seconds() > max_age_s:
                self._commands.pop(cid, None)
                removed += 1
        return removed

    @property
    def size(self) -> int:
        return len(self._commands)

    def serialize(self) -> list:
        return [
            {"command_id": c.command_id, "command_type": c.command_type.value,
             "trv_binding_id": c.trv_binding_id, "value": c.value, "sent_ts": c.sent_ts,
             "echo_window_s": c.echo_window_s}
            for c in sorted(self._commands.values(), key=lambda c: (c.sent_ts, c.command_id))]

    def restore(self, data: Sequence[Mapping]) -> None:
        self._commands = {}
        for d in data:
            self.record(SentCommand(
                command_id=d["command_id"], command_type=CommandType(d["command_type"]),
                trv_binding_id=d["trv_binding_id"], value=d["value"], sent_ts=d["sent_ts"],
                echo_window_s=d.get("echo_window_s", 120.0)))


# -- decision identity / capture coordinator ----------------------------------

def make_decision_id(zone_id: str, decision_type: DecisionType, opened_ts: str,
                     token: Optional[str] = None) -> DecisionId:
    """Deterministic, zone-scoped, name-free decision id."""
    basis = token if token else f"{decision_type.value}:{opened_ts}"
    digest = hashlib.sha256(f"{zone_id}|{basis}".encode("utf-8")).hexdigest()[:16]
    return DecisionId(f"dec_{digest}")


class CaptureCoordinator:
    """Stateful per-zone capture: decision identity continuation + echo filtering.

    Holds only a tiny amount of state (the open decision and a command ledger).
    Pure: no HA, no I/O. A new decision id is created when the controller starts a
    materially different decision; the same id continues otherwise; a superseded
    or ended decision is closed.
    """

    def __init__(self, zone_id: str, *, ledger: Optional[CommandLedger] = None) -> None:
        self._zone = zone_id
        self._ledger = ledger or CommandLedger()
        self._open: Optional[CapturedDecision] = None

    @property
    def ledger(self) -> CommandLedger:
        return self._ledger

    @property
    def open_decision(self) -> Optional[CapturedDecision]:
        return self._open

    def _decision_changed(self, inp: RuntimeCycleInput) -> bool:
        d = inp.controller_decision
        if self._open is None:
            return True
        if d.superseded:
            return True
        if d.decision_type is not self._open.decision_type:
            return True
        if d.decision_token and self._open.decision_id and \
                not self._open.decision_id.endswith(
                    make_decision_id(self._zone, d.decision_type, self._open.opened_ts,
                                     d.decision_token)[-16:]):
            # a new explicit controller token => new decision
            return self._open.decision_id != make_decision_id(
                self._zone, d.decision_type, self._open.opened_ts, d.decision_token)
        # a material target/setpoint jump opens a new decision
        if d.target_c is not None and self._open.target_c is not None \
                and abs(d.target_c - self._open.target_c) > 0.05:
            return True
        return False

    def capture(self, inp: RuntimeCycleInput) -> CapturedSnapshot:
        if inp.zone_id != self._zone:
            raise ValueError("capture input zone mismatch")
        self._ledger.ingest(inp.sent_commands)
        self._ledger.prune(inp.ts)
        d = inp.controller_decision

        if self._decision_changed(inp):
            if self._open is not None:
                self._open = None  # previous decision closed/superseded
            did = make_decision_id(self._zone, d.decision_type, inp.ts, d.decision_token)
            self._open = CapturedDecision(
                decision_id=did, zone_id=self._zone, decision_type=d.decision_type,
                opened_ts=inp.ts, target_c=d.target_c, trv_setpoint_c=d.trv_setpoint_c,
                boost_offset_c=d.boost_offset_c, superseded=False)
        decision_id = self._open.decision_id

        # echo filtering of manual corrections
        accepted = []
        rejected = []
        for mc in inp.manual_corrections:
            binding = mc.trv_binding_id or "_zone"
            if self._ledger.is_echo(binding, mc.new_target_c, mc.ts):
                rejected.append(mc.event_id)
            else:
                accepted.append(mc)

        quality = inp.indoor_temp.quality if inp.indoor_temp is not None else DataQuality.UNAVAILABLE
        return CapturedSnapshot(
            zone_id=self._zone, ts=inp.ts, decision_id=decision_id, target_c=inp.target_c,
            trv_setpoint_c=inp.trv_setpoint_c, indoor_temp=inp.indoor_temp,
            indoor_temp_quality=quality, decision_type=d.decision_type,
            controller_demand=inp.controller_demand, tpi_on=inp.tpi_on,
            boost_active=d.boost_active, preheat_active=d.preheat_active,
            fallback_active=d.fallback_active, window_open=inp.window_open,
            heating_failure=inp.heating_failure,
            schedule_comfort_time_utc=inp.schedule.comfort_time_utc if inp.schedule else None,
            schedule_comfort_temperature_c=inp.schedule.comfort_temperature_c
            if inp.schedule else None,
            outdoor_temp=inp.outdoor_temp, valve_opening=inp.valve_opening,
            hvac_action=inp.hvac_action, forecast_high=inp.forecast_high,
            accepted_manual_corrections=tuple(accepted),
            rejected_echo_corrections=tuple(rejected),
            boost_dispatch=inp.boost_dispatch,
            device_states=dict(inp.device_states or {}))

    def serialize(self) -> dict:
        return {
            "zone_id": self._zone, "ledger": self._ledger.serialize(),
            "open_decision": None if self._open is None else {
                "decision_id": self._open.decision_id, "decision_type": self._open.decision_type.value,
                "opened_ts": self._open.opened_ts, "target_c": self._open.target_c,
                "trv_setpoint_c": self._open.trv_setpoint_c,
                "boost_offset_c": self._open.boost_offset_c},
        }

    def restore(self, data: Mapping) -> None:
        self._ledger.restore(data.get("ledger", []))
        od = data.get("open_decision")
        if od is None:
            self._open = None
        else:
            self._open = CapturedDecision(
                decision_id=DecisionId(od["decision_id"]), zone_id=self._zone,
                decision_type=DecisionType(od["decision_type"]), opened_ts=od["opened_ts"],
                target_c=od.get("target_c"), trv_setpoint_c=od.get("trv_setpoint_c"),
                boost_offset_c=od.get("boost_offset_c"))


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
