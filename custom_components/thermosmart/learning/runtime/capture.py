"""Runtime capture for Learning shadow foundation (pure Python).

Materialises the actually-available ThermoSmart signals into typed snapshots and
a stable per-zone decision identity, plus a bounded command ledger used to tell
ThermoSmart's own setpoint writes (and device echoes) apart from real user
corrections. No HA imports, no service calls, no control effect — all inputs are
injected as typed values; timestamps are passed in.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping, Optional, Sequence


def _coerce_finite_offset(value) -> Optional[float]:
    """N2: normalize a persisted offset to a finite float, or None. A malformed value
    (non-numeric string, NaN, inf, etc.) becomes None instead of crashing a later
    numeric comparison. A valid 0.0 or positive/negative finite number is preserved.

    F-D: a bool is NOT a numeric offset — ``float(True)`` would silently become 1.0 and
    read as a real boost magnitude — so a bool is rejected (None), which the checked
    callers treat as a present-but-malformed field."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _coerce_finite_tuple(values) -> tuple:
    """N2: keep only finite numeric entries from a persisted setpoint sequence."""
    out = []
    for v in values or ():
        c = _coerce_finite_offset(v)
        if c is not None:
            out.append(c)
    return tuple(out)


def _coerce_finite_tuple_checked(values) -> tuple:
    """L1: normalize a persisted setpoint sequence AND report whether it was malformed.

    Returns ``(finite_entries, unreliable)``. ``unreliable`` is True when the container
    is not a list/tuple, or when ANY member is non-numeric / None / NaN / inf — i.e.
    whenever a per-device setpoint was present in the payload but could not be trusted.
    Unlike ``_coerce_finite_tuple`` (which SILENTLY drops bad members), this never lets
    a corrupt per-device value vanish quietly: a dropped confounding held/written
    setpoint could otherwise turn a confounded multi-TRV Boost episode into what looks
    like clean, successful adaptive learning. The caller marks the whole record
    unreliable so it can be failed closed instead."""
    if values is None:
        return (), False
    if not isinstance(values, (list, tuple)):
        # A malformed container (string, dict, scalar, …): no trustworthy evidence.
        return (), True
    out = []
    unreliable = False
    for v in values:
        c = _coerce_finite_offset(v)
        if c is None:
            unreliable = True      # present-but-invalid member: do NOT silently drop
        else:
            out.append(c)
    return tuple(out), unreliable


def _coerce_offset_checked(d, key) -> tuple:
    """F-C: coerce a persisted offset field, distinguishing ABSENT (valid historical
    absence) from PRESENT-BUT-MALFORMED (corruption).

    Returns ``(finite_float_or_None, malformed)``:
      * key missing, or present and explicitly ``None`` -> (None, False): a legitimate
        legacy absence, NOT corruption.
      * present, finite number (incl. a valid 0.0) -> (float, False).
      * present, non-numeric / NaN / infinity -> (None, True): corruption, so the caller
        marks the whole record persistence-unreliable (fail closed), rather than silently
        treating the corrupt field as absent.
    """
    if key not in d:
        return None, False
    raw = d[key]
    if raw is None:
        return None, False
    c = _coerce_finite_offset(raw)
    if c is None:
        return None, True
    return c, False


def _coerce_flag_checked(d, key) -> tuple:
    """F-C: validate the explicit exposure flag STRICTLY as ``bool`` or ``None`` — never by
    generic truthiness. Returns ``(bool_or_None, malformed)``:
      * key missing, or present and ``None`` -> (None, False): legacy absence.
      * present and a real ``bool`` -> (that bool, False): authoritative True/False
        (a valid explicit ``False``, e.g. a real per-device min_temp clamp, is preserved).
      * present and anything else (numeric ``0``/``1``, empty string, "true", …)
        -> (None, True): corruption. The flag is NOT trusted (treated as absent) and the
        record is marked persistence-unreliable.
    Note: ``isinstance(x, bool)`` rejects ``0``/``1`` ints (they are not ``bool``)."""
    if key not in d:
        return None, False
    raw = d[key]
    if raw is None:
        return None, False
    if isinstance(raw, bool):
        return raw, False
    return None, True


def _coerce_count_checked(d, key) -> tuple:
    """F-D: parse a persisted non-negative integer count safely, distinguishing absence
    from corruption. Returns ``(int_value, malformed)``:
      * key missing, or present and ``None`` -> (0, False): legitimate legacy absence.
      * present, a real non-negative ``int`` (not ``bool``) -> (that, False): a valid 0 is
        preserved.
      * anything else (string "x", NaN/inf float, any float, bool, negative int)
        -> (0, True): corruption. Never raises — an unchecked ``int(...)`` on "x"/NaN/None
        would raise during restore and drop the whole record, which for an older payload
        lacking the episode boost_seen latch could let a boosted episode close as a clean
        baseline. The record is marked unreliable instead (fail closed)."""
    if key not in d:
        return 0, False
    raw = d[key]
    if raw is None:
        return 0, False
    if isinstance(raw, bool):
        return 0, True
    if isinstance(raw, int):
        return (raw, False) if raw >= 0 else (0, True)
    return 0, True


def _coerce_bool_checked(d, key, default=False) -> tuple:
    """F-D: strict bool field (e.g. outcome_eligible). Returns ``(bool, malformed)``.
    Missing or ``None`` -> (default, False). A real ``bool`` -> (that, False). Any other
    present value -> (default, True): corruption. Never truthiness-coerced — a persisted
    ``"false"`` string must NOT read as True."""
    if key not in d or d[key] is None:
        return default, False
    raw = d[key]
    if isinstance(raw, bool):
        return raw, False
    return default, True


_KNOWN_DISPATCH_STATUSES = frozenset({
    "not_attempted", "already_satisfied", "fully_succeeded",
    "partially_succeeded", "failed"})


def _coerce_status_checked(d, key) -> tuple:
    """F-D: validate persisted dispatch_status. Returns ``(status_str, malformed)``:
      * key missing -> ("not_attempted", False): legitimate legacy absence/default.
      * a known status string -> (it, False): valid, semantics preserved.
      * anything else (unknown string "bogus", ``None``, numeric 5, bool) -> (stringified
        value, True): corruption. An unknown status would otherwise bypass the
        conservative legacy exposure fallback (which only triggers on known applied/held
        statuses), so the record is marked unreliable and fails closed instead."""
    if key not in d:
        return "not_attempted", False
    raw = d[key]
    if isinstance(raw, str) and raw in _KNOWN_DISPATCH_STATUSES:
        return raw, False
    return str(raw), True

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
    # F1: POST-dispatch confirmed effective boost offset (coordinator counterfactual
    # min across written AND held devices) — the authoritative offset adaptive learning
    # must use, distinct from ``boost_applied_c`` (the PRE-dispatch APPROVED offset kept
    # for baseline None/0.0/approved semantics). None on old records -> callers fall
    # back to boost_applied_c.
    effective_offset_c: Optional[float] = None
    # F1: held devices' actual observed setpoints (no entity ids) for multi-TRV ambiguity.
    held_effective_setpoints: tuple = ()
    # N1: authoritative dispatch-supported "any device boosted" exposure flag (True iff
    # at least one successfully-written or held device was above baseline). Independent
    # of effective_offset_c (the conservative min). None on older records -> callers
    # derive it conservatively from the per-device setpoints (see any_device_boosted()).
    any_device_boosted: Optional[bool] = None
    # L1/F-C/F-D: evidence-reliability marker. True when the record's evidence cannot be
    # trusted: a malformed persisted field (bad setpoint container or member, a non-finite
    # /non-numeric offset, an unknown dispatch_status, a corrupt count or exposure flag —
    # see from_dict), OR a non-finite live setpoint/offset glitch (see
    # ha_integration._boost_dispatch_from_live_record). A record flagged unreliable must be
    # FAILED CLOSED for adaptive Boost learning (no boost context) AND for baseline storage
    # (no clean baseline) — corrupt/glitched evidence must never be silently normalized and
    # then read as a clean episode. False on a normal live record; round-trips so a
    # re-serialized restored record stays flagged.
    persistence_unreliable: bool = False

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "boost_candidate_c": self.boost_candidate_c,
            "boost_applied_c": self.boost_applied_c,
            "baseline_setpoint_c": self.baseline_setpoint_c,
            "final_setpoint_c": self.final_setpoint_c,
            "effective_setpoint_min_c": self.effective_setpoint_min_c,
            "effective_setpoint_max_c": self.effective_setpoint_max_c,
            "dispatch_status": self.dispatch_status,
            "outcome_eligible": self.outcome_eligible,
            "outcome_reliability": self.outcome_reliability,
            "start_deficit_c": self.start_deficit_c,
            "expected_non_boost_duration_s": self.expected_non_boost_duration_s,
            "boost_evaluation_status": self.boost_evaluation_status,
            "device_control_type": self.device_control_type,
            "effective_setpoints": list(self.effective_setpoints),
            "targets_total": self.targets_total,
            "targets_failed": self.targets_failed,
            "effective_offset_c": self.effective_offset_c,
            "held_effective_setpoints": list(self.held_effective_setpoints),
            "any_device_boosted": self.any_device_boosted,
            "persistence_unreliable": self.persistence_unreliable,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "BoostDispatchRecord":
        # L1/F-C: coerce per-device setpoint tuples AND every scalar offset/flag field
        # with malformed-member detection. A PRESENT-BUT-MALFORMED field (not a legacy
        # absence) marks the whole record persistence-unreliable so it fails closed for
        # both adaptive learning and baseline storage, rather than silently normalizing
        # corruption into a value that could read as a clean episode.
        _eff_sp, _eff_unreliable = _coerce_finite_tuple_checked(d.get("effective_setpoints", []))
        _held_sp, _held_unreliable = _coerce_finite_tuple_checked(
            d.get("held_effective_setpoints", []))
        _malformed = [_eff_unreliable, _held_unreliable]

        def _off(key):
            val, bad = _coerce_offset_checked(d, key)
            _malformed.append(bad)
            return val

        _flag, _flag_bad = _coerce_flag_checked(d, "any_device_boosted")
        _malformed.append(_flag_bad)
        # F-D: safe count parsing, strict bool for outcome_eligible, and dispatch_status
        # validated against the known set. Each distinguishes legacy absence from a
        # present-but-malformed value and never raises during restore.
        _tt, _tt_bad = _coerce_count_checked(d, "targets_total")
        _tf, _tf_bad = _coerce_count_checked(d, "targets_failed")
        _oe, _oe_bad = _coerce_bool_checked(d, "outcome_eligible", default=False)
        _status, _status_bad = _coerce_status_checked(d, "dispatch_status")
        _malformed.extend((_tt_bad, _tf_bad, _oe_bad, _status_bad))
        return cls(
            decision_id=str(d["decision_id"]),
            boost_candidate_c=_off("boost_candidate_c"),
            boost_applied_c=_off("boost_applied_c"),
            baseline_setpoint_c=_off("baseline_setpoint_c"),
            final_setpoint_c=_off("final_setpoint_c"),
            effective_setpoint_min_c=_off("effective_setpoint_min_c"),
            effective_setpoint_max_c=_off("effective_setpoint_max_c"),
            # F-D: an unknown/malformed dispatch_status is kept (stringified) but flags the
            # record unreliable so it cannot bypass the conservative exposure fallback.
            dispatch_status=_status,
            # F-D: "false"/other non-bool must not be truthiness-coerced to True.
            outcome_eligible=_oe,
            outcome_reliability=str(d.get("outcome_reliability", "none")),
            start_deficit_c=_off("start_deficit_c"),
            expected_non_boost_duration_s=_off("expected_non_boost_duration_s"),
            boost_evaluation_status=str(d.get("boost_evaluation_status", "unknown")),
            device_control_type=str(d.get("device_control_type", "setpoint")),
            # L1: checked coercion — a malformed container or an invalid member marks the
            # record unreliable instead of silently dropping (possibly confounding)
            # per-device evidence.
            effective_setpoints=_eff_sp,
            # F-D: safe non-negative count parsing (no unchecked int() that could raise and
            # drop the record on restore).
            targets_total=_tt,
            targets_failed=_tf,
            # N2/F-C: a malformed persisted offset (e.g. "x", NaN, inf) normalizes to None
            # AND flags the record; a valid 0.0 and an explicit null (legacy absence) do not.
            effective_offset_c=_off("effective_offset_c"),
            held_effective_setpoints=_held_sp,
            # F-C: strict bool/None validation — a non-bool (0, 1, "", "true") is corruption,
            # not an authoritative flag, so it is dropped to None and flagged.
            any_device_boosted=_flag,
            persistence_unreliable=(bool(d.get("persistence_unreliable", False))
                                    or any(_malformed)),
        )


_EXPOSURE_EPS_C = 0.05


def record_any_device_boosted(bd) -> bool:
    """N1: the SINGLE authoritative "was this cycle boost-exposed" predicate, used by
    every consumer (pipeline provenance, dispatch-record retention, close-time baseline
    exclusion) so there is exactly one definition.

    True iff at least one SUCCESSFULLY-WRITTEN or currently-HELD device had a positive
    per-device effective offset above the zone baseline. This is INDEPENDENT of
    ``effective_offset_c`` (the conservative MINIMUM used for adaptive learning
    magnitude, which is 0.0 when any device sits at baseline). Failed/not-attempted
    writes never appear in ``effective_setpoints``, so they are not counted.

    Current-format records (M1): the explicit coordinator-computed
    ``any_device_boosted`` flag is authoritative and used alone.

    Legacy records (M2) that predate the flag (``any_device_boosted is None``):
      1. First derive exposure from TRUSTWORTHY per-device setpoint evidence — the
         persisted written + held setpoints vs the zone baseline — when that evidence is
         present (a finite baseline and at least one finite setpoint). This is
         conclusive: it never invents physical confirmation, only reads dispatch-
         supported setpoints that were actually recorded.
      2. When that per-device evidence is ABSENT or inconclusive (no baseline, or no
         recorded setpoints — e.g. an old in-flight Boost payload that stored neither
         held setpoints nor the flag), fall back to a CONSERVATIVE legacy
         baseline-exclusion rule: treat the cycle as exposed iff the dispatch was
         applied/held (status fully_succeeded / partially_succeeded / already_satisfied)
         AND a finite positive approved OR effective Boost offset exceeds the exposure
         threshold. This protects a previously-boosted episode from being misclassified
         as a clean baseline, WITHOUT fabricating a successful write and WITHOUT ever
         being used as adaptive-learning magnitude (that still comes solely from
         ``effective_offset_c`` in the EvidenceMaterializer). Failed / not-attempted
         legacy records with no positive device evidence stay non-exposed.
    """
    if bd is None:
        return False
    v = getattr(bd, "any_device_boosted", None)
    if v is not None:
        # Current-format record: the explicit flag is authoritative and used ALONE. A
        # valid explicit False (e.g. a real per-device min_temp clamp) is never overridden
        # by a zone-baseline comparison.
        return bool(v)
    # ── legacy record (flag absent): COMBINE per-device evidence OR the conservative
    # fallback. F-A: these are not mutually exclusive. An older payload may persist a
    # WRITTEN device at the zone baseline (per-device offset 0.0) while the actually
    # boosted HELD device was never persisted — so the per-device check alone reads
    # "not exposed" and, if it short-circuited, would skip the fallback and store a
    # contaminated baseline. Either signal indicating exposure is sufficient.
    base = getattr(bd, "baseline_setpoint_c", None)
    sps = list(getattr(bd, "effective_setpoints", ()) or ()) \
        + list(getattr(bd, "held_effective_setpoints", ()) or ())
    if base is not None and sps:
        try:
            if any((float(sp) - float(base)) > _EXPOSURE_EPS_C for sp in sps):
                return True             # (1) trustworthy per-device evidence of a boost
        except (TypeError, ValueError):
            pass
    # (2) conservative legacy fallback: an applied/held dispatch with a finite positive
    # approved OR effective offset above the threshold. Never fabricates a successful
    # write; the approved offset is NEVER used as adaptive-learning magnitude (that stays
    # effective_offset_c in the EvidenceMaterializer). Failed / not-attempted legacy
    # records with no positive device evidence stay non-exposed.
    status = getattr(bd, "dispatch_status", "not_attempted")
    if status in ("fully_succeeded", "partially_succeeded", "already_satisfied"):
        for off in (getattr(bd, "effective_offset_c", None),
                    getattr(bd, "boost_applied_c", None)):
            try:
                if off is not None and math.isfinite(float(off)) \
                        and float(off) > _EXPOSURE_EPS_C:
                    return True
            except (TypeError, ValueError):
                continue
    return False


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
    legacy_preheat_start_utc: Optional[str] = None
    legacy_boost_offset_c: Optional[float] = None
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

    # H1: NORMAL and BOOST are the same heating-decision CLASS. A Boost raises the
    # setpoint of an ongoing heat-to-comfort decision — it is a modifier of that
    # decision, not a new decision. Before the D4 fix, decision_type never flipped
    # NORMAL<->BOOST mid-cycle (the boost signal was dead), so a flip never split a
    # decision. D4 made the flip reachable; treating the two as the same class keeps
    # one stable decision_id across the flip, so the boost update context and the
    # outcome episode bind under the SAME id (no lost boost outcome, no orphaned
    # context, and an episode that ever boosted can never be stored as a clean
    # baseline). PREHEAT/IDLE remain distinct regimes and still open a new decision.
    # Per-cycle decision_type/boost_active are unchanged (still BOOST on boost
    # cycles) — only the decision_id/episode boundary is kept stable.
    _HEATING_CLASS = frozenset({DecisionType.NORMAL, DecisionType.BOOST})

    def _same_decision_class(self, a: "DecisionType", b: "DecisionType") -> bool:
        return a is b or (a in self._HEATING_CLASS and b in self._HEATING_CLASS)

    def _decision_changed(self, inp: RuntimeCycleInput) -> bool:
        d = inp.controller_decision
        if self._open is None:
            return True
        if d.superseded:
            return True
        if not self._same_decision_class(d.decision_type, self._open.decision_type):
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
