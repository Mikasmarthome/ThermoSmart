"""Shared episode-builder contracts and machinery (pure Python)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..contracts import Measurement, Regime, require_utc
from ..episode_schemas import (
    ControllerKind,
    DataQuality,
    EpisodeType,
    Trajectory,
    TrajectoryPoint,
)
from ..raw_schemas import HeatingDriveEndEvent
from ..regime import RegimeResult

BUILDER_VERSION = 1


# -- enums --------------------------------------------------------------------

class BuilderAction(Enum):
    NO_CHANGE = "no_change"
    OPENED = "opened"
    CONTINUED = "continued"
    CLOSED = "closed"        # produces a valid episode
    ABORTED = "aborted"      # open episode killed by a disturbance
    DISCARDED = "discarded"  # ended but not valid enough -> no episode


class ConfounderCode(Enum):
    WINDOW_OPEN = "window_open"
    SOLAR_GAIN = "solar_gain"
    HEATING_FAILURE = "heating_failure"
    SENSOR_GAP = "sensor_gap"
    IMPLAUSIBLE_JUMP = "implausible_jump"
    REHEATING = "reheating"
    MANUAL_CORRECTION = "manual_correction"
    MODE_CHANGE = "mode_change"
    MISSING_OPTIONAL_EVIDENCE = "missing_optional_evidence"


class ReasonCode(Enum):
    OPENED = "opened"
    CONTINUED = "continued"
    CLOSED_DRIVE_END = "closed_drive_end"
    CLOSED_TRANSITION = "closed_transition"
    CLOSED_CONTROLLER_END = "closed_controller_end"
    CLOSED_PEAK_COOLING = "closed_peak_cooling"
    CLOSED_MAX_DURATION = "closed_max_duration"
    CLOSED_WINDOW = "closed_window"
    ABORTED_DISTURBED = "aborted_disturbed"
    ABORTED_WINDOW = "aborted_window"
    ABORTED_HEATING_FAILURE = "aborted_heating_failure"
    ABORTED_REHEATING = "aborted_reheating"
    ABORTED_NO_CLOSE = "aborted_no_close"
    DISCARDED_TOO_SHORT = "discarded_too_short"
    DISCARDED_INSUFFICIENT = "discarded_insufficient_points"
    DISCARDED_HIGH_GAP = "discarded_high_gap"
    UNKNOWN_PAUSE = "unknown_pause"
    UNKNOWN_TIMEOUT = "unknown_timeout"
    DUPLICATE_EVENT = "duplicate_event"
    LATE_EVENT = "late_event"
    RELATIVE_COOLING_ONLY = "relative_cooling_only"
    NO_INDOOR_TEMP = "no_indoor_temp"
    TRAJECTORY_CAPPED = "trajectory_capped"


class OutcomeReason(Enum):
    REACHED = "reached"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"
    MANUAL_CORRECTION = "manual_correction"
    SUPERSEDED = "superseded"
    MODE_CHANGE = "mode_change"
    DISTURBED = "disturbed"
    INSUFFICIENT_DATA = "insufficient_data"


class DownsamplePolicy(Enum):
    STOP = "stop"          # stop accepting points once the cap is hit
    UNIFORM = "uniform"    # drop every other older point deterministically


# -- parameters ---------------------------------------------------------------

@dataclass(frozen=True)
class BuilderParameters:
    """Versioned builder thresholds (provisional; calibrate after a season)."""

    min_open_reliability: float = 0.4
    min_duration_s: float = 180.0
    min_points: int = 2
    max_episode_duration_s: float = 5400.0
    max_unknown_duration_s: float = 600.0
    max_gap_ratio: float = 0.5
    heating_trajectory_cap: int = 120
    afterheat_trajectory_cap: int = 120
    cooling_trajectory_cap: int = 240
    window_trajectory_cap: int = 120
    outcome_trajectory_cap: int = 240
    drive_end_dedup_window_s: float = 120.0
    outcome_stability_window_s: float = 600.0
    outcome_timeout_s: float = 5400.0
    downsample_policy: DownsamplePolicy = DownsamplePolicy.STOP
    parameter_version: int = 1


# -- inputs -------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionContext:
    decision_id: str
    target: float
    comfort_tolerance_at_start: float
    controller: ControllerKind
    expected_minutes: Optional[int] = None
    start_temp: Optional[float] = None


@dataclass(frozen=True)
class BuilderInput:
    ts: datetime
    regime: Optional[RegimeResult] = None
    indoor_temp: Optional[float] = None
    target: Optional[float] = None
    trv_setpoint: Optional[float] = None
    controller_demand: Optional[bool] = None
    tpi_on: Optional[bool] = None
    valve_opening: Optional[Measurement] = None
    hvac_action: Optional[str] = None
    window_open: Optional[bool] = None
    heating_failure: bool = False
    outdoor_temp: Optional[Measurement] = None
    solar_radiation: Optional[Measurement] = None
    drive_end_event: Optional[HeatingDriveEndEvent] = None
    trv_binding_id: Optional[str] = None
    indoor_temp_quality: DataQuality = DataQuality.OK
    # outcome / decision signals
    decision_start: Optional[DecisionContext] = None
    decision_superseded: bool = False
    manual_correction: bool = False
    mode_change: bool = False
    # drive-end evidence for afterheat builder: setpoint from the last heating step
    # (before coordinator reduced it to comfort).  Lets the builder capture
    # setpoint_before > setpoint_after so _drive_end_reliability() can infer drive end.
    drive_end_setpoint_c: Optional[float] = None
    # dedup
    event_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", require_utc(self.ts, "ts"))


# -- result -------------------------------------------------------------------

@dataclass(frozen=True)
class BuilderResult:
    action: BuilderAction
    episode: Optional[Any] = None
    open_episode_id: Optional[str] = None
    reliability: Optional[float] = None
    reason_codes: tuple[str, ...] = ()
    confounder_flags: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()


class BuilderStateError(Exception):
    """Raised on an invalid / version-mismatched / corrupted builder snapshot."""


# -- trajectory accumulation --------------------------------------------------

@dataclass
class _Point:
    offset_ms: int
    value: Optional[float]
    quality: DataQuality
    gap: bool


class TrajectoryAccumulator:
    """Collects monotonic trajectory points with a deterministic cap policy."""

    def __init__(self, cap: int, policy: DownsamplePolicy) -> None:
        self._cap = cap
        self._policy = policy
        self._points: list[_Point] = []
        self._capped = False

    def add(self, offset_ms: int, value: Optional[float],
            quality: DataQuality, gap: bool) -> None:
        if self._points and offset_ms <= self._points[-1].offset_ms:
            return  # enforce strictly increasing offsets; ignore non-monotonic
        if len(self._points) >= self._cap:
            if self._policy is DownsamplePolicy.STOP:
                self._capped = True
                return
            # UNIFORM: drop every other older point, keep the newest
            self._points = self._points[::2]
            self._capped = True
        self._points.append(_Point(offset_ms, value, quality, gap))

    @property
    def capped(self) -> bool:
        return self._capped

    @property
    def count(self) -> int:
        return len(self._points)

    @property
    def valid_count(self) -> int:
        return sum(1 for p in self._points if not p.gap and p.value is not None)

    def first_valid(self) -> Optional[float]:
        for p in self._points:
            if not p.gap and p.value is not None:
                return p.value
        return None

    def last_valid(self) -> Optional[float]:
        for p in reversed(self._points):
            if not p.gap and p.value is not None:
                return p.value
        return None

    @property
    def gap_ratio(self) -> float:
        if not self._points:
            return 0.0
        return sum(1 for p in self._points if p.gap) / len(self._points)

    def build(self, max_points: int) -> Trajectory:
        pts = tuple(TrajectoryPoint(p.offset_ms, p.value, p.quality, p.gap)
                    for p in self._points)
        return Trajectory(points=pts, max_points=max_points)

    def serialize(self) -> list[dict]:
        return [{"o": p.offset_ms, "v": p.value, "q": p.quality.value, "g": p.gap}
                for p in self._points]

    @classmethod
    def restore(cls, cap: int, policy: DownsamplePolicy, data: list[dict],
                capped: bool) -> "TrajectoryAccumulator":
        acc = cls(cap, policy)
        acc._points = [_Point(d["o"], d["v"], DataQuality(d["q"]), d["g"]) for d in data]
        acc._capped = capped
        return acc


# -- base builder -------------------------------------------------------------

class EpisodeBuilder:
    """Common lifecycle: order/dup guard, sequencing, snapshot/restore."""

    episode_type: EpisodeType = None  # set by subclass
    trajectory_cap: int = 120
    materialized: bool = True

    def __init__(self, learning_zone_id: str,
                 params: Optional[BuilderParameters] = None) -> None:
        if not learning_zone_id:
            raise BuilderStateError("learning_zone_id must be non-empty")
        self._zone = learning_zone_id
        self._params = params or BuilderParameters()
        self._seq = 0
        self._last_ts: Optional[datetime] = None
        self._last_fingerprint: Optional[tuple] = None
        self._seen_event_ids: set[str] = set()
        self._reset_episode()

    # -- subclass hooks --------------------------------------------------

    def _reset_episode(self) -> None:
        self._active = False
        self._episode_id: Optional[str] = None
        self._start_ts: Optional[datetime] = None
        self._start_temp: Optional[float] = None
        self._traj: Optional[TrajectoryAccumulator] = None
        self._reliabilities: list[float] = []
        self._confounders: set[ConfounderCode] = set()
        self._last_regime: Optional[Regime] = None
        self._unknown_since: Optional[datetime] = None
        self._extra: dict[str, Any] = {}

    def _step(self, inp: BuilderInput) -> BuilderResult:  # pragma: no cover
        raise NotImplementedError

    def _fingerprint(self, inp: BuilderInput) -> tuple:
        if inp.event_id is not None:
            return ("event", inp.event_id)
        return ("obs", inp.ts.isoformat(), inp.trv_binding_id, inp.indoor_temp)

    # -- public lifecycle ------------------------------------------------

    def process(self, inp: BuilderInput) -> BuilderResult:
        if inp.event_id is not None and inp.event_id in self._seen_event_ids:
            return BuilderResult(BuilderAction.NO_CHANGE,
                                 reason_codes=(ReasonCode.DUPLICATE_EVENT.value,),
                                 open_episode_id=self._episode_id)
        if self._last_ts is not None and inp.ts < self._last_ts:
            return BuilderResult(BuilderAction.NO_CHANGE,
                                 reason_codes=(ReasonCode.LATE_EVENT.value,),
                                 open_episode_id=self._episode_id)
        fp = self._fingerprint(inp)
        if self._last_ts is not None and inp.ts == self._last_ts and fp == self._last_fingerprint:
            return BuilderResult(BuilderAction.NO_CHANGE,
                                 reason_codes=(ReasonCode.DUPLICATE_EVENT.value,),
                                 open_episode_id=self._episode_id)

        result = self._step(inp)

        self._last_ts = inp.ts
        self._last_fingerprint = fp
        if inp.event_id is not None:
            self._seen_event_ids.add(inp.event_id)
        return result

    def current_state(self) -> dict:
        return {
            "active": self._active,
            "episode_id": self._episode_id,
            "start_ts": self._start_ts.isoformat() if self._start_ts else None,
            "last_regime": self._last_regime.value if self._last_regime else None,
            "valid_points": self._traj.valid_count if self._traj else 0,
            "confounders": sorted(c.value for c in self._confounders),
        }

    # -- snapshot / restore ----------------------------------------------

    def snapshot_state(self) -> dict:
        return {
            "builder_version": BUILDER_VERSION,
            "episode_type": self.episode_type.value,
            "zone": self._zone,
            "seq": self._seq,
            "last_ts": self._last_ts.isoformat() if self._last_ts else None,
            "last_fingerprint": list(self._last_fingerprint) if self._last_fingerprint else None,
            "seen_event_ids": sorted(self._seen_event_ids),
            "active": self._active,
            "episode_id": self._episode_id,
            "start_ts": self._start_ts.isoformat() if self._start_ts else None,
            "start_temp": self._start_temp,
            "trajectory": self._traj.serialize() if self._traj else None,
            "trajectory_capped": self._traj.capped if self._traj else False,
            "reliabilities": list(self._reliabilities),
            "confounders": sorted(c.value for c in self._confounders),
            "last_regime": self._last_regime.value if self._last_regime else None,
            "unknown_since": self._unknown_since.isoformat() if self._unknown_since else None,
            "extra": dict(self._extra),
        }

    def restore_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise BuilderStateError("snapshot must be a mapping")
        if state.get("builder_version") != BUILDER_VERSION:
            raise BuilderStateError(
                f"builder version mismatch: {state.get('builder_version')} != {BUILDER_VERSION}")
        if state.get("episode_type") != self.episode_type.value:
            raise BuilderStateError("episode_type mismatch in snapshot")
        if state.get("zone") != self._zone:
            raise BuilderStateError("zone mismatch in snapshot")
        try:
            self._seq = int(state["seq"])
            self._last_ts = _parse(state["last_ts"])
            self._last_fingerprint = tuple(state["last_fingerprint"]) \
                if state["last_fingerprint"] else None
            self._seen_event_ids = set(state["seen_event_ids"])
            self._active = bool(state["active"])
            self._episode_id = state["episode_id"]
            self._start_ts = _parse(state["start_ts"])
            self._start_temp = state["start_temp"]
            self._reliabilities = list(state["reliabilities"])
            self._confounders = {ConfounderCode(c) for c in state["confounders"]}
            self._last_regime = Regime(state["last_regime"]) if state["last_regime"] else None
            self._unknown_since = _parse(state["unknown_since"])
            self._extra = dict(state["extra"])
            if state["trajectory"] is not None:
                self._traj = TrajectoryAccumulator.restore(
                    self.trajectory_cap, self._params.downsample_policy,
                    state["trajectory"], state["trajectory_capped"])
            else:
                self._traj = None
        except (KeyError, TypeError, ValueError) as err:
            raise BuilderStateError(f"corrupted snapshot: {err}") from err

    def reset(self) -> None:
        self._seq = 0
        self._last_ts = None
        self._last_fingerprint = None
        self._seen_event_ids = set()
        self._reset_episode()

    # -- shared helpers --------------------------------------------------

    def _offset_ms(self, ts: datetime) -> int:
        return int((ts - self._start_ts).total_seconds() * 1000)

    def _open(self, inp: BuilderInput, cap: int) -> None:
        self._seq += 1
        self._active = True
        self._episode_id = f"{self._zone}:{self.episode_type.value}:{self._seq}"
        self._start_ts = inp.ts
        self._start_temp = inp.indoor_temp
        self._traj = TrajectoryAccumulator(cap, self._params.downsample_policy)
        self._reliabilities = []
        self._confounders = set()
        self._unknown_since = None
        self._accumulate(inp)

    def _accumulate(self, inp: BuilderInput) -> None:
        if self._traj is None:
            return
        gap = inp.indoor_temp is None
        self._traj.add(self._offset_ms(inp.ts), inp.indoor_temp,
                       inp.indoor_temp_quality, gap)
        if inp.regime is not None:
            self._reliabilities.append(inp.regime.reliability)
            self._last_regime = inp.regime.regime
        if inp.target is not None:
            self._extra["target"] = inp.target
        if inp.indoor_temp is not None:
            self._extra["last_temp"] = inp.indoor_temp
        self._note_confounders(inp)

    def _note_confounders(self, inp: BuilderInput) -> None:
        if inp.window_open is True:
            self._confounders.add(ConfounderCode.WINDOW_OPEN)
        if inp.heating_failure:
            self._confounders.add(ConfounderCode.HEATING_FAILURE)
        if inp.indoor_temp is None:
            self._confounders.add(ConfounderCode.SENSOR_GAP)
        if inp.valve_opening is None and inp.hvac_action is None:
            self._confounders.add(ConfounderCode.MISSING_OPTIONAL_EVIDENCE)
        if inp.regime is not None:
            from ..regime import ReasonCode as RR
            if RR.SOLAR_CONFOUNDER.value in inp.regime.reason_codes:
                self._confounders.add(ConfounderCode.SOLAR_GAIN)
            if "implausible_jump" in inp.regime.disturbance_reasons:
                self._confounders.add(ConfounderCode.IMPLAUSIBLE_JUMP)

    def _episode_duration_s(self, ts: datetime) -> float:
        return (ts - self._start_ts).total_seconds()

    def _episode_reliability(self, *, abort_discount: float = 1.0) -> float:
        rels = self._reliabilities or [0.0]
        worst = min(rels)
        mean = sum(rels) / len(rels)
        base = 0.5 * worst + 0.5 * mean
        gap_factor = max(0.0, 1.0 - 0.5 * (self._traj.gap_ratio if self._traj else 0.0))
        rel = base * gap_factor * abort_discount
        return max(0.0, min(1.0, rel))

    def _is_disturbance(self, inp: BuilderInput) -> Optional[ReasonCode]:
        if inp.heating_failure:
            return ReasonCode.ABORTED_HEATING_FAILURE
        if inp.window_open is True:
            return ReasonCode.ABORTED_WINDOW
        if inp.regime is not None and inp.regime.regime is Regime.DISTURBED:
            return ReasonCode.ABORTED_DISTURBED
        return None

    def _maybe_abort(self, inp: BuilderInput) -> Optional[BuilderResult]:
        reason = self._is_disturbance(inp)
        if reason is None:
            return None
        self._note_confounders(inp)
        return self._abort(reason)

    def _abort(self, reason: ReasonCode) -> BuilderResult:
        confounders = tuple(sorted(c.value for c in self._confounders))
        self._reset_episode()
        return BuilderResult(BuilderAction.ABORTED, reason_codes=(reason.value,),
                             confounder_flags=confounders)

    def _finalize(self, episode: Any, reasons: tuple[ReasonCode, ...]) -> BuilderResult:
        reliability = self._episode_reliability()
        confounders = tuple(sorted(c.value for c in self._confounders))
        eid = self._episode_id
        capped = self._traj.capped if self._traj else False
        self._reset_episode()
        codes = tuple(r.value for r in reasons)
        if capped:
            codes = codes + (ReasonCode.TRAJECTORY_CAPPED.value,)
        return BuilderResult(BuilderAction.CLOSED, episode=episode, open_episode_id=eid,
                             reliability=reliability, reason_codes=codes,
                             confounder_flags=confounders)

    def _discard(self, reason: ReasonCode) -> BuilderResult:
        confounders = tuple(sorted(c.value for c in self._confounders))
        self._reset_episode()
        return BuilderResult(BuilderAction.DISCARDED, reason_codes=(reason.value,),
                             confounder_flags=confounders)

    def _resolved_start_temp(self) -> Optional[float]:
        if self._start_temp is not None:
            return self._start_temp
        return self._traj.first_valid() if self._traj else None

    def _too_short_or_sparse(self, ts: datetime) -> Optional[ReasonCode]:
        if self._episode_duration_s(ts) < self._params.min_duration_s:
            return ReasonCode.DISCARDED_TOO_SHORT
        if self._traj is None or self._traj.valid_count < self._params.min_points:
            return ReasonCode.DISCARDED_INSUFFICIENT
        if self._traj.gap_ratio > self._params.max_gap_ratio:
            return ReasonCode.DISCARDED_HIGH_GAP
        return None


def _parse(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)
