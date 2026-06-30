"""Cross-cycle learning pipeline for LE 2.0 (pure Python).

Drives the existing Phase-4 FeatureExtractor, Phase-5 ThermalRegimeClassifier and
Phase-6 episode builders from a stream of CapturedSnapshots, producing completed
authoritative episodes and the matching model. No second episode logic lives
here — it only wires the existing components. Pure: no HA, no I/O, no control.
The full path is:

    CapturedSnapshot -> rolling trajectory -> FeatureExtractor
    -> ThermalRegimeClassifier -> stateful builders -> completed episodes
    -> (model_name, episode, optional context) for the orchestrator to apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..builders import (
    AfterheatEpisodeBuilder,
    BuilderAction,
    BuilderInput,
    DecisionContext,
    HeatingEpisodeBuilder,
    OutcomeEpisodeBuilder,
    PassiveCoolingEpisodeBuilder,
)
from ..episode_schemas import ControllerKind
from ..features import FeatureExtractor
from ..episode_schemas import Trajectory, TrajectoryPoint
from ..regime import RegimeInput, RegimeResult, ThermalRegimeClassifier
from .capture import CapturedSnapshot, DecisionType


@dataclass(frozen=True)
class CompletedEpisode:
    model_name: str
    episode: Any
    decision_id: Optional[str] = None


@dataclass(frozen=True)
class PipelineCycleResult:
    regime: Optional[str]
    previous_regime: Optional[str]
    completed: tuple[CompletedEpisode, ...]
    feature_valid_points: int
    deduplicated: bool
    disturbed: bool


_BUILDER_MODEL = (
    ("heat_rate", "heating"),
    ("onset_delay", "heating"),   # same builder, separate model: no learning overlap
    ("heat_loss", "cooling"),
    ("afterheat", "afterheat"),
    ("outcome", "outcome"),
)


class RuntimePipeline:
    """Per-zone stateful episode pipeline (one instance per learning zone)."""

    def __init__(self, zone_id: str, *, extractor: Optional[FeatureExtractor] = None,
                 classifier: Optional[ThermalRegimeClassifier] = None,
                 max_trajectory_points: int = 120) -> None:
        self._zone = zone_id
        self._extractor = extractor or FeatureExtractor()
        self._classifier = classifier or ThermalRegimeClassifier()
        self._builders = {
            "heating": HeatingEpisodeBuilder(zone_id),
            "cooling": PassiveCoolingEpisodeBuilder(zone_id),
            "afterheat": AfterheatEpisodeBuilder(zone_id),
            "outcome": OutcomeEpisodeBuilder(zone_id),
        }
        self._cap = max_trajectory_points
        self._traj: list[tuple[int, Optional[float]]] = []
        self._t0_iso: Optional[str] = None
        self._processed_ts: list[str] = []
        self._open_decision_id: Optional[str] = None
        self._previous_regime: Optional[str] = None
        self._prev_demand_raw: Optional[bool] = None   # raw controller_demand (not fallback proxy)
        self._prev_hvac: Optional[str] = None          # fallback when demand data is absent
        self._prev_trv_setpoint: Optional[float] = None
        self._last_heating_setpoint: Optional[float] = None
        self._drive_end_ts: Optional[datetime] = None

    # -- helpers --------------------------------------------------------

    def _offset_ms(self, ts_iso: str) -> int:
        if self._t0_iso is None:
            self._t0_iso = ts_iso
        t0 = datetime.fromisoformat(self._t0_iso)
        return max(0, int((datetime.fromisoformat(ts_iso) - t0).total_seconds() * 1000))

    def _trajectory(self) -> Trajectory:
        pts = []
        last = -1
        for off, val in self._traj:
            o = off if off > last else last + 1
            last = o
            if val is None:
                from ..contracts import DataQuality
                pts.append(TrajectoryPoint(o, None, DataQuality.UNAVAILABLE, gap=True))
            else:
                pts.append(TrajectoryPoint(o, float(val)))
        return Trajectory(points=tuple(pts), max_points=max(self._cap, len(pts) or 1))

    # -- main -----------------------------------------------------------

    def process(self, snapshot: CapturedSnapshot) -> PipelineCycleResult:
        if snapshot.ts in self._processed_ts:
            return PipelineCycleResult(
                regime=self._previous_regime, previous_regime=self._previous_regime,
                completed=(), feature_valid_points=0, deduplicated=True, disturbed=False)
        self._processed_ts.append(snapshot.ts)
        if len(self._processed_ts) > self._cap * 2:
            self._processed_ts = self._processed_ts[-self._cap:]

        temp = snapshot.indoor_temp.value if snapshot.indoor_temp is not None else None
        self._traj.append((self._offset_ms(snapshot.ts), temp))
        if len(self._traj) > self._cap:
            self._traj = self._traj[-self._cap:]

        from ..features import FeatureName
        features = self._extractor.extract_trajectory_features(self._trajectory())
        vp = features.get(FeatureName.TRAJ_VALID_POINTS)
        valid = int(vp.value or 0) if vp.is_present else 0

        demand = snapshot.controller_demand
        if demand is None and temp is not None and snapshot.trv_setpoint_c is not None:
            demand = snapshot.trv_setpoint_c > temp + 0.3   # honest fallback proxy

        # Drive-end detection: primary via TPI controller_demand True→False (fires the
        # step the boost ends — coordinator sets effective_target to comfort, TPI duty
        # goes to 0, demand=False — while hvac_action may still read "heating" for one
        # more step due to TRV state lag).  Fallback to hvac_action when demand absent.
        raw_demand = snapshot.controller_demand   # before the fallback proxy
        curr_hvac = snapshot.hvac_action
        demand_fell = (self._prev_demand_raw is True and raw_demand is False)
        hvac_fell = (raw_demand is None
                     and self._prev_hvac == "heating"
                     and curr_hvac is not None and curr_hvac != "heating")
        if demand_fell or hvac_fell:
            self._drive_end_ts = datetime.fromisoformat(snapshot.ts)
            # Capture the setpoint from the step BEFORE demand fell; that step still
            # had the boost setpoint (e.g. 24.4 °C) so setpoint_before > setpoint_after
            # enables drive-end inference in _drive_end_reliability().
            self._last_heating_setpoint = self._prev_trv_setpoint
        if raw_demand is not None:
            self._prev_demand_raw = raw_demand
        self._prev_hvac = curr_hvac
        self._prev_trv_setpoint = snapshot.trv_setpoint_c

        seconds_since: Optional[float] = None
        if self._drive_end_ts is not None:
            try:
                seconds_since = (datetime.fromisoformat(snapshot.ts) - self._drive_end_ts).total_seconds()
            except Exception:
                seconds_since = None

        regime = self._classifier.classify(RegimeInput(
            indoor_temp=temp, target=snapshot.target_c, trv_setpoint=snapshot.trv_setpoint_c,
            controller_demand=demand, tpi_on=snapshot.tpi_on,
            window_open=snapshot.window_open, heating_failure=snapshot.heating_failure,
            outdoor_temp=snapshot.outdoor_temp, trajectory_features=features,
            valve_opening=snapshot.valve_opening, hvac_action=snapshot.hvac_action,
            seconds_since_drive_end=seconds_since))

        # decision context for the outcome builder (open on a new decision)
        decision_start = None
        if snapshot.decision_id != self._open_decision_id:
            self._open_decision_id = snapshot.decision_id
            if snapshot.target_c is not None:
                decision_start = DecisionContext(
                    decision_id=snapshot.decision_id, target=snapshot.target_c,
                    comfort_tolerance_at_start=0.5, controller=ControllerKind.THERMOSMART,
                    start_temp=temp)

        # Provide the pre-cutoff boost setpoint within the afterheat window so the
        # AfterheatBuilder captures setpoint_before > setpoint_after for drive-end inference.
        _drive_end_sp = (self._last_heating_setpoint
                         if seconds_since is not None and seconds_since <= 1800 else None)

        binp = BuilderInput(
            ts=datetime.fromisoformat(snapshot.ts), regime=regime, indoor_temp=temp,
            target=snapshot.target_c, trv_setpoint=snapshot.trv_setpoint_c,
            window_open=snapshot.window_open, heating_failure=snapshot.heating_failure,
            outdoor_temp=snapshot.outdoor_temp,
            indoor_temp_quality=snapshot.indoor_temp_quality, decision_start=decision_start,
            drive_end_setpoint_c=_drive_end_sp)

        # Call each builder exactly once per step, then fan out the result to every
        # model that shares the builder key.  Without this, two models that share a
        # builder (e.g. heat_rate and onset_delay both use "heating") would cause the
        # second model's call to hit the per-step dedup guard and receive NO_CHANGE
        # instead of the closed episode — silently dropping updates.
        builder_results: dict[str, "BuilderResult"] = {}
        for _bkey in dict.fromkeys(bk for _, bk in _BUILDER_MODEL):
            try:
                builder_results[_bkey] = self._builders[_bkey].process(binp)
            except Exception:
                pass

        completed: list[CompletedEpisode] = []
        for model_name, builder_key in _BUILDER_MODEL:
            res = builder_results.get(builder_key)
            if res is None:
                continue
            if res.action is BuilderAction.CLOSED and res.episode is not None:
                did = getattr(res.episode, "decision_id", None)
                completed.append(CompletedEpisode(model_name=model_name, episode=res.episode,
                                                  decision_id=did))

        prev = self._previous_regime
        self._previous_regime = regime.regime.value
        from ..contracts import Regime as _R
        return PipelineCycleResult(
            regime=regime.regime.value, previous_regime=prev, completed=tuple(completed),
            feature_valid_points=valid, deduplicated=False,
            disturbed=regime.regime is _R.DISTURBED)

    # -- state ----------------------------------------------------------

    def serialize(self) -> dict:
        return {
            "zone_id": self._zone, "t0_iso": self._t0_iso,
            "trajectory": [[o, v] for o, v in self._traj],
            "processed_ts": list(self._processed_ts[-self._cap:]),
            "open_decision_id": self._open_decision_id,
            "previous_regime": self._previous_regime,
            "prev_demand_raw": self._prev_demand_raw,
            "prev_hvac": self._prev_hvac,
            "prev_trv_setpoint": self._prev_trv_setpoint,
            "last_heating_setpoint": self._last_heating_setpoint,
            "drive_end_ts": self._drive_end_ts.isoformat() if self._drive_end_ts is not None else None,
            "builders": {k: b.snapshot_state() for k, b in self._builders.items()},
        }

    def restore(self, data: Any) -> tuple[str, ...]:
        errors: list[str] = []
        if not isinstance(data, dict):
            return ("pipeline:invalid",)
        self._t0_iso = data.get("t0_iso")
        self._traj = [(int(o), v) for o, v in data.get("trajectory", [])]
        self._processed_ts = list(data.get("processed_ts", []))
        self._open_decision_id = data.get("open_decision_id")
        self._previous_regime = data.get("previous_regime")
        self._prev_demand_raw = data.get("prev_demand_raw")
        self._prev_hvac = data.get("prev_hvac")
        self._prev_trv_setpoint = data.get("prev_trv_setpoint")
        self._last_heating_setpoint = data.get("last_heating_setpoint")
        _det = data.get("drive_end_ts")
        self._drive_end_ts = datetime.fromisoformat(_det) if _det is not None else None
        for k, b in self._builders.items():
            bs = data.get("builders", {}).get(k)
            if bs is None:
                continue
            try:
                b.restore_state(bs)
            except Exception as err:  # corrupt builder segment -> reinit that builder
                errors.append(f"builder:{k}:{type(err).__name__}")
        return tuple(errors)
