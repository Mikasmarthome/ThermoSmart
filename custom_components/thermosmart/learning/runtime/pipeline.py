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
        regime = self._classifier.classify(RegimeInput(
            indoor_temp=temp, target=snapshot.target_c, trv_setpoint=snapshot.trv_setpoint_c,
            controller_demand=demand, tpi_on=snapshot.tpi_on,
            window_open=snapshot.window_open, heating_failure=snapshot.heating_failure,
            outdoor_temp=snapshot.outdoor_temp, trajectory_features=features))

        # decision context for the outcome builder (open on a new decision)
        decision_start = None
        if snapshot.decision_id != self._open_decision_id:
            self._open_decision_id = snapshot.decision_id
            if snapshot.target_c is not None:
                decision_start = DecisionContext(
                    decision_id=snapshot.decision_id, target=snapshot.target_c,
                    comfort_tolerance_at_start=0.5, controller=ControllerKind.THERMOSMART,
                    start_temp=temp)

        binp = BuilderInput(
            ts=datetime.fromisoformat(snapshot.ts), regime=regime, indoor_temp=temp,
            target=snapshot.target_c, trv_setpoint=snapshot.trv_setpoint_c,
            window_open=snapshot.window_open, heating_failure=snapshot.heating_failure,
            outdoor_temp=snapshot.outdoor_temp,
            indoor_temp_quality=snapshot.indoor_temp_quality, decision_start=decision_start)

        completed: list[CompletedEpisode] = []
        for model_name, builder_key in _BUILDER_MODEL:
            builder = self._builders[builder_key]
            try:
                res = builder.process(binp)
            except Exception:
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
        for k, b in self._builders.items():
            bs = data.get("builders", {}).get(k)
            if bs is None:
                continue
            try:
                b.restore_state(bs)
            except Exception as err:  # corrupt builder segment -> reinit that builder
                errors.append(f"builder:{k}:{type(err).__name__}")
        return tuple(errors)
