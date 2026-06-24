"""Post-reach boost outcome observer + deferred finalization (B2b-4d, pure Python).

The shared OutcomeEpisodeBuilder stays authoritative for the THERMAL episode close
(REACHED after its stable-target dwell). Only the BOOST evaluation is deferred: at
thermal REACHED a per-zone ``BoostPendingOutcome`` is opened in
``TARGET_REACHED_PENDING_OBSERVATION``; post-reach samples are aggregated (bounded) until
a FIXED deadline, then the boost outcome is finalized EXACTLY ONCE — so a boost/afterheat
overshoot that begins *after* the in-band dwell is still attributed to the same decision
and a premature SUCCESS is impossible.

No control, no HA imports. Restart-safe via compact serialize/restore.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping, Optional

PENDING_SCHEMA_VERSION = 2


def _slot_hash(binding: str) -> str:
    """Stable, anonymised per-device slot key (never an entity id in export)."""
    return hashlib.sha1(str(binding).encode("utf-8")).hexdigest()[:12]

# Post-reach observation window (seconds). Long enough for typical afterheat, hard-bounded.
OBS_WINDOW_DEFAULT_S = 900.0
OBS_WINDOW_MIN_S = 600.0
OBS_WINDOW_MAX_S = 1800.0
# A restart/observation gap beyond this breaks continuity -> BOOST_INTERRUPTED.
OBS_MAX_GAP_S = 1800.0
# Bounded post-reach sample ring.
POST_REACH_SAMPLE_CAP = 80


def observation_window_s(predicted_afterheat_duration_s: Optional[float]) -> float:
    if predicted_afterheat_duration_s is None:
        return OBS_WINDOW_DEFAULT_S
    return max(OBS_WINDOW_MIN_S, min(OBS_WINDOW_MAX_S, float(predicted_afterheat_duration_s)))


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _episode_to_dict(ep: Any) -> dict:
    return {
        "episode_id": ep.episode_id, "learning_zone_id": ep.learning_zone_id,
        "decision_id": ep.decision_id, "start_ts": ep.start_ts.isoformat(),
        "end_ts": ep.end_ts.isoformat(), "regime": ep.regime.value,
        "reliability": ep.reliability, "start_temp": ep.start_temp, "end_temp": ep.end_temp,
        "target": ep.target, "comfort_tolerance_at_start": ep.comfort_tolerance_at_start,
        "reason": ep.reason.value, "controller": ep.controller.value,
        "confounder_flags": list(ep.confounder_flags),
        "trajectory": [(p.offset_ms, p.value, p.gap) for p in ep.trajectory.points],
        "max_points": ep.trajectory.max_points,
    }


def _episode_from_dict(d: Mapping[str, Any]) -> Any:
    from ..contracts import DataQuality, Regime
    from ..episode_schemas import (ControllerKind, EpisodeReason, OutcomeEpisode,
                                   Trajectory, TrajectoryPoint)
    pts = tuple(TrajectoryPoint(int(o), v, gap=bool(g)) for o, v, g in d["trajectory"])
    return OutcomeEpisode(
        episode_id=d["episode_id"], learning_zone_id=d["learning_zone_id"],
        decision_id=d["decision_id"], episode_schema_version=1, builder_version=1,
        classifier_version=1, start_ts=datetime.fromisoformat(d["start_ts"]),
        end_ts=datetime.fromisoformat(d["end_ts"]), regime=Regime(d["regime"]),
        reliability=d["reliability"], start_temp=d["start_temp"], end_temp=d["end_temp"],
        target=d["target"], comfort_tolerance_at_start=d["comfort_tolerance_at_start"],
        reason=EpisodeReason(d["reason"]), controller=ControllerKind(d["controller"]),
        trajectory=Trajectory(points=pts, max_points=int(d["max_points"])),
        confounder_flags=tuple(d.get("confounder_flags", [])))


@dataclass
class BoostPendingOutcome:
    """One active per-zone deferred boost finalization. Mutable; aggregates in place."""
    zone_id: str
    decision_id: str
    episode_id: str
    state: str                                   # BoostOutcomeLifecycleState value
    target: float
    episode_end_ts: str
    observation_start_ts: str
    deadline_ts: str                             # FIXED at open; never recomputed
    last_valid_ts: str
    episode: dict                                # compact closed-episode evidence
    boost_context: dict                          # bctx.to_dict()
    post_reach: list = field(default_factory=list)  # bounded [(offset_ms, value)]
    peak_overshoot_c: float = 0.0
    extra_confounders: list = field(default_factory=list)
    # B2b-4e: per anonymised device slot -> bounded availability aggregate (no event list).
    device_slots: dict = field(default_factory=dict)
    # B2b-4e: persisted finalization intent (crash-atomic exactly-once authority).
    finalized: bool = False
    update_applied: bool = False
    finalization_reason: str = ""
    schema_version: int = PENDING_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id, "decision_id": self.decision_id,
            "episode_id": self.episode_id, "state": self.state, "target": self.target,
            "episode_end_ts": self.episode_end_ts,
            "observation_start_ts": self.observation_start_ts, "deadline_ts": self.deadline_ts,
            "last_valid_ts": self.last_valid_ts, "episode": self.episode,
            "boost_context": self.boost_context, "post_reach": list(self.post_reach),
            "peak_overshoot_c": self.peak_overshoot_c,
            "extra_confounders": list(self.extra_confounders),
            "device_slots": {k: dict(v) for k, v in self.device_slots.items()},
            "finalized": self.finalized, "update_applied": self.update_applied,
            "finalization_reason": self.finalization_reason,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Optional["BoostPendingOutcome"]:
        if int(d.get("schema_version", 0)) != PENDING_SCHEMA_VERSION:
            return None                          # unknown version -> degrade safely (drop)
        try:
            return cls(
                zone_id=d["zone_id"], decision_id=d["decision_id"],
                episode_id=d["episode_id"], state=d["state"], target=d["target"],
                episode_end_ts=d["episode_end_ts"],
                observation_start_ts=d["observation_start_ts"], deadline_ts=d["deadline_ts"],
                last_valid_ts=d["last_valid_ts"], episode=d["episode"],
                boost_context=d["boost_context"],
                post_reach=[tuple(x) for x in d.get("post_reach", [])],
                peak_overshoot_c=d.get("peak_overshoot_c", 0.0),
                extra_confounders=list(d.get("extra_confounders", [])),
                device_slots={k: dict(v) for k, v in d.get("device_slots", {}).items()},
                finalized=bool(d.get("finalized", False)),
                update_applied=bool(d.get("update_applied", False)),
                finalization_reason=d.get("finalization_reason", ""))
        except (KeyError, TypeError, ValueError):
            return None                          # corrupt entry isolated

    # ── observation ────────────────────────────────────────────────────────
    def ingest(self, ts: str, temp: Optional[float], *, overshoot_threshold_c: float = 0.5,
               valid: bool = True) -> None:
        if not valid or temp is None:
            return
        ep_start = _parse(self.episode["start_ts"])
        t = _parse(ts)
        if ep_start is None or t is None:
            return
        offset_ms = int((t - ep_start).total_seconds() * 1000)
        self.post_reach.append((offset_ms, float(temp)))
        if len(self.post_reach) > POST_REACH_SAMPLE_CAP:
            self.post_reach = self.post_reach[-POST_REACH_SAMPLE_CAP:]
        over = temp - (self.target + overshoot_threshold_c)
        if over > 0:
            self.peak_overshoot_c = max(self.peak_overshoot_c, temp - self.target)
        self.last_valid_ts = ts

    def add_confounder(self, flag: str) -> None:
        if flag not in self.extra_confounders:
            self.extra_confounders.append(flag)

    # ── B2b-4e: cross-cycle device availability (stable anonymous slots) ─────
    def seed_devices(self, ts: str, device_states: Mapping[str, str], control_type: str,
                     effective_offsets: Optional[Mapping[str, float]] = None) -> None:
        """Establish the dispatched device slot set at observation start."""
        eo = effective_offsets or {}
        for binding, raw in (device_states or {}).items():
            slot = _slot_hash(binding)
            st = str(raw)
            self.device_slots[slot] = {
                "available_at_dispatch": st == "available",
                "ever_unavailable": st == "unavailable", "ever_unknown": st == "unknown",
                "last_known_availability": st, "first_unavailable_ts":
                    ts if st == "unavailable" else None, "last_observed_ts": ts,
                "control_type": control_type, "effective_offset_c": eo.get(binding),
                "removed": False}

    def ingest_devices(self, ts: str, device_states: Mapping[str, str], control_type: str,
                       *, continuity_gap: bool = False) -> None:
        """Per-tick real device-state ingest. A continuity gap (restart/reload) marks
        'unknown' as continuity, never an observed device failure."""
        seen = set()
        for binding, raw in (device_states or {}).items():
            slot = _slot_hash(binding)
            seen.add(slot)
            st = str(raw)
            agg = self.device_slots.get(slot)
            if agg is None:
                # a device appearing only after dispatch is a configuration change
                agg = {"available_at_dispatch": False, "ever_unavailable": False,
                       "ever_unknown": False, "last_known_availability": st,
                       "first_unavailable_ts": None, "last_observed_ts": ts,
                       "control_type": control_type, "effective_offset_c": None,
                       "removed": False, "appeared_after_dispatch": True}
                self.device_slots[slot] = agg
            if st == "unavailable" and not continuity_gap:
                agg["ever_unavailable"] = True
                agg["first_unavailable_ts"] = agg["first_unavailable_ts"] or ts
            elif st == "unknown":
                agg["ever_unknown"] = True       # continuity, not a real failure
            if agg.get("control_type") not in (None, control_type):
                agg["control_type_changed"] = True
            agg["last_known_availability"] = st
            agg["last_observed_ts"] = ts
        # expected (dispatched) slots that vanished from the device set -> config change
        for slot, agg in self.device_slots.items():
            if agg.get("available_at_dispatch") and slot not in seen and not continuity_gap:
                agg["removed"] = True

    def device_confounders(self) -> list:
        flags = []
        for agg in self.device_slots.values():
            if agg.get("removed") or agg.get("control_type_changed") \
                    or agg.get("appeared_after_dispatch"):
                flags.append("device_configuration_change")
            if agg.get("ever_unavailable"):
                flags.append("partial_device_effect")
        return sorted(set(flags))

    def availability_summary(self) -> dict:
        slots = list(self.device_slots.values())
        return {
            "slots_observed": len(slots),
            "ever_unavailable": sum(1 for a in slots if a.get("ever_unavailable")),
            "ever_unknown": sum(1 for a in slots if a.get("ever_unknown")),
            "device_configuration_change": any(
                a.get("removed") or a.get("control_type_changed")
                or a.get("appeared_after_dispatch") for a in slots),
        }

    def is_due(self, now_ts: str) -> bool:
        dl, now = _parse(self.deadline_ts), _parse(now_ts)
        return dl is not None and now is not None and now >= dl

    def gap_s(self, now_ts: str) -> Optional[float]:
        last, now = _parse(self.last_valid_ts), _parse(now_ts)
        if last is None or now is None:
            return None
        return max(0.0, (now - last).total_seconds())

    def build_final_episode(self):
        """Rebuild the OutcomeEpisode with post-reach samples appended (bounded by cap) and
        any runtime/interruption confounders merged."""
        ep = _episode_from_dict(self.episode)
        all_cf = set(self.extra_confounders) | set(self.device_confounders())
        if self.post_reach or all_cf:
            from ..episode_schemas import Trajectory, TrajectoryPoint
            existing = list(ep.trajectory.points)
            last_off = existing[-1].offset_ms if existing else 0
            extra_pts = [TrajectoryPoint(int(o), float(v)) for o, v in self.post_reach
                         if int(o) > last_off]
            cap = ep.trajectory.max_points
            merged = (existing + extra_pts)[-cap:]
            traj = Trajectory(points=tuple(merged), max_points=cap)
            cf = tuple(sorted(set(ep.confounder_flags) | all_cf))
            ep = replace(ep, trajectory=traj, confounder_flags=cf)
        return ep

    @property
    def boost_decision_id(self) -> str:
        return self.decision_id
