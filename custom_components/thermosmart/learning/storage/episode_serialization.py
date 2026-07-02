"""Versioned, defensive serialization for LE 2.0 episode dataclasses.

Mirrors the pattern already established for raw records in
``serialization.py``: one encoder/decoder pair per type, a small dispatch
table, and a public ``serialize_*``/``deserialize_*`` API. Reuses the same
``encode_utc``/``decode_utc``/``encode_measurement``/``decode_measurement``
primitives so timestamps and measurements round-trip identically to raw
records.

Storage-layer shape (``serialize_episode``/``deserialize_episode``):
    A FLAT dict — ``episode_schema_version``, ``episode_type``, and every
    field of the underlying dataclass at the top level (no nested "data"
    wrapper). This is deliberate: it is exactly the entry shape
    ``evaluate_episode_retention()`` / ``async_prune_episodes()``
    (``retention.py`` / ``retention_service.py``) already expect —
    ``entry.get("episode_type")``, ``entry.get("end_ts")`` /
    ``entry.get("start_ts")`` — so this format plugs into the existing
    retention foundation with zero changes there. It also matches the
    ``EpisodesStore`` container convention from ``reset.py``'s
    ``_empty_episodes()`` (``{"episodes": {episode_id: <this dict>}}``).

The storage-layer shape retains ALL fields, including ``learning_zone_id``,
``decision_id``, ``trv_binding_id`` and ``episode_id`` (which itself embeds
the raw zone id, see ``builders/base.py``'s ``f"{zone}:{type}:{seq}"``
convention) — exactly like raw records already do. Those are storage-internal,
never exported directly. ``episode_for_support_export()`` /
``episode_for_research_export()`` build separate, public-safe derived views
that strip all of that out.

No HA imports. No runtime imports. No storage I/O of any kind — this module
only converts between dataclass instances and JSON-safe dicts. Never mutates
the episode objects it serializes. Never raises: malformed/unknown input is
handled defensively (skip or return None).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from ..episode_schemas import (
    AfterheatEpisode,
    ControllerKind,
    EpisodeReason,
    EpisodeType,
    HeatingEpisode,
    OutcomeEpisode,
    PassiveCoolingEpisode,
    Trajectory,
    TrajectoryPoint,
    WindowCoolingEpisode,
)
from ..contracts import DataQuality, Regime
from .serialization import decode_measurement, decode_utc, encode_measurement, encode_utc

EPISODE_SCHEMA_VERSION = 1


class EpisodeSerializationError(Exception):
    """Raised internally only; never propagates past this module's public API."""


# -- trajectory codecs ---------------------------------------------------------

def _enc_trajectory_point(p: TrajectoryPoint) -> dict:
    return {"offset_ms": p.offset_ms, "value": p.value, "quality": p.quality.value, "gap": p.gap}


def _dec_trajectory_point(d: Mapping[str, Any]) -> TrajectoryPoint:
    return TrajectoryPoint(
        offset_ms=d["offset_ms"],
        value=d.get("value"),
        quality=DataQuality(d.get("quality", "ok")),
        gap=bool(d.get("gap", False)),
    )


def _enc_trajectory(traj: Optional[Trajectory]) -> Optional[dict]:
    if traj is None:
        return None
    return {
        "max_points": traj.max_points,
        "points": [_enc_trajectory_point(p) for p in traj.points],
    }


def _dec_trajectory(d: Any) -> Optional[Trajectory]:
    if d is None:
        return None
    if not isinstance(d, Mapping):
        raise EpisodeSerializationError("trajectory payload must be a mapping")
    points = tuple(_dec_trajectory_point(p) for p in (d.get("points") or ()))
    return Trajectory(points=points, max_points=d["max_points"])


# -- per-type encoders ----------------------------------------------------------

def _enc_heating(ep: HeatingEpisode) -> dict:
    return {
        "episode_id": ep.episode_id,
        "learning_zone_id": ep.learning_zone_id,
        "episode_schema_version": ep.episode_schema_version,
        "builder_version": ep.builder_version,
        "classifier_version": ep.classifier_version,
        "start_ts": encode_utc(ep.start_ts),
        "end_ts": encode_utc(ep.end_ts),
        "regime": ep.regime.value,
        "reliability": ep.reliability,
        "start_temp": ep.start_temp,
        "target": ep.target,
        "time_quality": ep.time_quality.value,
        "confounder_flags": list(ep.confounder_flags),
        "trajectory": _enc_trajectory(ep.trajectory),
        "controller": ep.controller.value if ep.controller is not None else None,
        "trv_binding_id": ep.trv_binding_id,
    }


def _dec_heating(d: Mapping[str, Any]) -> HeatingEpisode:
    return HeatingEpisode(
        episode_id=d["episode_id"],
        learning_zone_id=d["learning_zone_id"],
        episode_schema_version=d["episode_schema_version"],
        builder_version=d["builder_version"],
        classifier_version=d["classifier_version"],
        start_ts=decode_utc(d["start_ts"]),
        end_ts=decode_utc(d["end_ts"]),
        regime=Regime(d["regime"]),
        reliability=d["reliability"],
        start_temp=d["start_temp"],
        target=d["target"],
        time_quality=DataQuality(d.get("time_quality", "ok")),
        confounder_flags=tuple(str(f) for f in (d.get("confounder_flags") or ())),
        trajectory=_dec_trajectory(d.get("trajectory")),
        controller=ControllerKind(d["controller"]) if d.get("controller") else None,
        trv_binding_id=d.get("trv_binding_id"),
    )


def _enc_afterheat(ep: AfterheatEpisode) -> dict:
    return {
        "episode_id": ep.episode_id,
        "learning_zone_id": ep.learning_zone_id,
        "episode_schema_version": ep.episode_schema_version,
        "builder_version": ep.builder_version,
        "classifier_version": ep.classifier_version,
        "start_ts": encode_utc(ep.start_ts),
        "end_ts": encode_utc(ep.end_ts),
        "regime": ep.regime.value,
        "reliability": ep.reliability,
        "indoor_temp_at_close": ep.indoor_temp_at_close,
        "target": ep.target,
        "trv_setpoint_before": ep.trv_setpoint_before,
        "trv_setpoint_after": ep.trv_setpoint_after,
        "trajectory": _enc_trajectory(ep.trajectory),
        "time_quality": ep.time_quality.value,
        "confounder_flags": list(ep.confounder_flags),
        "valve_before": encode_measurement(ep.valve_before),
        "valve_after": encode_measurement(ep.valve_after),
        "radiator_profile_id": ep.radiator_profile_id,
        "trv_binding_id": ep.trv_binding_id,
    }


def _dec_afterheat(d: Mapping[str, Any]) -> AfterheatEpisode:
    traj = _dec_trajectory(d.get("trajectory"))
    if traj is None:
        raise EpisodeSerializationError("AfterheatEpisode requires a trajectory")
    return AfterheatEpisode(
        episode_id=d["episode_id"],
        learning_zone_id=d["learning_zone_id"],
        episode_schema_version=d["episode_schema_version"],
        builder_version=d["builder_version"],
        classifier_version=d["classifier_version"],
        start_ts=decode_utc(d["start_ts"]),
        end_ts=decode_utc(d["end_ts"]),
        regime=Regime(d["regime"]),
        reliability=d["reliability"],
        indoor_temp_at_close=d["indoor_temp_at_close"],
        target=d["target"],
        trv_setpoint_before=d["trv_setpoint_before"],
        trv_setpoint_after=d["trv_setpoint_after"],
        trajectory=traj,
        time_quality=DataQuality(d.get("time_quality", "ok")),
        confounder_flags=tuple(str(f) for f in (d.get("confounder_flags") or ())),
        valve_before=decode_measurement(d.get("valve_before")),
        valve_after=decode_measurement(d.get("valve_after")),
        radiator_profile_id=d.get("radiator_profile_id"),
        trv_binding_id=d.get("trv_binding_id"),
    )


def _enc_passive_cooling(ep: PassiveCoolingEpisode) -> dict:
    return {
        "episode_id": ep.episode_id,
        "learning_zone_id": ep.learning_zone_id,
        "episode_schema_version": ep.episode_schema_version,
        "builder_version": ep.builder_version,
        "classifier_version": ep.classifier_version,
        "start_ts": encode_utc(ep.start_ts),
        "end_ts": encode_utc(ep.end_ts),
        "regime": ep.regime.value,
        "reliability": ep.reliability,
        "start_temp": ep.start_temp,
        "end_temp": ep.end_temp,
        "trajectory": _enc_trajectory(ep.trajectory),
        "time_quality": ep.time_quality.value,
        "confounder_flags": list(ep.confounder_flags),
    }


def _dec_passive_cooling(d: Mapping[str, Any]) -> PassiveCoolingEpisode:
    traj = _dec_trajectory(d.get("trajectory"))
    if traj is None:
        raise EpisodeSerializationError("PassiveCoolingEpisode requires a trajectory")
    return PassiveCoolingEpisode(
        episode_id=d["episode_id"],
        learning_zone_id=d["learning_zone_id"],
        episode_schema_version=d["episode_schema_version"],
        builder_version=d["builder_version"],
        classifier_version=d["classifier_version"],
        start_ts=decode_utc(d["start_ts"]),
        end_ts=decode_utc(d["end_ts"]),
        regime=Regime(d["regime"]),
        reliability=d["reliability"],
        start_temp=d["start_temp"],
        end_temp=d["end_temp"],
        trajectory=traj,
        time_quality=DataQuality(d.get("time_quality", "ok")),
        confounder_flags=tuple(str(f) for f in (d.get("confounder_flags") or ())),
    )


def _enc_window_cooling(ep: WindowCoolingEpisode) -> dict:
    return {
        "episode_id": ep.episode_id,
        "learning_zone_id": ep.learning_zone_id,
        "episode_schema_version": ep.episode_schema_version,
        "builder_version": ep.builder_version,
        "classifier_version": ep.classifier_version,
        "start_ts": encode_utc(ep.start_ts),
        "end_ts": encode_utc(ep.end_ts),
        "regime": ep.regime.value,
        "reliability": ep.reliability,
        "temp_at_open": ep.temp_at_open,
        "temp_at_close": ep.temp_at_close,
        "time_quality": ep.time_quality.value,
        "confounder_flags": list(ep.confounder_flags),
        "trajectory": _enc_trajectory(ep.trajectory),
    }


def _dec_window_cooling(d: Mapping[str, Any]) -> WindowCoolingEpisode:
    return WindowCoolingEpisode(
        episode_id=d["episode_id"],
        learning_zone_id=d["learning_zone_id"],
        episode_schema_version=d["episode_schema_version"],
        builder_version=d["builder_version"],
        classifier_version=d["classifier_version"],
        start_ts=decode_utc(d["start_ts"]),
        end_ts=decode_utc(d["end_ts"]),
        regime=Regime(d["regime"]),
        reliability=d["reliability"],
        temp_at_open=d["temp_at_open"],
        temp_at_close=d["temp_at_close"],
        time_quality=DataQuality(d.get("time_quality", "ok")),
        confounder_flags=tuple(str(f) for f in (d.get("confounder_flags") or ())),
        trajectory=_dec_trajectory(d.get("trajectory")),
    )


def _enc_outcome(ep: OutcomeEpisode) -> dict:
    return {
        "episode_id": ep.episode_id,
        "learning_zone_id": ep.learning_zone_id,
        "decision_id": ep.decision_id,
        "episode_schema_version": ep.episode_schema_version,
        "builder_version": ep.builder_version,
        "classifier_version": ep.classifier_version,
        "start_ts": encode_utc(ep.start_ts),
        "end_ts": encode_utc(ep.end_ts),
        "regime": ep.regime.value,
        "reliability": ep.reliability,
        "start_temp": ep.start_temp,
        "end_temp": ep.end_temp,
        "target": ep.target,
        "comfort_tolerance_at_start": ep.comfort_tolerance_at_start,
        "reason": ep.reason.value,
        "controller": ep.controller.value,
        "trajectory": _enc_trajectory(ep.trajectory),
        "time_quality": ep.time_quality.value,
        "confounder_flags": list(ep.confounder_flags),
        "legacy_peak_temp": ep.legacy_peak_temp,
    }


def _dec_outcome(d: Mapping[str, Any]) -> OutcomeEpisode:
    traj = _dec_trajectory(d.get("trajectory"))
    if traj is None:
        raise EpisodeSerializationError("OutcomeEpisode requires a trajectory")
    return OutcomeEpisode(
        episode_id=d["episode_id"],
        learning_zone_id=d["learning_zone_id"],
        decision_id=d["decision_id"],
        episode_schema_version=d["episode_schema_version"],
        builder_version=d["builder_version"],
        classifier_version=d["classifier_version"],
        start_ts=decode_utc(d["start_ts"]),
        end_ts=decode_utc(d["end_ts"]),
        regime=Regime(d["regime"]),
        reliability=d["reliability"],
        start_temp=d["start_temp"],
        end_temp=d["end_temp"],
        target=d["target"],
        comfort_tolerance_at_start=d["comfort_tolerance_at_start"],
        reason=EpisodeReason(d["reason"]),
        controller=ControllerKind(d["controller"]),
        trajectory=traj,
        time_quality=DataQuality(d.get("time_quality", "ok")),
        confounder_flags=tuple(str(f) for f in (d.get("confounder_flags") or ())),
        legacy_peak_temp=d.get("legacy_peak_temp"),
    )


_SCHEMA_FOR_TYPE: dict[EpisodeType, type] = {
    EpisodeType.HEATING: HeatingEpisode,
    EpisodeType.AFTERHEAT: AfterheatEpisode,
    EpisodeType.PASSIVE_COOLING: PassiveCoolingEpisode,
    EpisodeType.WINDOW_COOLING: WindowCoolingEpisode,
    EpisodeType.OUTCOME: OutcomeEpisode,
}

_ENCODERS: dict[EpisodeType, Callable[[Any], dict]] = {
    EpisodeType.HEATING: _enc_heating,
    EpisodeType.AFTERHEAT: _enc_afterheat,
    EpisodeType.PASSIVE_COOLING: _enc_passive_cooling,
    EpisodeType.WINDOW_COOLING: _enc_window_cooling,
    EpisodeType.OUTCOME: _enc_outcome,
}

_DECODERS: dict[EpisodeType, Callable[[Mapping[str, Any]], Any]] = {
    EpisodeType.HEATING: _dec_heating,
    EpisodeType.AFTERHEAT: _dec_afterheat,
    EpisodeType.PASSIVE_COOLING: _dec_passive_cooling,
    EpisodeType.WINDOW_COOLING: _dec_window_cooling,
    EpisodeType.OUTCOME: _dec_outcome,
}


def _episode_type_for(episode: Any) -> Optional[EpisodeType]:
    for etype, schema in _SCHEMA_FOR_TYPE.items():
        if isinstance(episode, schema):
            return etype
    return None


# -- public storage-layer API ---------------------------------------------------

def serialize_episode(episode: Any) -> Optional[dict]:
    """Serialize one episode dataclass to a flat, versioned, JSON-safe dict.

    Returns None for an unrecognised episode type or on any encoding
    failure — never raises. Never mutates ``episode``.
    """
    etype = _episode_type_for(episode)
    if etype is None:
        return None
    encoder = _ENCODERS.get(etype)
    if encoder is None:
        return None
    try:
        data = encoder(episode)
    except Exception:
        return None
    return {
        "episode_schema_version": EPISODE_SCHEMA_VERSION,
        "episode_type": etype.value,
        **data,
    }


def deserialize_episode(raw: Mapping[str, Any]) -> Optional[Any]:
    """Reconstruct one episode dataclass from a serialize_episode() payload.

    Returns None on schema-version mismatch, unknown/missing episode type,
    malformed shape, or a missing required field — never raises.
    """
    if not isinstance(raw, Mapping):
        return None
    if raw.get("episode_schema_version") != EPISODE_SCHEMA_VERSION:
        return None
    try:
        etype = EpisodeType(raw.get("episode_type"))
    except ValueError:
        return None
    decoder = _DECODERS.get(etype)
    if decoder is None:
        return None
    try:
        return decoder(raw)
    except Exception:
        return None


def serialize_episode_list(episodes: Sequence[Any]) -> list[dict]:
    """Serialize a sequence of episodes, skipping any that fail to encode
    (unknown type, encode error). Preserves the order of successful entries.
    Never raises.
    """
    result: list[dict] = []
    if not isinstance(episodes, Sequence):
        return result
    for ep in episodes:
        payload = serialize_episode(ep)
        if payload is not None:
            result.append(payload)
    return result


def deserialize_episode_list(raw_list: Sequence[Mapping[str, Any]]) -> list:
    """Deserialize a sequence of serialize_episode() payloads, skipping any
    malformed/unknown entries. Preserves the order of successful entries.
    Never raises.
    """
    result: list = []
    if not isinstance(raw_list, Sequence):
        return result
    for raw in raw_list:
        episode = deserialize_episode(raw)
        if episode is not None:
            result.append(episode)
    return result


# -- public-safe derived export shapes -------------------------------------------

def _episode_duration_seconds(episode: Any) -> Optional[float]:
    try:
        start = getattr(episode, "start_ts", None)
        end = getattr(episode, "end_ts", None)
        if start is None or end is None:
            return None
        return round((end - start).total_seconds(), 1)
    except Exception:
        return None


def episode_for_support_export(episode: Any) -> Optional[dict]:
    """Return a compact, public-safe support-export shape for one episode.

    No entity ids, no zone names, no decision/trv/episode/radiator ids, no raw
    timestamps (only a derived duration), no trajectory points. Returns None
    for an unrecognised episode type or on any unexpected failure — never
    raises. Only fields that actually exist on the given episode type are
    populated; no fabricated fields for types that lack them.
    """
    etype = _episode_type_for(episode)
    if etype is None:
        return None
    try:
        reason = getattr(episode, "reason", None)
        controller = getattr(episode, "controller", None)
        return {
            "episode_type": etype.value,
            "duration_seconds": _episode_duration_seconds(episode),
            "regime": episode.regime.value,
            "reliability": round(float(episode.reliability), 3),
            "time_quality": episode.time_quality.value,
            "confounder_flags": list(episode.confounder_flags),
            "reason": reason.value if reason is not None else None,
            "reason_is_timeout": (
                (reason is EpisodeReason.TIMEOUT) if reason is not None else None
            ),
            "source": controller.value if controller is not None else None,
        }
    except Exception:
        return None


def episode_for_research_export(episode: Any) -> Optional[dict]:
    """Return a richer, still public-safe research-export shape for one episode.

    Adds episode-level physical metrics on top of the support shape (still no
    ids, no raw timestamps, no trajectory points). Fields that only exist on
    some episode types are included only when present (getattr-guarded) — no
    fabricated fields for episode types that lack them. Returns None for an
    unrecognised episode type or on any unexpected failure — never raises.
    """
    base = episode_for_support_export(episode)
    if base is None:
        return None
    try:
        extra = {
            "start_temp": getattr(episode, "start_temp", None),
            "end_temp": getattr(episode, "end_temp", None),
            "target": getattr(episode, "target", None),
            "indoor_temp_at_close": getattr(episode, "indoor_temp_at_close", None),
            "trv_setpoint_before": getattr(episode, "trv_setpoint_before", None),
            "trv_setpoint_after": getattr(episode, "trv_setpoint_after", None),
            "temp_at_open": getattr(episode, "temp_at_open", None),
            "temp_at_close": getattr(episode, "temp_at_close", None),
            "comfort_tolerance_at_start": getattr(episode, "comfort_tolerance_at_start", None),
            "legacy_peak_temp": getattr(episode, "legacy_peak_temp", None),
        }
        return {**base, **extra}
    except Exception:
        return None
