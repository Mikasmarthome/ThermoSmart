"""Small HA-store-facing retention enforcement layer for LE 2.0 raw segments
and episodes.

Thin glue between the pure decisions in ``retention.py`` and the versioned
store wrappers in ``stores.py``. No business logic lives here beyond loading,
delegating to the pure evaluator, and writing back the result. Never raises —
every failure is captured in the returned result dataclasses so a caller can
run this best-effort without risking any control-path disruption.

No control modification of any kind. No HA imports beyond what ``stores.py``
already requires (the injectable ``StoreFactory`` abstraction keeps this
module HA-independent and directly unit-testable with a fake factory).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from ..clock import Clock
from ..raw_schemas import RawTrackName
from ..registry import EpisodeRegistry, RawTrackRegistry, RetentionPolicy
from .retention import evaluate_episode_retention, evaluate_segment_retention
from .segments import SegmentIndex
from .stores import EpisodesStore, RawSegmentIndexStore, RawSegmentStore, StoreFactory


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RawTrackPruneResult:
    """Outcome of pruning one (zone, track)'s raw segments. Never raises."""
    pruned_count: int
    last_error: Optional[str]


@dataclass(frozen=True)
class EpisodePruneResult:
    """Outcome of pruning one zone's EpisodesStore payload. Never raises."""
    pruned_count: int
    last_error: Optional[str]


@dataclass(frozen=True)
class ZoneRetentionResult:
    """Aggregate outcome of pruning one zone's raw segments + episodes."""
    raw_segments_pruned: int
    episodes_pruned: int
    last_error: Optional[str]

    def to_dict(self) -> dict:
        """Public-safe, read-only summary. No control fields, no ids."""
        return {
            "raw_segments_pruned": self.raw_segments_pruned,
            "episodes_pruned": self.episodes_pruned,
            "last_error": self.last_error,
        }


# ── Segment index encode/decode (retention-local; minimal, additive) ────────

def _decode_segment_index(
    raw: Mapping[str, object], learning_zone_id: str, track_name: RawTrackName,
) -> SegmentIndex:
    """Reconstruct a SegmentIndex from the stored RawSegmentIndexStore payload.

    Tolerates the minimal shape written by ``reset.py``'s ``_empty_raw_index()``
    (no ``corrupted_segment_ids``) as well as a fuller future shape.
    """
    return SegmentIndex(
        learning_zone_id=learning_zone_id,
        track_name=track_name,
        track_schema_version=int(raw.get("track_schema_version", 1)),
        next_sequence_number=int(raw.get("next_sequence_number", 0)),
        active_segment_id=raw.get("active_segment_id"),
        sealed_segment_ids=tuple(raw.get("sealed_segment_ids") or ()),
        corrupted_segment_ids=tuple(raw.get("corrupted_segment_ids") or ()),
        last_flush_utc=None,  # not needed for retention decisions
    )


def _encode_segment_index(index: SegmentIndex) -> dict:
    """Serialize a SegmentIndex back to the RawSegmentIndexStore payload shape."""
    return {
        "track_name": index.track_name.value,
        "track_schema_version": index.track_schema_version,
        "active_segment_id": index.active_segment_id,
        "sealed_segment_ids": list(index.sealed_segment_ids),
        "next_sequence_number": index.next_sequence_number,
        "corrupted_segment_ids": list(index.corrupted_segment_ids),
    }


# ── Raw segment retention ────────────────────────────────────────────────────

async def async_prune_raw_track_segments(
    factory: StoreFactory,
    learning_zone_id: str,
    track_name: RawTrackName,
    *,
    policy: RetentionPolicy,
    clock: Clock,
) -> RawTrackPruneResult:
    """Best-effort retention enforcement for one (zone, track)'s raw segments.

    Never raises. Loads the track's SegmentIndex, inspects every SEALED
    segment's own stored metadata (never the active segment), deletes the
    segments the pure ``evaluate_segment_retention()`` decision marks, and
    saves the updated index back — but only when something was actually
    deleted (idempotent no-op otherwise).

    Non-fatal by construction:
      - a missing index (nothing captured yet for this track) is a no-op;
      - an unreadable/corrupt individual segment file is skipped — it cannot
        be positively identified, so it is left alone rather than guessed at;
      - a failed delete or index save is captured in ``last_error`` and does
        not abort already-completed deletes.
    """
    index_store = RawSegmentIndexStore(factory, learning_zone_id, track_name)
    try:
        raw_index = await index_store.load()
    except Exception as err:
        return RawTrackPruneResult(pruned_count=0, last_error=str(err))
    if raw_index is None:
        return RawTrackPruneResult(pruned_count=0, last_error=None)

    try:
        index = _decode_segment_index(raw_index, learning_zone_id, track_name)
    except Exception as err:
        return RawTrackPruneResult(pruned_count=0, last_error=str(err))

    segment_store = RawSegmentStore(factory, learning_zone_id, track_name)
    sealed_segments: dict[str, Mapping] = {}
    segment_id_to_sequence: dict[str, int] = {}
    for seq in range(index.next_sequence_number):
        try:
            payload = await segment_store.load_segment(seq)
        except Exception:
            continue  # unreadable/corrupt segment file -> skip, non-fatal
        if not isinstance(payload, Mapping):
            continue
        seg_meta = payload.get("segment")
        if not isinstance(seg_meta, Mapping):
            continue
        sid = seg_meta.get("segment_id")
        if not isinstance(sid, str):
            continue
        sealed_segments[sid] = seg_meta
        segment_id_to_sequence[sid] = seq

    decision = evaluate_segment_retention(
        index, sealed_segments, policy=policy, now_utc=clock.now_utc(),
    )
    if not decision.delete_segment_ids:
        return RawTrackPruneResult(pruned_count=0, last_error=None)

    deleted = 0
    last_error: Optional[str] = None
    for sid in decision.delete_segment_ids:
        seq = segment_id_to_sequence.get(sid)
        if seq is None:
            continue  # defensive; every delete-candidate came from sealed_segments
        try:
            await segment_store.delete_segment(seq)
            deleted += 1
        except Exception as err:
            last_error = str(err)  # non-fatal; keep trying remaining deletes

    try:
        await index_store.save(_encode_segment_index(decision.updated_index))
    except Exception as err:
        last_error = str(err)

    return RawTrackPruneResult(pruned_count=deleted, last_error=last_error)


# ── Episode retention ─────────────────────────────────────────────────────────

async def async_prune_episodes(
    factory: StoreFactory,
    learning_zone_id: str,
    episode_registry: EpisodeRegistry,
    *,
    clock: Clock,
) -> EpisodePruneResult:
    """Best-effort retention enforcement for one zone's EpisodesStore payload.

    Never raises. Groups stored episodes by their own ``episode_type`` field
    and applies each type's own declared RetentionPolicy (from
    ``episode_registry``) independently, so e.g. a short outcome-episode
    retention never affects heating-episode retention. Entries without a
    recognised ``episode_type`` are conservatively kept untouched — they
    cannot be matched to a policy.

    A missing store (nothing captured yet) is a no-op.
    """
    store = EpisodesStore(factory, learning_zone_id)
    try:
        raw_payload = await store.load()
    except Exception as err:
        return EpisodePruneResult(pruned_count=0, last_error=str(err))
    if raw_payload is None:
        return EpisodePruneResult(pruned_count=0, last_error=None)

    raw_entries = raw_payload.get("episodes") if isinstance(raw_payload, Mapping) else None
    if not isinstance(raw_entries, Mapping):
        return EpisodePruneResult(pruned_count=0, last_error=None)

    now_utc = clock.now_utc()

    # Group entries by their own episode_type field; unknown/missing types are
    # always kept (never matched to a policy, never deleted).
    by_type: dict[str, dict[str, object]] = {}
    unmatched: dict[str, object] = {}
    for eid, entry in raw_entries.items():
        etype = entry.get("episode_type") if isinstance(entry, Mapping) else None
        if isinstance(etype, str):
            by_type.setdefault(etype, {})[eid] = entry
        else:
            unmatched[eid] = entry

    merged_keep: dict[str, object] = dict(unmatched)
    total_deleted = 0
    last_error: Optional[str] = None

    for etype_value, group in by_type.items():
        definition = None
        for candidate in episode_registry.definitions():
            if candidate.episode_type.value == etype_value:
                definition = candidate
                break
        if definition is None:
            # Unregistered episode type -> conservative keep, no policy to apply.
            merged_keep.update(group)
            continue
        try:
            decision = evaluate_episode_retention(
                {"episodes": group}, policy=definition.retention, now_utc=now_utc,
            )
        except Exception as err:
            last_error = str(err)
            merged_keep.update(group)  # evaluation failed -> conservative keep
            continue
        merged_keep.update(decision.updated_payload["episodes"])
        total_deleted += len(decision.delete_episode_ids)

    if total_deleted == 0:
        return EpisodePruneResult(pruned_count=0, last_error=last_error)

    try:
        await store.save({"episodes": merged_keep})
    except Exception as err:
        return EpisodePruneResult(pruned_count=0, last_error=str(err))

    return EpisodePruneResult(pruned_count=total_deleted, last_error=last_error)


# ── Zone-level convenience aggregate ──────────────────────────────────────────

async def async_prune_zone_storage(
    factory: StoreFactory,
    learning_zone_id: str,
    raw_registry: RawTrackRegistry,
    episode_registry: EpisodeRegistry,
    *,
    clock: Clock,
) -> ZoneRetentionResult:
    """Best-effort retention enforcement for one zone's raw segments + episodes.

    Never raises. Iterates every raw track declared in ``raw_registry``,
    applying each track's own RetentionPolicy, then prunes the zone's single
    EpisodesStore (per-episode-type policy from ``episode_registry``).
    Intended for an infrequent, explicit maintenance call (e.g. zone startup)
    — not for per-cycle use. See module docstring for rationale.
    """
    raw_pruned = 0
    last_error: Optional[str] = None

    for definition in raw_registry.definitions():
        result = await async_prune_raw_track_segments(
            factory, learning_zone_id, definition.track_name,
            policy=definition.retention, clock=clock,
        )
        raw_pruned += result.pruned_count
        if result.last_error is not None:
            last_error = result.last_error

    episode_result = await async_prune_episodes(
        factory, learning_zone_id, episode_registry, clock=clock,
    )
    if episode_result.last_error is not None:
        last_error = episode_result.last_error

    return ZoneRetentionResult(
        raw_segments_pruned=raw_pruned,
        episodes_pruned=episode_result.pruned_count,
        last_error=last_error,
    )
