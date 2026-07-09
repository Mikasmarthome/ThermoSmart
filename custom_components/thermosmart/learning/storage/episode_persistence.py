"""Pure, bounded helpers for appending completed episodes into an
EpisodesStore-shaped payload, with per-episode-type retention pruning applied
immediately after every append.

Live-wired: called from ``LearningShadowController.record_completed_episode_safe()``
(``learning/runtime/ha_integration.py``), bound as ``LearningRuntime``'s
``episode_sink`` and invoked from ``run_cycle()``'s completed-episode loop
(``learning/runtime/lifecycle.py``).

Pipeline:
    completed episode dataclass(es)
    -> serialize_episode()            (episode_serialization.py)
    -> append into an EpisodesStore "episodes" dict, skipping unknown/
       malformed episodes and duplicate episode_ids
    -> evaluate_episode_retention()   (retention.py) applied per episode type,
       using that type's own EpisodeDefinition.retention from the registry
       (capture_registry.py) — no new/arbitrary retention numbers here.

No HA imports. No runtime imports. No control imports. No storage I/O of any
kind — every function here is a pure transformation of in-memory dicts; the
caller decides if/when to persist the resulting payload through
``EpisodesStore`` (via ``LearningCaptureStores.episodes_store()``). Never
mutates the ``episode`` objects passed in, nor the input ``payload`` in place
(a new dict is always returned).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from ..registry import EpisodeRegistry
from .episode_serialization import serialize_episode
from .retention import evaluate_episode_retention


@dataclass(frozen=True)
class EpisodeAppendResult:
    """Outcome of an append/prune pass over an EpisodesStore payload.

    Describes what changed — never mutates the input payload in place.
    """
    updated_payload: dict
    appended_episode_ids: tuple[str, ...]
    skipped_count: int          # unknown/malformed episodes, not appended
    duplicate_count: int        # episode_id already present, not re-appended
    pruned_episode_ids: tuple[str, ...]


def _episodes_dict(payload: Any) -> dict:
    if isinstance(payload, Mapping):
        existing = payload.get("episodes")
        if isinstance(existing, Mapping):
            return dict(existing)
    return {}


def _definition_for_type(episode_registry: EpisodeRegistry, episode_type_value: Any):
    for candidate in episode_registry.definitions():
        if candidate.episode_type.value == episode_type_value:
            return candidate
    return None


def _prune_entries_by_type(
    entries: dict, episode_type_values: set, *,
    episode_registry: EpisodeRegistry, now_utc: datetime,
) -> tuple[dict, list[str]]:
    """Apply each given EpisodeType's own RetentionPolicy to its own subset
    of ``entries``. Mutates and returns the SAME dict (caller already owns a
    private copy). Never raises — a failed evaluation for one type leaves
    that type's entries untouched and continues with the others.
    """
    pruned: list[str] = []
    for etype_value in episode_type_values:
        try:
            definition = _definition_for_type(episode_registry, etype_value)
            if definition is None:
                continue  # unregistered type -> no policy to apply, leave as-is
            group = {
                eid: e for eid, e in entries.items()
                if isinstance(e, Mapping) and e.get("episode_type") == etype_value
            }
            decision = evaluate_episode_retention(
                {"episodes": group}, policy=definition.retention, now_utc=now_utc,
            )
        except Exception:
            continue  # non-fatal: registry/retention failure leaves entries as-is
        for pruned_id in decision.delete_episode_ids:
            entries.pop(pruned_id, None)
            pruned.append(pruned_id)
    return entries, pruned


def append_completed_episode(
    payload: Any,
    episode: Any,
    *,
    episode_registry: EpisodeRegistry,
    now_utc: datetime,
) -> EpisodeAppendResult:
    """Append ONE completed episode dataclass into an EpisodesStore payload.

    Pure function — no storage I/O, no mutation of the input payload or of
    ``episode``. Only pass fully completed (materialised) episode dataclasses
    — this function has no way to detect "in progress" state itself, that is
    the caller's responsibility (see module docstring). Never raises.
    """
    return append_completed_episodes(
        payload, (episode,), episode_registry=episode_registry, now_utc=now_utc,
    )


def append_completed_episodes(
    payload: Any,
    episodes: Sequence[Any],
    *,
    episode_registry: EpisodeRegistry,
    now_utc: datetime,
) -> EpisodeAppendResult:
    """Append a sequence of completed episodes; see append_completed_episode().

    Processes episodes in order, preserving that order among the entries that
    are actually appended. Each episode is serialized independently via
    ``serialize_episode()`` — an unrecognised/malformed episode is skipped
    (``skipped_count`` incremented), never raises. A duplicate ``episode_id``
    (already present in ``payload``, or repeated within this same call) is
    not re-appended (``duplicate_count`` incremented) — the existing entry is
    left untouched. After all appends, each touched EpisodeType's own
    ``RetentionPolicy`` (from ``episode_registry``) is applied immediately to
    that type's own entries only — other, untouched types are never re-pruned
    in this call (bounded, cheap).
    """
    entries = _episodes_dict(payload)
    appended: list[str] = []
    skipped = 0
    duplicates = 0
    touched_types: set = set()

    for episode in episodes:
        serialized = serialize_episode(episode)
        if serialized is None:
            skipped += 1
            continue
        eid = serialized.get("episode_id")
        if not isinstance(eid, str) or not eid:
            skipped += 1
            continue
        if eid in entries:
            duplicates += 1
            continue
        entries[eid] = serialized
        appended.append(eid)
        touched_types.add(serialized.get("episode_type"))

    entries, pruned = _prune_entries_by_type(
        entries, touched_types, episode_registry=episode_registry, now_utc=now_utc,
    )

    return EpisodeAppendResult(
        updated_payload={"episodes": entries},
        appended_episode_ids=tuple(appended),
        skipped_count=skipped,
        duplicate_count=duplicates,
        pruned_episode_ids=tuple(pruned),
    )


def prune_episode_entries(
    payload: Any,
    *,
    episode_registry: EpisodeRegistry,
    now_utc: datetime,
) -> EpisodeAppendResult:
    """Apply each present EpisodeType's own RetentionPolicy across ALL
    entries in an EpisodesStore payload, without appending anything new.

    A standalone maintenance pass — mirrors
    ``retention_service.async_prune_episodes()``'s per-type grouping, but
    purely in-memory (no storage I/O). Useful for re-running retention on an
    already-loaded payload, e.g. right before a save. Never raises.
    """
    entries = _episodes_dict(payload)
    present_types = {
        e.get("episode_type") for e in entries.values() if isinstance(e, Mapping)
    }
    entries, pruned = _prune_entries_by_type(
        entries, present_types, episode_registry=episode_registry, now_utc=now_utc,
    )
    return EpisodeAppendResult(
        updated_payload={"episodes": entries},
        appended_episode_ids=(),
        skipped_count=0,
        duplicate_count=0,
        pruned_episode_ids=tuple(pruned),
    )
