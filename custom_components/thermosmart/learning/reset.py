"""LE 2.0 zone initialization and reset (HA-dependent storage shell).

Creates only empty, versioned v2 store structures — never fabricated model
states or priors. Idempotent, recovery-safe, and driven by explicitly provided
registries (no hardcoded track lists). Uses an injectable store factory so it is
testable without a running Home Assistant. Performs no automatic work on import,
touches no coordinator / entity / config, and never imports or modifies the v1
store ``thermosmart_learning_data``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .clock import Clock, SystemClock
from .init_state import (
    CONTAINER_VERSION,
    INIT_VERSION,
    InitializationAction,
    InitializationError,
    InitializationParameters,
    InitializationResult,
    InitializationState,
    InitializationStatus,
    ResetReason,
)
from .registry import (
    EpisodeRegistry,
    ModelRegistry,
    RawTrackRegistry,
    validate_registry_graph,
)
from .raw_schemas import RawTrackName
from .storage import naming
from .storage.stores import (
    AdaptationHistoryStore,
    ApplicationLifecycleStore,
    EpisodesStore,
    GlobalIndexStore,
    ModelStateStore,
    RawSegmentIndexStore,
    RawSegmentStore,
    ResearchDailyStore,
    StoreFactory,
    StoreVersionError,
    SupportCriticalEventStore,
    ZoneMetadataStore,
)

V1_STORE_KEY = "thermosmart_learning_data"
V1_STORE_VERSION = 1

_MODEL_STATE_CONTAINER_VERSION = 1


def _comp_raw_index(track: RawTrackName) -> str:
    return f"raw_index:{track.value}"


# -- empty component payloads (no fabricated content) -------------------------

def _empty_raw_index(track: RawTrackName) -> dict:
    return {
        "track_name": track.value,
        "track_schema_version": 1,
        "active_segment_id": None,
        "sealed_segment_ids": [],
        "next_sequence_number": 0,
    }


def _empty_episodes() -> dict:
    return {"episodes": {}}


def _empty_model_state() -> dict:
    # Empty and versioned. Concrete models add their state additively later.
    return {"models": {}, "model_state_container_version": _MODEL_STATE_CONTAINER_VERSION}


# -- v1 detection (read-only, never imported / modified) ----------------------

async def _detect_v1(factory: StoreFactory) -> bool:
    store = factory.create(V1_STORE_KEY, V1_STORE_VERSION)
    data = await store.async_load()
    return data is not None


# -- component validity -------------------------------------------------------

def _container_valid(data: object, learning_zone_id: str) -> Optional[str]:
    if not isinstance(data, dict):
        return "container is not a mapping"
    if data.get("learning_zone_id") != learning_zone_id:
        return "learning_zone_id mismatch"
    if data.get("container_version") != CONTAINER_VERSION:
        return "container_version incompatible"
    return None


def _component_valid(name: str, data: object) -> bool:
    if not isinstance(data, dict):
        return False
    if name.startswith("raw_index:"):
        return "track_schema_version" in data
    if name == "episodes":
        return "episodes" in data
    if name == "models":
        return "models" in data
    return True


# -- store accessors ----------------------------------------------------------

def _raw_tracks(raw_registry: RawTrackRegistry) -> tuple[RawTrackName, ...]:
    return tuple(raw_registry.names())


async def _load_components(
    factory: StoreFactory, lz: str, tracks: tuple[RawTrackName, ...]
) -> tuple[list[str], list[str], list[str]]:
    """Return (present, missing, corrupted) component names."""
    present: list[str] = []
    missing: list[str] = []
    corrupted: list[str] = []

    async def check(name: str, store) -> None:
        data = await store.load()
        if data is None:
            missing.append(name)
        elif _component_valid(name, data):
            present.append(name)
        else:
            corrupted.append(name)

    for track in tracks:
        await check(_comp_raw_index(track), RawSegmentIndexStore(factory, lz, track))
    await check("episodes", EpisodesStore(factory, lz))
    await check("models", ModelStateStore(factory, lz))
    return present, missing, corrupted


async def _create_component(factory: StoreFactory, lz: str, name: str) -> None:
    if name.startswith("raw_index:"):
        track = RawTrackName(name.split(":", 1)[1])
        await RawSegmentIndexStore(factory, lz, track).save(_empty_raw_index(track))
    elif name == "episodes":
        await EpisodesStore(factory, lz).save(_empty_episodes())
    elif name == "models":
        await ModelStateStore(factory, lz).save(_empty_model_state())


def _all_component_names(tracks: tuple[RawTrackName, ...]) -> list[str]:
    return [_comp_raw_index(t) for t in tracks] + ["episodes", "models"]


async def _write_container(
    factory: StoreFactory, lz: str, entry_id: str, status: InitializationStatus,
    *, clock: Clock, v1_detected: bool, components: list[str], info_logged: bool,
    created_ts: str,
) -> None:
    await ZoneMetadataStore(factory, lz).save({
        "learning_zone_id": lz,
        "entry_id": entry_id,
        "container_version": CONTAINER_VERSION,
        "init_status": status.value,
        "init_version": INIT_VERSION,
        "created_ts": created_ts,
        "initialized_ts": clock.now_utc().isoformat()
        if status is InitializationStatus.INITIALIZED else None,
        "raw_tracks": [c.split(":", 1)[1] for c in components if c.startswith("raw_index:")],
        "v1_detected": v1_detected,
        "v1_import_skipped": True,
        "components": components,
        "info_logged": info_logged,
    })


# -- public API ---------------------------------------------------------------

async def ensure_v2_initialized(
    factory: StoreFactory,
    entry_id: str,
    *,
    raw_registry: RawTrackRegistry,
    episode_registry: EpisodeRegistry,
    model_registry: ModelRegistry,
    clock: Optional[Clock] = None,
    params: Optional[InitializationParameters] = None,
) -> InitializationResult:
    """Idempotently ensure a zone's empty v2 stores exist. Never overwrites
    valid data; resumes partial initialization; reports a structured result."""
    clock = clock or SystemClock()
    params = params or InitializationParameters()
    lz = naming.validate_learning_zone_id(entry_id)  # decision: learning_zone_id == entry_id

    # registry graph must be valid (empty ModelRegistry is allowed)
    issues = validate_registry_graph(raw_registry, episode_registry, model_registry)
    if issues:
        return InitializationResult(
            status=InitializationStatus.FAILED_TERMINAL,
            action=InitializationAction.FAILED, learning_zone_id=lz,
            errors=tuple(i.message for i in issues),
            timestamp_utc=clock.now_utc().isoformat())

    tracks = _raw_tracks(raw_registry)
    v1_detected = await _detect_v1(factory)

    try:
        container = ZoneMetadataStore(factory, lz)
        existing = await container.load()
    except StoreVersionError as err:
        return _recovery(lz, str(err), v1_detected, clock)

    if existing is not None:
        problem = _container_valid(existing, lz)
        if problem is not None:
            return _recovery(lz, problem, v1_detected, clock)

    created_ts = (existing or {}).get("created_ts") or clock.now_utc().isoformat()
    present, missing, corrupted = await _load_components(factory, lz, tracks)

    if corrupted:
        return _recovery(lz, f"corrupted components: {corrupted}", v1_detected, clock,
                         failed_component=corrupted[0])

    status_value = (existing or {}).get("init_status")
    if existing is not None and status_value == InitializationStatus.INITIALIZED.value \
            and not missing:
        return InitializationResult(
            status=InitializationStatus.INITIALIZED,
            action=InitializationAction.ALREADY_INITIALIZED, learning_zone_id=lz,
            reused_components=tuple(present), v1_detected=v1_detected,
            info_log_emitted=False, timestamp_utc=clock.now_utc().isoformat())

    fresh = existing is None
    info_already = (existing or {}).get("info_logged", False)
    created: list[str] = []
    try:
        # mark INITIALIZING (provisional container) before creating components
        await _write_container(factory, lz, entry_id, InitializationStatus.INITIALIZING,
                               clock=clock, v1_detected=v1_detected,
                               components=_all_component_names(tracks),
                               info_logged=info_already, created_ts=created_ts)
        for name in missing:
            await _create_component(factory, lz, name)
            created.append(name)
        # validate everything before flipping the marker
        present2, missing2, corrupted2 = await _load_components(factory, lz, tracks)
        if missing2 or corrupted2:
            raise InitializationError(f"post-create validation failed: "
                                      f"missing={missing2} corrupted={corrupted2}")
        await _write_container(factory, lz, entry_id, InitializationStatus.INITIALIZED,
                               clock=clock, v1_detected=v1_detected,
                               components=_all_component_names(tracks),
                               info_logged=True, created_ts=created_ts)
    except InitializationError:
        raise
    except Exception as err:  # noqa: BLE001 - a failed save must be recoverable
        failed = missing[len(created)] if len(created) < len(missing) else None
        return InitializationResult(
            status=InitializationStatus.FAILED_RECOVERABLE,
            action=InitializationAction.FAILED, learning_zone_id=lz,
            failed_component=failed, v1_detected=v1_detected, errors=(str(err),),
            timestamp_utc=clock.now_utc().isoformat())

    action = (InitializationAction.INITIALIZED_NOW if fresh
              else InitializationAction.RECOVERED_PARTIAL)
    return InitializationResult(
        status=InitializationStatus.INITIALIZED, action=action, learning_zone_id=lz,
        created_components=tuple(created), reused_components=tuple(present),
        v1_detected=v1_detected, info_log_emitted=not info_already,
        timestamp_utc=clock.now_utc().isoformat())


def _recovery(lz, message, v1_detected, clock, failed_component=None):
    return InitializationResult(
        status=InitializationStatus.RECOVERY_REQUIRED,
        action=InitializationAction.FAILED, learning_zone_id=lz,
        failed_component=failed_component, v1_detected=v1_detected,
        errors=(message,), timestamp_utc=clock.now_utc().isoformat())


async def initialize_zone(
    factory: StoreFactory, entry_id: str, *, raw_registry, episode_registry,
    model_registry, clock=None, params=None,
) -> InitializationResult:
    """Explicit fresh/resumed initialization (alias of ensure for clarity)."""
    return await ensure_v2_initialized(
        factory, entry_id, raw_registry=raw_registry, episode_registry=episode_registry,
        model_registry=model_registry, clock=clock, params=params)


async def inspect_initialization_state(
    factory: StoreFactory, entry_id: str, *, raw_registry: RawTrackRegistry,
) -> InitializationState:
    lz = naming.validate_learning_zone_id(entry_id)
    tracks = _raw_tracks(raw_registry)
    v1_detected = await _detect_v1(factory)
    try:
        existing = await ZoneMetadataStore(factory, lz).load()
    except StoreVersionError:
        return InitializationState(InitializationStatus.RECOVERY_REQUIRED,
                                   learning_zone_id=lz, v1_detected=v1_detected)
    if existing is None:
        return InitializationState(InitializationStatus.NOT_INITIALIZED,
                                   learning_zone_id=lz, v1_detected=v1_detected)
    present, missing, corrupted = await _load_components(factory, lz, tracks)
    status = InitializationStatus(existing.get("init_status",
                                               InitializationStatus.NOT_INITIALIZED.value))
    if corrupted:
        status = InitializationStatus.RECOVERY_REQUIRED
    elif missing and status is InitializationStatus.INITIALIZED:
        status = InitializationStatus.INITIALIZING
    return InitializationState(
        status=status, learning_zone_id=lz, container_version=existing.get("container_version"),
        present_components=tuple(present), missing_components=tuple(missing),
        corrupted_components=tuple(corrupted), v1_detected=v1_detected)


async def reset_v2_learning_state(
    factory: StoreFactory, entry_id: str, reason: ResetReason, *,
    raw_registry: RawTrackRegistry, episode_registry: EpisodeRegistry,
    model_registry: ModelRegistry, clock: Optional[Clock] = None,
    reinitialize: bool = True,
) -> InitializationResult:
    """Explicitly delete and (optionally) re-create a zone's v2 learning stores.

    Only ever called explicitly. Touches no config / options / entities /
    device bindings, and never the v1 store. Requires a structured reason.
    """
    if not isinstance(reason, ResetReason):
        raise InitializationError("reset requires a ResetReason")
    clock = clock or SystemClock()
    lz = naming.validate_learning_zone_id(entry_id)
    tracks = _raw_tracks(raw_registry)

    for track in tracks:
        await RawSegmentIndexStore(factory, lz, track).delete()
    await EpisodesStore(factory, lz).delete()
    await ModelStateStore(factory, lz).delete()
    await ZoneMetadataStore(factory, lz).delete()
    try:
        await AdaptationHistoryStore(factory, lz).delete()
    except Exception:
        pass  # non-fatal: store may not exist yet on first reset
    try:
        await ApplicationLifecycleStore(factory, lz).delete()
    except Exception:
        pass  # non-fatal: store may not exist yet on first reset

    if not reinitialize:
        return InitializationResult(
            status=InitializationStatus.NOT_INITIALIZED,
            action=InitializationAction.FAILED, learning_zone_id=lz,
            errors=(f"reset:{reason.value}",), timestamp_utc=clock.now_utc().isoformat())
    return await ensure_v2_initialized(
        factory, entry_id, raw_registry=raw_registry, episode_registry=episode_registry,
        model_registry=model_registry, clock=clock)


@dataclass(frozen=True)
class ZonePurgeResult:
    """Outcome of purging one zone's LE 2.0 storage on real removal. Never raises."""
    deleted_keys: tuple[str, ...]
    errors: tuple[str, ...]


async def async_purge_zone_storage(
    factory: StoreFactory, entry_id: str, *, raw_registry: RawTrackRegistry,
) -> ZonePurgeResult:
    """Delete every known LE 2.0 storage key for one zone.

    Only ever called on REAL zone/entry removal (``async_remove_entry`` in
    ``__init__.py``) — never on unload/reload, which must never lose data.
    Every key deleted here is derived the same way it was created (via
    ``learning/storage/naming.py`` through the store wrapper classes) — no
    directory scan, no wildcard, no cross-zone reach. Best-effort per store:
    one failing delete never blocks the rest. Does not touch the v1 store
    ``thermosmart_learning_data`` (shared across zones — that is the caller's
    responsibility once it has confirmed no zone remains) nor the LE 2.0
    shadow-state store (a different key scheme; see
    ``learning/runtime/ha_store.py``'s ``HomeAssistantStoreAdapter.async_delete()``).
    """
    lz = naming.validate_learning_zone_id(entry_id)
    deleted: list[str] = []
    errors: list[str] = []

    async def _delete(store, key: str) -> None:
        try:
            await store.delete()
            deleted.append(key)
        except Exception as err:
            errors.append(f"{key}: {type(err).__name__}")

    await _delete(ZoneMetadataStore(factory, lz), naming.container_key(lz))
    await _delete(ModelStateStore(factory, lz), naming.model_state_key(lz))
    await _delete(AdaptationHistoryStore(factory, lz), naming.adaptation_history_key(lz))
    await _delete(ApplicationLifecycleStore(factory, lz), naming.application_lifecycle_key(lz))
    await _delete(EpisodesStore(factory, lz), naming.episodes_key(lz))
    await _delete(SupportCriticalEventStore(factory, lz), naming.support_critical_events_key(lz))
    await _delete(ResearchDailyStore(factory, lz), naming.research_daily_key(lz))

    for track in raw_registry.names():
        index_key = naming.raw_index_key(lz, track)
        next_seq = 0
        try:
            raw = await RawSegmentIndexStore(factory, lz, track).load()
            next_seq = int((raw or {}).get("next_sequence_number", 0))
        except Exception as err:
            errors.append(f"{index_key} (index load): {type(err).__name__}")
        await _delete(RawSegmentIndexStore(factory, lz, track), index_key)

        segment_store = RawSegmentStore(factory, lz, track)
        for seq in range(next_seq):
            seg_key = naming.raw_segment_key(lz, track, seq)
            try:
                await segment_store.delete_segment(seq)
                deleted.append(seg_key)
            except Exception as err:
                errors.append(f"{seg_key}: {type(err).__name__}")

    return ZonePurgeResult(deleted_keys=tuple(deleted), errors=tuple(errors))


async def rebuild_global_index(
    factory: StoreFactory, entry_ids: list[str], *, clock: Optional[Clock] = None,
) -> dict:
    """Rebuild the reconstructable global cache index from zone containers."""
    clock = clock or SystemClock()
    learning_zones: dict[str, dict] = {}
    for entry_id in entry_ids:
        lz = naming.validate_learning_zone_id(entry_id)
        container = await ZoneMetadataStore(factory, lz).load()
        if container is not None:
            learning_zones[lz] = {
                "entry_id": container.get("entry_id"),
                "container_version": container.get("container_version"),
            }
    index = {"learning_zones": learning_zones, "rebuilt_ts": clock.now_utc().isoformat()}
    await GlobalIndexStore(factory).save(index)
    return index
