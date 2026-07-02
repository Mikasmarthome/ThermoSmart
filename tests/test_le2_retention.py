"""Tests for LE2 raw segment and episode retention enforcement.

Covers:
  - Pure decision logic: evaluate_segment_retention(), evaluate_episode_retention()
    (custom_components/thermosmart/learning/storage/retention.py)
  - Store-integration layer: async_prune_raw_track_segments(), async_prune_episodes(),
    async_prune_zone_storage()
    (custom_components/thermosmart/learning/storage/retention_service.py)

17 test groups:
  T1  — Raw retention keeps active/open segment
  T2  — Raw retention deletes old sealed segment
  T3  — Raw retention keeps malformed/no-timestamp segment
  T4  — Raw retention respects zone separation
  T5  — Raw retention respects track separation
  T6  — Missing segment file does not crash cleanup
  T7  — Delete error is non-fatal
  T8  — Segment index remains consistent after pruning
  T9  — Episode retention deletes old episodes
  T10 — Episode retention keeps recent episodes
  T11 — Episode retention keeps malformed/no-timestamp episodes conservatively
  T12 — Episode retention respects max count if policy has max count
  T13 — Episode retention does not mix episode types incorrectly
  T14 — Retention is idempotent
  T15 — Retention does not run on every observe cycle (not wired to hot path)
  T16 — No Heating-Control path is touched
  T17 — Existing tests remain green (regression smoke-test)
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from typing import Any, Optional

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.raw_schemas import RawTrackName, RoomObservation
from custom_components.thermosmart.learning.registry import (
    EpisodeRegistry,
    EpisodeDefinition,
    PrivacyClass,
    RawTrackDefinition,
    RawTrackRegistry,
    RetentionPolicy,
)
from custom_components.thermosmart.learning.episode_schemas import EpisodeType, HeatingEpisode
from custom_components.thermosmart.learning.contracts import ExportScope
from custom_components.thermosmart.learning.storage.segments import SegmentIndex
from custom_components.thermosmart.learning.storage.retention import (
    evaluate_segment_retention,
    evaluate_episode_retention,
)
from custom_components.thermosmart.learning.storage import retention_service as _retention_service
from custom_components.thermosmart.learning.storage.retention_service import (
    async_prune_raw_track_segments,
    async_prune_episodes,
    async_prune_zone_storage,
)
from custom_components.thermosmart.learning.storage.stores import (
    RawSegmentIndexStore,
    RawSegmentStore,
    EpisodesStore,
)

_ZONE_A = "zone_alpha_01"
_ZONE_B = "zone_beta_02"
_NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _seg_meta(seq: int, sealed_utc: Optional[str], segment_id: Optional[str] = None) -> dict:
    return {
        "segment_id": segment_id or f"seg-{seq}",
        "sequence_number": seq,
        "sealed_utc": sealed_utc,
        "record_count": 1,
    }


def _index(
    *, active_segment_id, sealed_segment_ids, next_sequence_number,
    zone=_ZONE_A, track=RawTrackName.ROOM,
) -> SegmentIndex:
    return SegmentIndex(
        learning_zone_id=zone,
        track_name=track,
        track_schema_version=1,
        next_sequence_number=next_sequence_number,
        active_segment_id=active_segment_id,
        sealed_segment_ids=tuple(sealed_segment_ids),
    )


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# ── T1: Raw retention keeps active/open segment ──────────────────────────────

def test_t1_keeps_active_segment_even_if_old():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    index = _index(
        active_segment_id="seg-active",
        sealed_segment_ids=("seg-active",),  # defensive: even if present in sealed list
        next_sequence_number=1,
    )
    metas = {"seg-active": _seg_meta(0, old_ts, "seg-active")}
    policy = RetentionPolicy(max_age_days=30)
    decision = evaluate_segment_retention(index, metas, policy=policy, now_utc=_NOW)
    assert "seg-active" not in decision.delete_segment_ids
    assert "seg-active" in decision.keep_segment_ids


# ── T2: Raw retention deletes old sealed segment ─────────────────────────────

def test_t2_deletes_old_sealed_segment():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    index = _index(
        active_segment_id=None,
        sealed_segment_ids=("seg-old",),
        next_sequence_number=1,
    )
    metas = {"seg-old": _seg_meta(0, old_ts)}
    policy = RetentionPolicy(max_age_days=30)
    decision = evaluate_segment_retention(index, metas, policy=policy, now_utc=_NOW)
    assert decision.delete_segment_ids == ("seg-old",)
    assert decision.keep_segment_ids == ()
    assert decision.updated_index.sealed_segment_ids == ()


def test_t2_keeps_recent_sealed_segment():
    recent_ts = _iso(_NOW)
    index = _index(
        active_segment_id=None,
        sealed_segment_ids=("seg-recent",),
        next_sequence_number=1,
    )
    metas = {"seg-recent": _seg_meta(0, recent_ts)}
    policy = RetentionPolicy(max_age_days=30)
    decision = evaluate_segment_retention(index, metas, policy=policy, now_utc=_NOW)
    assert decision.delete_segment_ids == ()
    assert decision.keep_segment_ids == ("seg-recent",)


# ── T3: Raw retention keeps malformed/no-timestamp segment ──────────────────

@pytest.mark.parametrize("bad_ts", [None, "", "not-a-timestamp", 12345])
def test_t3_keeps_malformed_timestamp_segment(bad_ts):
    index = _index(
        active_segment_id=None,
        sealed_segment_ids=("seg-bad",),
        next_sequence_number=1,
    )
    metas = {"seg-bad": _seg_meta(0, bad_ts)}
    policy = RetentionPolicy(max_age_days=30)
    decision = evaluate_segment_retention(index, metas, policy=policy, now_utc=_NOW)
    assert decision.delete_segment_ids == ()
    assert "seg-bad" in decision.keep_segment_ids


def test_t3_keeps_segment_with_unknown_metadata():
    """A sealed_segment_id with no corresponding metadata at all is kept."""
    index = _index(
        active_segment_id=None,
        sealed_segment_ids=("seg-unknown",),
        next_sequence_number=1,
    )
    policy = RetentionPolicy(max_age_days=30)
    decision = evaluate_segment_retention(index, {}, policy=policy, now_utc=_NOW)
    assert decision.delete_segment_ids == ()
    assert "seg-unknown" in decision.keep_segment_ids


def test_t3_no_policy_thresholds_prunes_nothing():
    old_ts = _iso(datetime(2000, 1, 1, tzinfo=timezone.utc))
    index = _index(
        active_segment_id=None,
        sealed_segment_ids=("seg-ancient",),
        next_sequence_number=1,
    )
    metas = {"seg-ancient": _seg_meta(0, old_ts)}
    policy = RetentionPolicy()  # no max_age_days, no max_records
    decision = evaluate_segment_retention(index, metas, policy=policy, now_utc=_NOW)
    assert decision.delete_segment_ids == ()


def test_t3_max_records_evicts_oldest_by_sequence():
    index = _index(
        active_segment_id=None,
        sealed_segment_ids=("seg-0", "seg-1", "seg-2"),
        next_sequence_number=3,
    )
    metas = {
        "seg-0": _seg_meta(0, _iso(_NOW)),
        "seg-1": _seg_meta(1, _iso(_NOW)),
        "seg-2": _seg_meta(2, _iso(_NOW)),
    }
    policy = RetentionPolicy(max_records=2)
    decision = evaluate_segment_retention(index, metas, policy=policy, now_utc=_NOW)
    assert decision.delete_segment_ids == ("seg-0",)
    assert set(decision.keep_segment_ids) == {"seg-1", "seg-2"}


def test_t3_max_records_never_evicts_unknown_metadata():
    index = _index(
        active_segment_id=None,
        sealed_segment_ids=("seg-known", "seg-unknown"),
        next_sequence_number=2,
    )
    metas = {"seg-known": _seg_meta(0, _iso(_NOW))}
    # max_records=1 with 2 sealed segments forces an overflow eviction decision.
    policy = RetentionPolicy(max_records=1)
    decision = evaluate_segment_retention(index, metas, policy=policy, now_utc=_NOW)
    # Only "seg-known" has known metadata -> only it may be evicted.
    assert "seg-unknown" not in decision.delete_segment_ids
    assert "seg-unknown" in decision.keep_segment_ids


# ── T4: Raw retention respects zone separation ───────────────────────────────

def test_t4_zone_separation_pure_logic():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    index_a = _index(
        active_segment_id=None, sealed_segment_ids=("seg-a-old",),
        next_sequence_number=1, zone=_ZONE_A,
    )
    index_b = _index(
        active_segment_id=None, sealed_segment_ids=("seg-b-old",),
        next_sequence_number=1, zone=_ZONE_B,
    )
    policy = RetentionPolicy(max_age_days=30)
    meta_a = {"seg-a-old": _seg_meta(0, old_ts)}
    meta_b = {"seg-b-old": _seg_meta(0, old_ts)}

    decision_a = evaluate_segment_retention(index_a, meta_a, policy=policy, now_utc=_NOW)
    decision_b = evaluate_segment_retention(index_b, meta_b, policy=policy, now_utc=_NOW)

    assert decision_a.delete_segment_ids == ("seg-a-old",)
    assert decision_b.delete_segment_ids == ("seg-b-old",)
    assert decision_a.updated_index.learning_zone_id == _ZONE_A
    assert decision_b.updated_index.learning_zone_id == _ZONE_B


# ── T5: Raw retention respects track separation ──────────────────────────────

def test_t5_track_separation_pure_logic():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    index_room = _index(
        active_segment_id=None, sealed_segment_ids=("seg-room",),
        next_sequence_number=1, track=RawTrackName.ROOM,
    )
    index_trv = _index(
        active_segment_id=None, sealed_segment_ids=("seg-trv",),
        next_sequence_number=1, track=RawTrackName.TRV,
    )
    policy = RetentionPolicy(max_age_days=30)
    meta_room = {"seg-room": _seg_meta(0, old_ts)}
    meta_trv = {"seg-trv": _seg_meta(0, old_ts)}

    decision_room = evaluate_segment_retention(index_room, meta_room, policy=policy, now_utc=_NOW)
    decision_trv = evaluate_segment_retention(index_trv, meta_trv, policy=policy, now_utc=_NOW)

    assert decision_room.updated_index.track_name is RawTrackName.ROOM
    assert decision_trv.updated_index.track_name is RawTrackName.TRV
    assert decision_room.delete_segment_ids == ("seg-room",)
    assert decision_trv.delete_segment_ids == ("seg-trv",)


# ── Store-layer fixtures ──────────────────────────────────────────────────────

class _FakeRawStore:
    def __init__(self, *, initial: Any = None, load_raises=None, save_raises=None,
                 remove_raises=None) -> None:
        self._data = initial
        self._load_raises = load_raises
        self._save_raises = save_raises
        self._remove_raises = remove_raises
        self.save_call_count = 0
        self.remove_call_count = 0

    async def async_load(self) -> Any:
        if self._load_raises:
            raise self._load_raises
        return self._data

    async def async_save(self, data: Any) -> None:
        self.save_call_count += 1
        if self._save_raises:
            raise self._save_raises
        self._data = data

    async def async_remove(self) -> None:
        self.remove_call_count += 1
        if self._remove_raises:
            raise self._remove_raises
        self._data = None


class _FakeStoreFactory:
    def __init__(self) -> None:
        self._stores: dict[str, _FakeRawStore] = {}

    def create(self, key: str, version: int) -> _FakeRawStore:
        if key not in self._stores:
            self._stores[key] = _FakeRawStore()
        return self._stores[key]

    def get(self, key: str) -> Optional[_FakeRawStore]:
        return self._stores.get(key)

    def inject(self, key: str, store: _FakeRawStore) -> None:
        self._stores[key] = store


def _index_key(zone: str, track: RawTrackName) -> str:
    from custom_components.thermosmart.learning.storage import naming
    return naming.raw_index_key(zone, track)


def _segment_key(zone: str, track: RawTrackName, seq: int) -> str:
    from custom_components.thermosmart.learning.storage import naming
    return naming.raw_segment_key(zone, track, seq)


def _envelope(data: Any, version: int = 1) -> dict:
    return {"store_schema_version": version, "data": data}


def _seal_segment_payload(seq: int, sealed_utc: Optional[str], segment_id: Optional[str] = None) -> dict:
    return {
        "format_version": 1,
        "segment": _seg_meta(seq, sealed_utc, segment_id),
        "records": [],
    }


# ── T6: Missing segment file does not crash cleanup ──────────────────────────

def test_t6_missing_segment_file_does_not_crash():
    factory = _FakeStoreFactory()
    factory.inject(_index_key(_ZONE_A, RawTrackName.ROOM), _FakeRawStore(initial=_envelope({
        "track_name": "room", "track_schema_version": 1,
        "active_segment_id": None, "sealed_segment_ids": ["seg-0"],
        "next_sequence_number": 1,
    })))
    # No segment file injected at all for sequence 0 -> load returns None.
    policy = RetentionPolicy(max_age_days=30)
    clock = FakeClock(_NOW)

    result = asyncio.run(async_prune_raw_track_segments(
        factory, _ZONE_A, RawTrackName.ROOM, policy=policy, clock=clock,
    ))
    assert result.pruned_count == 0
    assert result.last_error is None


def test_t6_missing_index_is_noop():
    factory = _FakeStoreFactory()  # nothing injected -> index load returns None
    policy = RetentionPolicy(max_age_days=30)
    clock = FakeClock(_NOW)

    result = asyncio.run(async_prune_raw_track_segments(
        factory, _ZONE_A, RawTrackName.ROOM, policy=policy, clock=clock,
    ))
    assert result.pruned_count == 0
    assert result.last_error is None


# ── T7: Delete error is non-fatal ─────────────────────────────────────────────

def test_t7_delete_error_is_nonfatal():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    factory = _FakeStoreFactory()
    factory.inject(_index_key(_ZONE_A, RawTrackName.ROOM), _FakeRawStore(initial=_envelope({
        "track_name": "room", "track_schema_version": 1,
        "active_segment_id": None, "sealed_segment_ids": ["seg-0", "seg-1"],
        "next_sequence_number": 2,
    })))
    factory.inject(
        _segment_key(_ZONE_A, RawTrackName.ROOM, 0),
        _FakeRawStore(initial=_envelope(_seal_segment_payload(0, old_ts, "seg-0")),
                      remove_raises=OSError("simulated disk error")),
    )
    factory.inject(
        _segment_key(_ZONE_A, RawTrackName.ROOM, 1),
        _FakeRawStore(initial=_envelope(_seal_segment_payload(1, old_ts, "seg-1"))),
    )
    policy = RetentionPolicy(max_age_days=30)
    clock = FakeClock(_NOW)

    result = asyncio.run(async_prune_raw_track_segments(
        factory, _ZONE_A, RawTrackName.ROOM, policy=policy, clock=clock,
    ))
    # seg-1 deleted successfully; seg-0's delete failed but did not raise.
    assert result.pruned_count == 1
    assert result.last_error is not None
    assert "simulated disk error" in result.last_error


# ── T8: Segment index remains consistent after pruning ───────────────────────

def test_t8_index_updated_after_pruning():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    factory = _FakeStoreFactory()
    factory.inject(_index_key(_ZONE_A, RawTrackName.ROOM), _FakeRawStore(initial=_envelope({
        "track_name": "room", "track_schema_version": 1,
        "active_segment_id": "seg-active", "sealed_segment_ids": ["seg-0"],
        "next_sequence_number": 2,
    })))
    factory.inject(
        _segment_key(_ZONE_A, RawTrackName.ROOM, 0),
        _FakeRawStore(initial=_envelope(_seal_segment_payload(0, old_ts, "seg-0"))),
    )
    factory.inject(
        _segment_key(_ZONE_A, RawTrackName.ROOM, 1),
        _FakeRawStore(initial=_envelope(_seal_segment_payload(1, None, "seg-active"))),
    )
    policy = RetentionPolicy(max_age_days=30)
    clock = FakeClock(_NOW)

    result = asyncio.run(async_prune_raw_track_segments(
        factory, _ZONE_A, RawTrackName.ROOM, policy=policy, clock=clock,
    ))
    assert result.pruned_count == 1

    index_store_data = factory.get(_index_key(_ZONE_A, RawTrackName.ROOM))._data
    saved_index = index_store_data["data"]
    assert saved_index["sealed_segment_ids"] == []
    assert saved_index["active_segment_id"] == "seg-active"  # untouched


# ── T9/T10: Episode retention deletes old, keeps recent ──────────────────────

def _episode_entry(end_ts: Optional[str], episode_type: str = "heating") -> dict:
    return {"end_ts": end_ts, "episode_type": episode_type}


def test_t9_episode_retention_deletes_old():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    payload = {"episodes": {"ep-old": _episode_entry(old_ts)}}
    policy = RetentionPolicy(max_age_days=30)
    decision = evaluate_episode_retention(payload, policy=policy, now_utc=_NOW)
    assert decision.delete_episode_ids == ("ep-old",)
    assert decision.keep_episode_ids == ()


def test_t10_episode_retention_keeps_recent():
    recent_ts = _iso(_NOW)
    payload = {"episodes": {"ep-recent": _episode_entry(recent_ts)}}
    policy = RetentionPolicy(max_age_days=30)
    decision = evaluate_episode_retention(payload, policy=policy, now_utc=_NOW)
    assert decision.delete_episode_ids == ()
    assert decision.keep_episode_ids == ("ep-recent",)


# ── T11: Episode retention keeps malformed/no-timestamp conservatively ──────

@pytest.mark.parametrize("bad_end_ts", [None, "", "garbage"])
def test_t11_episode_retention_keeps_malformed(bad_end_ts):
    payload = {"episodes": {"ep-bad": _episode_entry(bad_end_ts)}}
    policy = RetentionPolicy(max_age_days=30)
    decision = evaluate_episode_retention(payload, policy=policy, now_utc=_NOW)
    assert decision.delete_episode_ids == ()
    assert "ep-bad" in decision.keep_episode_ids


def test_t11_episode_retention_falls_back_to_start_ts():
    old_start = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    payload = {"episodes": {"ep-x": {"start_ts": old_start, "episode_type": "heating"}}}
    policy = RetentionPolicy(max_age_days=30)
    decision = evaluate_episode_retention(payload, policy=policy, now_utc=_NOW)
    assert decision.delete_episode_ids == ("ep-x",)


def test_t11_malformed_payload_shape_returns_empty_safely():
    decision = evaluate_episode_retention({"not_episodes": {}}, policy=RetentionPolicy(max_age_days=1), now_utc=_NOW)
    assert decision.delete_episode_ids == ()
    assert decision.keep_episode_ids == ()
    assert decision.updated_payload == {"episodes": {}}


# ── T12: Episode retention respects max count ────────────────────────────────

def test_t12_episode_retention_max_records():
    payload = {"episodes": {
        "ep-0": _episode_entry(_iso(_NOW)),
        "ep-1": _episode_entry(_iso(_NOW)),
        "ep-2": _episode_entry(_iso(_NOW)),
    }}
    # Give them distinct ages so eviction order is deterministic.
    payload["episodes"]["ep-0"]["end_ts"] = _iso(datetime(2026, 1, 1, tzinfo=timezone.utc))
    payload["episodes"]["ep-1"]["end_ts"] = _iso(datetime(2026, 3, 1, tzinfo=timezone.utc))
    payload["episodes"]["ep-2"]["end_ts"] = _iso(datetime(2026, 5, 1, tzinfo=timezone.utc))
    policy = RetentionPolicy(max_records=2)
    decision = evaluate_episode_retention(payload, policy=policy, now_utc=_NOW)
    assert decision.delete_episode_ids == ("ep-0",)
    assert set(decision.keep_episode_ids) == {"ep-1", "ep-2"}


# ── T13: Episode retention does not mix episode types incorrectly ───────────

def _episode_registry_with_policies(
    heating_policy: RetentionPolicy, outcome_policy: RetentionPolicy,
) -> EpisodeRegistry:
    from custom_components.thermosmart.learning.episode_schemas import (
        HeatingEpisode, OutcomeEpisode,
    )
    reg = EpisodeRegistry()
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.HEATING, schema_type=HeatingEpisode,
        episode_schema_version=1, builder_version=1,
        consumed_raw_tracks=(RawTrackName.ROOM,), max_trajectory_points=10,
        max_duration_seconds=3600, retention=heating_policy,
        trajectory_required=False, materialized=True,
    ))
    reg.register(EpisodeDefinition(
        episode_type=EpisodeType.OUTCOME, schema_type=OutcomeEpisode,
        episode_schema_version=1, builder_version=1,
        consumed_raw_tracks=(), max_trajectory_points=10,
        max_duration_seconds=3600, retention=outcome_policy,
        trajectory_required=True, materialized=True,
    ))
    return reg


def test_t13_episode_types_pruned_independently():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    recent_ts = _iso(_NOW)
    factory = _FakeStoreFactory()
    from custom_components.thermosmart.learning.storage import naming
    key = naming.episodes_key(_ZONE_A)
    factory.inject(key, _FakeRawStore(initial=_envelope({
        "episodes": {
            "heat-old": {"end_ts": old_ts, "episode_type": "heating"},
            "outcome-old": {"end_ts": old_ts, "episode_type": "outcome"},
        }
    })))
    # Heating: strict 30-day retention -> heat-old pruned.
    # Outcome: lenient 3650-day retention -> outcome-old survives.
    registry = _episode_registry_with_policies(
        heating_policy=RetentionPolicy(max_age_days=30),
        outcome_policy=RetentionPolicy(max_age_days=3650),
    )
    clock = FakeClock(_NOW)

    result = asyncio.run(async_prune_episodes(factory, _ZONE_A, registry, clock=clock))
    assert result.pruned_count == 1

    saved = factory.get(key)._data["data"]["episodes"]
    assert "heat-old" not in saved
    assert "outcome-old" in saved


def test_t13_unregistered_episode_type_kept():
    factory = _FakeStoreFactory()
    from custom_components.thermosmart.learning.storage import naming
    key = naming.episodes_key(_ZONE_A)
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    factory.inject(key, _FakeRawStore(initial=_envelope({
        "episodes": {"unknown-old": {"end_ts": old_ts, "episode_type": "window_cooling"}}
    })))
    registry = _episode_registry_with_policies(
        heating_policy=RetentionPolicy(max_age_days=1),
        outcome_policy=RetentionPolicy(max_age_days=1),
    )  # window_cooling is not registered here
    clock = FakeClock(_NOW)

    result = asyncio.run(async_prune_episodes(factory, _ZONE_A, registry, clock=clock))
    assert result.pruned_count == 0
    saved = factory.get(key)._data["data"]["episodes"]
    assert "unknown-old" in saved


# ── T14: Retention is idempotent ─────────────────────────────────────────────

def test_t14_raw_retention_idempotent():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    factory = _FakeStoreFactory()
    factory.inject(_index_key(_ZONE_A, RawTrackName.ROOM), _FakeRawStore(initial=_envelope({
        "track_name": "room", "track_schema_version": 1,
        "active_segment_id": None, "sealed_segment_ids": ["seg-0"],
        "next_sequence_number": 1,
    })))
    factory.inject(
        _segment_key(_ZONE_A, RawTrackName.ROOM, 0),
        _FakeRawStore(initial=_envelope(_seal_segment_payload(0, old_ts, "seg-0"))),
    )
    policy = RetentionPolicy(max_age_days=30)
    clock = FakeClock(_NOW)

    first = asyncio.run(async_prune_raw_track_segments(
        factory, _ZONE_A, RawTrackName.ROOM, policy=policy, clock=clock,
    ))
    second = asyncio.run(async_prune_raw_track_segments(
        factory, _ZONE_A, RawTrackName.ROOM, policy=policy, clock=clock,
    ))
    assert first.pruned_count == 1
    assert second.pruned_count == 0  # already gone; nothing left to prune
    assert second.last_error is None


def test_t14_episode_retention_idempotent():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    factory = _FakeStoreFactory()
    from custom_components.thermosmart.learning.storage import naming
    key = naming.episodes_key(_ZONE_A)
    factory.inject(key, _FakeRawStore(initial=_envelope({
        "episodes": {"ep-old": {"end_ts": old_ts, "episode_type": "heating"}}
    })))
    registry = _episode_registry_with_policies(
        heating_policy=RetentionPolicy(max_age_days=30),
        outcome_policy=RetentionPolicy(max_age_days=30),
    )
    clock = FakeClock(_NOW)

    first = asyncio.run(async_prune_episodes(factory, _ZONE_A, registry, clock=clock))
    second = asyncio.run(async_prune_episodes(factory, _ZONE_A, registry, clock=clock))
    assert first.pruned_count == 1
    assert second.pruned_count == 0


# ── T15: Retention does not run on every observe cycle ───────────────────────

def test_t15_not_wired_into_hot_observe_path():
    """retention_service is not imported by ha_integration.py at all — it is
    an explicit, standalone maintenance call, not an automatic per-cycle hook.
    This deliberately keeps the (currently dormant) raw/episode capture
    subsystem's retention decoupled from the live shadow-controller cycle
    until that subsystem is actually wired up (a separate, future step)."""
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha)
    assert "retention_service" not in source
    assert "async_prune_" not in source


# ── T16: No Heating-Control path is touched ──────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
    "setpoint",
)


def test_t16_no_control_keywords_in_retention_modules():
    from custom_components.thermosmart.learning.storage import retention as _retention_mod
    for mod in (_retention_mod, _retention_service):
        source = inspect.getsource(mod).lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in source, f"forbidden control token found in {mod.__name__}: {token}"


# ── T17: Existing tests remain green (regression smoke-test) ────────────────

def test_t17_regression_imports_ok():
    import tests.test_application_lifecycle_storage  # noqa: F401
    import tests.test_application_lifecycle_state  # noqa: F401
    import tests.test_adaptation_monitoring  # noqa: F401
    import tests.test_preheat_application_plan  # noqa: F401
    import tests.test_application_readiness  # noqa: F401
    import tests.test_promotion_readiness  # noqa: F401
    import tests.test_application_orchestrator  # noqa: F401
    import tests.test_orchestration_export_trace  # noqa: F401


# ── Zone-level aggregate smoke test ───────────────────────────────────────────

def test_zone_level_aggregate_prunes_raw_and_episodes():
    old_ts = _iso(datetime(2020, 1, 1, tzinfo=timezone.utc))
    factory = _FakeStoreFactory()
    factory.inject(_index_key(_ZONE_A, RawTrackName.ROOM), _FakeRawStore(initial=_envelope({
        "track_name": "room", "track_schema_version": 1,
        "active_segment_id": None, "sealed_segment_ids": ["seg-0"],
        "next_sequence_number": 1,
    })))
    factory.inject(
        _segment_key(_ZONE_A, RawTrackName.ROOM, 0),
        _FakeRawStore(initial=_envelope(_seal_segment_payload(0, old_ts, "seg-0"))),
    )
    from custom_components.thermosmart.learning.storage import naming
    factory.inject(naming.episodes_key(_ZONE_A), _FakeRawStore(initial=_envelope({
        "episodes": {"ep-old": {"end_ts": old_ts, "episode_type": "heating"}}
    })))

    raw_registry = RawTrackRegistry()
    raw_registry.register(RawTrackDefinition(
        track_name=RawTrackName.ROOM, schema_type=RoomObservation,
        track_schema_version=1, privacy_class=PrivacyClass.LOW,
        allowed_export_scopes=(ExportScope.SUPPORT, ExportScope.RESEARCH),
        retention=RetentionPolicy(max_age_days=30),
        segmented=True, immutable_events=False,
    ))
    episode_registry = _episode_registry_with_policies(
        heating_policy=RetentionPolicy(max_age_days=30),
        outcome_policy=RetentionPolicy(max_age_days=30),
    )
    clock = FakeClock(_NOW)

    result = asyncio.run(async_prune_zone_storage(
        factory, _ZONE_A, raw_registry, episode_registry, clock=clock,
    ))
    assert result.raw_segments_pruned == 1
    assert result.episodes_pruned == 1
    assert result.last_error is None
    assert result.to_dict()["raw_segments_pruned"] == 1
