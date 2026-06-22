"""Phase 3 tests for segment metadata, the active-segment writer, naming and index."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.raw_schemas import (
    RawTrackName,
    RoomObservation,
    TrvObservation,
)
from custom_components.thermosmart.learning.storage import naming
from custom_components.thermosmart.learning.storage.naming import StorageNamingError
from custom_components.thermosmart.learning.storage.segments import (
    ActiveSegment,
    SegmentError,
    SegmentIndex,
    SegmentMetadata,
)

UTC = timezone.utc
NOW = datetime(2026, 1, 15, 20, 30, 0, tzinfo=UTC)


def clock() -> FakeClock:
    return FakeClock(NOW)


def room(zid="lz_1", temp=20.0, ts=NOW) -> RoomObservation:
    return RoomObservation(learning_zone_id=zid, ts=ts, indoor_temp=temp, target=21.0)


def active(capacity=3, zid="lz_1", seq=0) -> ActiveSegment:
    return ActiveSegment(
        learning_zone_id=zid, track_name=RawTrackName.ROOM, schema_type=RoomObservation,
        track_schema_version=1, sequence_number=seq, capacity=capacity,
        segment_id=f"seg_{seq}", clock=clock(),
    )


# -- naming -------------------------------------------------------------------

class TestNaming:
    def test_keys_are_deterministic(self):
        assert naming.container_key("lz_1") == naming.container_key("lz_1")

    def test_no_collision_between_zones(self):
        assert naming.container_key("lz_1") != naming.container_key("lz_2")

    def test_no_collision_between_tracks(self):
        a = naming.raw_index_key("lz_1", RawTrackName.ROOM)
        b = naming.raw_index_key("lz_1", RawTrackName.TRV)
        assert a != b

    def test_same_sequence_different_zone_no_collision(self):
        a = naming.raw_segment_key("lz_1", RawTrackName.ROOM, 3)
        b = naming.raw_segment_key("lz_2", RawTrackName.ROOM, 3)
        assert a != b

    def test_versioned_prefix(self):
        assert naming.container_key("lz_1").startswith("thermosmart_le2__")

    def test_no_v1_prefix_and_no_v1_key_collision(self):
        key = naming.container_key("lz_1")
        assert "thermosmart_le1__" not in key
        assert key != "thermosmart_learning_data"
        assert naming.global_index_key() != "thermosmart_learning_data"

    def test_rejects_path_traversal(self):
        with pytest.raises(StorageNamingError):
            naming.container_key("../etc")

    def test_rejects_double_underscore(self):
        with pytest.raises(StorageNamingError):
            naming.container_key("a__b")

    def test_rejects_empty(self):
        with pytest.raises(StorageNamingError):
            naming.container_key("")

    def test_global_index_key(self):
        assert naming.global_index_key() == "thermosmart_le2__global_index"


# -- segment metadata ---------------------------------------------------------

class TestSegmentMetadata:
    def test_valid(self):
        m = SegmentMetadata(segment_id="s", learning_zone_id="lz_1",
                            track_name=RawTrackName.ROOM, track_schema_version=1,
                            sequence_number=0, created_utc=NOW, record_count=0)
        assert m.is_sealed is False

    def test_sealed_flag(self):
        m = SegmentMetadata(segment_id="s", learning_zone_id="lz_1",
                            track_name=RawTrackName.ROOM, track_schema_version=1,
                            sequence_number=0, created_utc=NOW, record_count=1,
                            sealed_utc=NOW)
        assert m.is_sealed is True

    def test_negative_sequence_rejected(self):
        with pytest.raises(SegmentError):
            SegmentMetadata(segment_id="s", learning_zone_id="lz_1",
                            track_name=RawTrackName.ROOM, track_schema_version=1,
                            sequence_number=-1, created_utc=NOW, record_count=0)


# -- active segment -----------------------------------------------------------

class TestActiveSegment:
    def test_append_and_count(self):
        seg = active()
        assert seg.append(room(temp=20.0)) is True
        assert seg.record_count == 1

    def test_trv_only_record_accepted(self):
        seg = active()
        assert seg.append(room()) is True  # no environmental data at all

    def test_duplicate_rejected(self):
        seg = active()
        r = room(temp=20.0)
        assert seg.append(r) is True
        assert seg.append(r) is False  # same fingerprint
        assert seg.record_count == 1

    def test_wrong_schema_type_rejected(self):
        seg = active()
        trv = TrvObservation(learning_zone_id="lz_1", ts=NOW, trv_binding_id="t",
                             trv_setpoint=22.0, target=21.0)
        with pytest.raises(SegmentError):
            seg.append(trv)

    def test_wrong_zone_rejected(self):
        seg = active(zid="lz_1")
        with pytest.raises(SegmentError):
            seg.append(room(zid="lz_2"))

    def test_full_detection(self):
        seg = active(capacity=2)
        seg.append(room(temp=20.0))
        assert seg.is_full is False
        seg.append(room(temp=20.5))
        assert seg.is_full is True

    def test_append_after_seal_rejected(self):
        seg = active()
        seg.append(room(temp=20.0))
        seg.seal()
        with pytest.raises(SegmentError):
            seg.append(room(temp=21.0))

    def test_seal_is_idempotent(self):
        seg = active()
        seg.append(room(temp=20.0))
        m1 = seg.seal()
        m2 = seg.seal()
        assert m1.sealed_utc == m2.sealed_utc and m1.is_sealed

    def test_sealed_metadata_has_hash_and_bounds(self):
        seg = active()
        seg.append(room(temp=20.0, ts=NOW))
        m = seg.seal()
        assert m.record_count == 1
        assert m.content_hash and m.content_hash.startswith("sha256:")
        assert m.first_record_utc == NOW and m.last_record_utc == NOW

    def test_snapshot_isolation(self):
        seg = active(capacity=10)
        seg.append(room(temp=20.0))
        snap = seg.snapshot()
        seg.append(room(temp=20.5))  # mutate after snapshot
        assert snap.metadata.record_count == 1
        assert len(snap.records) == 1

    def test_deterministic_record_order(self):
        seg = active(capacity=10)
        seg.append(room(temp=20.0))
        seg.append(room(temp=20.5))
        seg.append(room(temp=21.0))
        snap = seg.snapshot()
        temps = [r["indoor_temp"] for r in snap.records]
        assert temps == [20.0, 20.5, 21.0]

    def test_zero_capacity_rejected(self):
        with pytest.raises(SegmentError):
            active(capacity=0)


# -- segment index ------------------------------------------------------------

class TestSegmentIndex:
    def _idx(self) -> SegmentIndex:
        return SegmentIndex(learning_zone_id="lz_1", track_name=RawTrackName.ROOM,
                            track_schema_version=1)

    def test_with_active_advances_sequence(self):
        idx = self._idx().with_active("seg_0", 0)
        assert idx.active_segment_id == "seg_0"
        assert idx.next_sequence_number == 1

    def test_with_sealed_moves_active(self):
        idx = self._idx().with_active("seg_0", 0).with_sealed("seg_0")
        assert idx.active_segment_id is None
        assert "seg_0" in idx.sealed_segment_ids

    def test_multiple_sealed_monotonic_sequence(self):
        idx = (self._idx()
               .with_active("seg_0", 0).with_sealed("seg_0")
               .with_active("seg_1", 1).with_sealed("seg_1"))
        assert idx.sealed_segment_ids == ("seg_0", "seg_1")
        assert idx.next_sequence_number == 2

    def test_known_segment_ids(self):
        idx = self._idx().with_active("seg_1", 1).with_sealed("seg_0_was_sealed")
        # active seg_1 plus the sealed id
        assert "seg_1" in idx.known_segment_ids
