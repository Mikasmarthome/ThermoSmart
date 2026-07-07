"""Phase 3 tests for deterministic raw-segment recovery."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.thermosmart.learning.raw_schemas import RawTrackName, RoomObservation
from custom_components.thermosmart.learning.storage.segments import SegmentIndex, SegmentMetadata
from custom_components.thermosmart.learning.storage.serialization import decode_raw, encode_raw
from custom_components.thermosmart.learning.storage.recovery import (
    RecoveryCode,
    detect_orphans_and_dangling,
    partition_active_records,
    reconstruct_index_from_metadata,
    validate_sealed_segment,
)

UTC = timezone.utc
NOW = datetime(2026, 1, 15, 20, 0, 0, tzinfo=UTC)


def meta(seg_id, seq, sealed=True, count=1, content_hash=None):
    return SegmentMetadata(
        segment_id=seg_id, learning_zone_id="lz_1", track_name=RawTrackName.ROOM,
        track_schema_version=1, sequence_number=seq, created_utc=NOW,
        record_count=count, sealed_utc=NOW if sealed else None, content_hash=content_hash,
    )


class TestReconstructIndex:
    def test_missing_index_rebuilt_from_metadata(self):
        metas = [meta("s0", 0), meta("s1", 1), meta("s2", 2, sealed=False)]
        report = reconstruct_index_from_metadata("lz_1", RawTrackName.ROOM, 1, metas)
        assert report.recoverable
        idx = report.reconstructed_index
        assert idx.active_segment_id == "s2"
        assert idx.sealed_segment_ids == ("s0", "s1")
        assert idx.next_sequence_number == 3

    def test_deterministic_ordering(self):
        metas = [meta("s2", 2), meta("s0", 0), meta("s1", 1)]
        report = reconstruct_index_from_metadata("lz_1", RawTrackName.ROOM, 1, metas)
        assert report.reconstructed_index.sealed_segment_ids == ("s0", "s1", "s2")

    def test_all_sealed_no_active(self):
        report = reconstruct_index_from_metadata("lz_1", RawTrackName.ROOM, 1,
                                                 [meta("s0", 0), meta("s1", 1)])
        assert report.reconstructed_index.active_segment_id is None

    def test_multiple_active_is_recovery_error(self):
        metas = [meta("s0", 0, sealed=False), meta("s1", 1, sealed=False)]
        report = reconstruct_index_from_metadata("lz_1", RawTrackName.ROOM, 1, metas)
        assert not report.recoverable
        assert any(i.code is RecoveryCode.MULTIPLE_ACTIVE_SEGMENTS for i in report.issues)
        # no data silently chosen / overwritten
        assert report.reconstructed_index is None


class TestOrphansAndDangling:
    def test_orphan_segment_detected(self):
        idx = SegmentIndex(learning_zone_id="lz_1", track_name=RawTrackName.ROOM,
                           track_schema_version=1).with_sealed("s0")
        issues = detect_orphans_and_dangling(idx, ["s0", "s_orphan"])
        assert any(i.code is RecoveryCode.ORPHAN_SEGMENT for i in issues)

    def test_dangling_reference_detected(self):
        idx = SegmentIndex(learning_zone_id="lz_1", track_name=RawTrackName.ROOM,
                           track_schema_version=1).with_sealed("s0")
        issues = detect_orphans_and_dangling(idx, [])  # s0 missing on disk
        assert any(i.code is RecoveryCode.DANGLING_INDEX_REFERENCE for i in issues)


class TestSealedSegmentValidation:
    def _record_payload(self):
        return encode_raw(RawTrackName.ROOM, RoomObservation(
            learning_zone_id="lz_1", ts=NOW, indoor_temp=20.0, target=21.0))

    def test_valid_sealed_segment(self):
        from custom_components.thermosmart.learning.storage.segments import _content_hash
        records = (self._record_payload(),)
        m = meta("s0", 0, count=1, content_hash=_content_hash(records))
        assert validate_sealed_segment(m, list(records)) == ()

    def test_count_mismatch_detected(self):
        m = meta("s0", 0, count=5)
        issues = validate_sealed_segment(m, [self._record_payload()])
        assert any(i.code is RecoveryCode.CORRUPTED_SEALED_SEGMENT for i in issues)

    def test_hash_mismatch_detected(self):
        m = meta("s0", 0, count=1, content_hash="sha256:deadbeef")
        issues = validate_sealed_segment(m, [self._record_payload()])
        assert any(i.code is RecoveryCode.CORRUPTED_SEALED_SEGMENT for i in issues)


class TestPartitionActiveRecords:
    def test_keeps_valid_flags_invalid(self):
        good = encode_raw(RawTrackName.ROOM, RoomObservation(
            learning_zone_id="lz_1", ts=NOW, indoor_temp=20.0, target=21.0))
        bad = {"learning_zone_id": "lz_1"}  # missing fields -> undecodable

        def decode(d):
            return decode_raw(RawTrackName.ROOM, d)

        valid, issues = partition_active_records([good, bad], decode)
        assert len(valid) == 1
        assert any(i.code is RecoveryCode.INVALID_ACTIVE_RECORD for i in issues)
