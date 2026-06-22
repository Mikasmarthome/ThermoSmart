"""Deterministic recovery for raw segment storage (pure Python).

Recovery never silently picks a winner or deletes data. It rebuilds an index
from segment metadata, isolates corrupted/orphaned segments, and reports a
structured :class:`RecoveryReport`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

from ..raw_schemas import RawTrackName
from .segments import SegmentIndex, SegmentMetadata, _content_hash


class RecoveryCode(Enum):
    MISSING_INDEX = "missing_index"
    MULTIPLE_ACTIVE_SEGMENTS = "multiple_active_segments"
    CORRUPTED_SEALED_SEGMENT = "corrupted_sealed_segment"
    INVALID_ACTIVE_RECORD = "invalid_active_record"
    ORPHAN_SEGMENT = "orphan_segment"
    DANGLING_INDEX_REFERENCE = "dangling_index_reference"


@dataclass(frozen=True)
class RecoveryIssue:
    code: RecoveryCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryReport:
    reconstructed_index: Optional[SegmentIndex]
    issues: tuple[RecoveryIssue, ...] = ()

    @property
    def recoverable(self) -> bool:
        return self.reconstructed_index is not None


def _sorted(metadatas: Sequence[SegmentMetadata]) -> list[SegmentMetadata]:
    # Deterministic order: by sequence, then segment id.
    return sorted(metadatas, key=lambda m: (m.sequence_number, m.segment_id))


def reconstruct_index_from_metadata(
    learning_zone_id: str,
    track_name: RawTrackName,
    track_schema_version: int,
    metadatas: Sequence[SegmentMetadata],
) -> RecoveryReport:
    """Rebuild a :class:`SegmentIndex` from available segment metadata."""
    ordered = _sorted(metadatas)
    issues: list[RecoveryIssue] = []

    active = [m for m in ordered if not m.is_sealed]
    sealed = [m for m in ordered if m.is_sealed]

    if len(active) > 1:
        # Do not silently choose one — surface the conflict for safe handling.
        issues.append(RecoveryIssue(
            RecoveryCode.MULTIPLE_ACTIVE_SEGMENTS,
            "multiple active segments found; manual recovery required",
            {"candidates": [m.segment_id for m in active]},
        ))
        return RecoveryReport(reconstructed_index=None, issues=tuple(issues))

    highest = max((m.sequence_number for m in ordered), default=-1)
    index = SegmentIndex(
        learning_zone_id=learning_zone_id,
        track_name=track_name,
        track_schema_version=track_schema_version,
        next_sequence_number=highest + 1,
        active_segment_id=active[0].segment_id if active else None,
        sealed_segment_ids=tuple(m.segment_id for m in sealed),
    )
    return RecoveryReport(reconstructed_index=index, issues=tuple(issues))


def detect_orphans_and_dangling(
    index: SegmentIndex,
    available_segment_ids: Sequence[str],
) -> tuple[RecoveryIssue, ...]:
    """Find segments on disk not in the index, and index refs with no segment."""
    available = set(available_segment_ids)
    known = index.known_segment_ids
    issues: list[RecoveryIssue] = []
    for seg_id in sorted(available - known):
        issues.append(RecoveryIssue(
            RecoveryCode.ORPHAN_SEGMENT,
            f"segment '{seg_id}' is not referenced by the index",
            {"segment_id": seg_id},
        ))
    for seg_id in sorted(known - available):
        issues.append(RecoveryIssue(
            RecoveryCode.DANGLING_INDEX_REFERENCE,
            f"index references missing segment '{seg_id}'",
            {"segment_id": seg_id},
        ))
    return tuple(issues)


def validate_sealed_segment(
    metadata: SegmentMetadata,
    encoded_records: Sequence[dict],
) -> tuple[RecoveryIssue, ...]:
    """Detect hash / count mismatch in a sealed segment."""
    issues: list[RecoveryIssue] = []
    if metadata.record_count != len(encoded_records):
        issues.append(RecoveryIssue(
            RecoveryCode.CORRUPTED_SEALED_SEGMENT,
            f"segment '{metadata.segment_id}' record count mismatch",
            {"segment_id": metadata.segment_id,
             "expected": metadata.record_count, "found": len(encoded_records)},
        ))
    if metadata.content_hash is not None:
        actual = _content_hash(tuple(encoded_records))
        if actual != metadata.content_hash:
            issues.append(RecoveryIssue(
                RecoveryCode.CORRUPTED_SEALED_SEGMENT,
                f"segment '{metadata.segment_id}' content hash mismatch",
                {"segment_id": metadata.segment_id},
            ))
    return tuple(issues)


def partition_active_records(
    encoded_records: Sequence[dict],
    decode: Callable[[dict], Any],
) -> tuple[list[Any], list[RecoveryIssue]]:
    """Keep decodable records from an incomplete active segment; flag the rest.

    The active segment is replaceable, so invalid trailing records are reported
    rather than aborting recovery. Sealed segments are never touched here.
    """
    valid: list[Any] = []
    issues: list[RecoveryIssue] = []
    for position, raw in enumerate(encoded_records):
        try:
            valid.append(decode(raw))
        except Exception as err:  # noqa: BLE001 - recovery must be defensive
            issues.append(RecoveryIssue(
                RecoveryCode.INVALID_ACTIVE_RECORD,
                f"active record at position {position} is invalid",
                {"position": position, "error": str(err)},
            ))
    return valid, issues
