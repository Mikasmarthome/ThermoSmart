"""Raw segment metadata, the active-segment writer, and the per-track index.

Pure Python. A raw track is stored as a rotating series of sealed (immutable)
segments plus exactly one active segment. The active segment is the only thing
re-written on a flush; sealed segments are never written again.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Optional

from ..contracts import require_utc
from ..raw_schemas import RawTrackName

SEGMENT_FORMAT_VERSION = 1


class SegmentError(Exception):
    """Raised on an invalid segment operation."""


# -- metadata -----------------------------------------------------------------

@dataclass(frozen=True)
class SegmentMetadata:
    segment_id: str
    learning_zone_id: str
    track_name: RawTrackName
    track_schema_version: int
    sequence_number: int
    created_utc: datetime
    record_count: int
    sealed_utc: Optional[datetime] = None
    first_record_utc: Optional[datetime] = None
    last_record_utc: Optional[datetime] = None
    content_hash: Optional[str] = None
    format_version: int = SEGMENT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise SegmentError("segment_id must be non-empty")
        if not self.learning_zone_id:
            raise SegmentError("learning_zone_id must be non-empty")
        if not isinstance(self.track_name, RawTrackName):
            raise SegmentError("track_name must be a RawTrackName")
        if self.sequence_number < 0:
            raise SegmentError("sequence_number must be >= 0")
        if self.record_count < 0:
            raise SegmentError("record_count must be >= 0")
        object.__setattr__(self, "created_utc", require_utc(self.created_utc, "created_utc"))
        if self.sealed_utc is not None:
            object.__setattr__(self, "sealed_utc", require_utc(self.sealed_utc, "sealed_utc"))
        if self.first_record_utc is not None:
            object.__setattr__(self, "first_record_utc",
                               require_utc(self.first_record_utc, "first_record_utc"))
        if self.last_record_utc is not None:
            object.__setattr__(self, "last_record_utc",
                               require_utc(self.last_record_utc, "last_record_utc"))

    @property
    def is_sealed(self) -> bool:
        return self.sealed_utc is not None


# -- fingerprints (duplicate / replay guard) ----------------------------------

def _fingerprint(track_name: RawTrackName, record: Any) -> tuple:
    # Immutable events are identified by their own id; observations by their
    # identifying measurement tuple. ``0.0`` values participate normally.
    for id_attr in ("event_id", "decision_id", "evaluation_id"):
        if hasattr(record, id_attr):
            return (track_name, id_attr, getattr(record, id_attr))
    if track_name is RawTrackName.TRV:
        return (track_name, record.ts, record.trv_binding_id,
                record.trv_setpoint, record.target)
    return (track_name, record.ts, getattr(record, "indoor_temp", None),
            getattr(record, "target", None))


def _content_hash(encoded_records: tuple[dict, ...]) -> str:
    payload = json.dumps(list(encoded_records), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -- snapshot -----------------------------------------------------------------

@dataclass(frozen=True)
class SegmentSnapshot:
    """An immutable point-in-time copy of a segment for persistence."""

    metadata: SegmentMetadata
    records: tuple[dict, ...]

    def to_payload(self) -> dict:
        meta = self.metadata
        return {
            "format_version": meta.format_version,
            "segment": {
                "segment_id": meta.segment_id,
                "learning_zone_id": meta.learning_zone_id,
                "track_name": meta.track_name.value,
                "track_schema_version": meta.track_schema_version,
                "sequence_number": meta.sequence_number,
                "created_utc": _enc_opt_dt(meta.created_utc),
                "sealed_utc": _enc_opt_dt(meta.sealed_utc),
                "record_count": meta.record_count,
                "first_record_utc": _enc_opt_dt(meta.first_record_utc),
                "last_record_utc": _enc_opt_dt(meta.last_record_utc),
                "content_hash": meta.content_hash,
            },
            "records": list(self.records),
        }


def _enc_opt_dt(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    from .serialization import encode_utc
    return encode_utc(value)


# -- active segment writer ----------------------------------------------------


# -- segment index ------------------------------------------------------------

@dataclass(frozen=True)
class SegmentIndex:
    """Reconstructable cache of a track's segment layout for one zone."""

    learning_zone_id: str
    track_name: RawTrackName
    track_schema_version: int
    next_sequence_number: int = 0
    active_segment_id: Optional[str] = None
    sealed_segment_ids: tuple[str, ...] = ()
    corrupted_segment_ids: tuple[str, ...] = ()
    last_flush_utc: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.learning_zone_id:
            raise SegmentError("learning_zone_id must be non-empty")
        if not isinstance(self.track_name, RawTrackName):
            raise SegmentError("track_name must be a RawTrackName")
        if self.next_sequence_number < 0:
            raise SegmentError("next_sequence_number must be >= 0")

    def with_active(self, segment_id: str, sequence_number: int) -> "SegmentIndex":
        return replace(
            self,
            active_segment_id=segment_id,
            next_sequence_number=max(self.next_sequence_number, sequence_number + 1),
        )

    def with_sealed(self, segment_id: str) -> "SegmentIndex":
        if segment_id in self.sealed_segment_ids:
            return self
        active = None if self.active_segment_id == segment_id else self.active_segment_id
        return replace(
            self,
            active_segment_id=active,
            sealed_segment_ids=self.sealed_segment_ids + (segment_id,),
        )

    def with_flush(self, when: datetime) -> "SegmentIndex":
        return replace(self, last_flush_utc=require_utc(when, "last_flush_utc"))

    def with_corrupted(self, segment_id: str) -> "SegmentIndex":
        if segment_id in self.corrupted_segment_ids:
            return self
        return replace(self, corrupted_segment_ids=self.corrupted_segment_ids + (segment_id,))

    @property
    def known_segment_ids(self) -> frozenset[str]:
        ids = set(self.sealed_segment_ids)
        if self.active_segment_id is not None:
            ids.add(self.active_segment_id)
        return frozenset(ids)
