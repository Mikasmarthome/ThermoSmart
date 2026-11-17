"""Outdoor-temperature evidence for Learning episodes (pure Python, no HA).

Phase D2-A. This module captures *trustworthy* outdoor-temperature evidence over
the life of a heating or passive-cooling episode and exposes it as an immutable,
optional sidecar (:class:`OutdoorEvidence`). It is STRICTLY observational:

* It never changes any model prediction, model update, regime, TPI, Preheat,
  Boost or TRV output. Nothing in the control or learning path reads it (D2-A).
* It never treats missing, invalid, implausible or corrupted data as valid
  evidence — only a present, plausible value with a usable :class:`DataQuality`
  contributes to the aggregates. Everything else is counted toward the sample
  total (so coverage is honest) and otherwise ignored.

Freshness policy
----------------
Home Assistant state timestamps cannot establish outdoor-reading freshness
reliably at this layer: the weather/sensor reading reaches the learning runtime
as a plain value with no ``last_updated``/``last_changed`` attached, and even
where a timestamp exists, ``last_changed`` only moves when the *value* changes —
a stable-but-fresh reading (e.g. a steady 5.0 °C) would be wrongly judged stale.
We therefore do NOT invent a freshness certainty: :attr:`OutdoorEvidence.freshness_known`
is always ``False`` and quality is derived only from what is reliably known
(presence, plausibility and the measurement's own :class:`DataQuality`). A caller
that later gains real freshness metadata can tighten this without reinterpreting
old evidence.

Source policy
-------------
A per-sample source identity (e.g. ``"sensor"`` vs ``"weather_entity"``) is
tracked when provided. If the source changes within one episode,
:attr:`OutdoorEvidence.source_changed` is set and :attr:`OutdoorEvidence.source`
is left ``None`` — the aggregate is still reported but the change is explicit, so
a consumer never silently combines incompatible sources. When no source identity
is provided the evidence simply records ``source=None`` and ``source_changed=False``
(an explicit unknown, not a fabricated identity).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .contracts import DataQuality, Measurement

OUTDOOR_EVIDENCE_SCHEMA_VERSION = 1

# Qualities whose value may be used as outdoor evidence. A STALE reading is still
# a real (if older) observation; UNAVAILABLE / INVALID / UNKNOWN never are.
_USABLE_QUALITIES = (DataQuality.OK, DataQuality.STALE)

# Reuse the integration-wide plausibility band without importing Home Assistant.
# Kept in sync intentionally with temperature_units.PLAUSIBLE_TEMPERATURE_* ; a
# pure-Python copy avoids pulling the HA-dependent module into the learning core.
_PLAUSIBLE_MIN_C = -30.0
_PLAUSIBLE_MAX_C = 50.0


def _is_plausible(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and _PLAUSIBLE_MIN_C <= v <= _PLAUSIBLE_MAX_C


@dataclass(frozen=True)
class OutdoorEvidence:
    """Immutable summary of the outdoor temperature over one episode.

    All temperature fields are ``None`` when the episode produced no usable
    outdoor sample. ``coverage`` is ``usable_count / sample_count`` (0.0 when no
    samples were seen at all).
    """

    time_weighted_mean_c: Optional[float]
    min_c: Optional[float]
    max_c: Optional[float]
    first_c: Optional[float]
    last_c: Optional[float]
    sample_count: int
    usable_count: int
    coverage: float
    duration_s: float
    quality: DataQuality
    freshness_known: bool
    source: Optional[str]
    source_changed: bool
    schema_version: int = OUTDOOR_EVIDENCE_SCHEMA_VERSION

    @property
    def has_usable_evidence(self) -> bool:
        return self.usable_count > 0 and self.time_weighted_mean_c is not None


class OutdoorEvidenceAccumulator:
    """Incremental, bounded accumulator of per-cycle outdoor readings.

    State is a fixed set of scalars (no growing sample list), so serialising it
    into a builder snapshot stays tiny and survives snapshot/restore mid-episode.
    Time-weighted mean uses a left-step integration over *usable* samples: each
    usable reading holds until the next usable reading (or the episode end).
    """

    def __init__(self) -> None:
        self._sample_count = 0
        self._usable_count = 0
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._first: Optional[float] = None
        self._last: Optional[float] = None
        self._any_stale = False
        self._sources: set[str] = set()
        # time-weighted-mean running state (left-step over usable samples)
        self._wsum = 0.0          # Σ weights closed out so far (ms)
        self._vwsum = 0.0         # Σ value * weight closed out so far
        self._simple_sum = 0.0    # Σ usable values (fallback mean)
        self._last_usable_offset: Optional[int] = None
        self._last_usable_value: Optional[float] = None

    # -- observation ----------------------------------------------------

    def observe(self, offset_ms: int, measurement: Optional[Measurement],
                source: Optional[str] = None) -> None:
        """Record one cycle's outdoor reading. Always counts toward the sample
        total; contributes to the value aggregates only when usable."""
        self._sample_count += 1
        if isinstance(source, str) and source:
            self._sources.add(source)

        value = getattr(measurement, "value", None) if measurement is not None else None
        quality = getattr(measurement, "quality", None) if measurement is not None else None
        usable = (
            measurement is not None
            and quality in _USABLE_QUALITIES
            and _is_plausible(value)
        )
        if not usable:
            return

        v = float(value)
        self._usable_count += 1
        self._simple_sum += v
        if quality is DataQuality.STALE:
            self._any_stale = True
        if self._first is None:
            self._first = v
        self._last = v
        self._min = v if self._min is None else min(self._min, v)
        self._max = v if self._max is None else max(self._max, v)

        # close out the previous usable sample's holding interval
        off = int(offset_ms)
        if self._last_usable_offset is not None and off > self._last_usable_offset:
            w = float(off - self._last_usable_offset)
            self._wsum += w
            self._vwsum += (self._last_usable_value or 0.0) * w
        self._last_usable_offset = off
        self._last_usable_value = v

    # -- finalisation ---------------------------------------------------

    def finalize(self, *, duration_s: float, end_offset_ms: Optional[int] = None,
                 source: Optional[str] = None) -> Optional[OutdoorEvidence]:
        """Produce the immutable summary, or ``None`` when no sample was seen.

        ``source`` is an optional last-cycle source hint used only as a tie-break
        when exactly one source was observed; a disagreement sets ``source_changed``.
        """
        if self._sample_count == 0:
            return None

        # close out the final usable holding interval up to the episode end
        wsum, vwsum = self._wsum, self._vwsum
        if self._last_usable_offset is not None and end_offset_ms is not None:
            tail = float(end_offset_ms) - float(self._last_usable_offset)
            if tail > 0.0:
                wsum += tail
                vwsum += (self._last_usable_value or 0.0) * tail

        if self._usable_count == 0:
            mean = None
        elif wsum > 0.0:
            mean = round(vwsum / wsum, 4)
        else:
            # single usable sample, or zero-span: honest simple mean
            mean = round(self._simple_sum / self._usable_count, 4)

        known_sources = set(self._sources)
        if isinstance(source, str) and source:
            known_sources.add(source)
        source_changed = len(known_sources) > 1
        resolved_source = (next(iter(known_sources)) if len(known_sources) == 1 else None)

        if self._usable_count == 0:
            quality = DataQuality.UNKNOWN
        elif self._any_stale:
            quality = DataQuality.STALE
        else:
            quality = DataQuality.OK

        coverage = round(self._usable_count / self._sample_count, 4) if self._sample_count else 0.0
        return OutdoorEvidence(
            time_weighted_mean_c=mean,
            min_c=(round(self._min, 4) if self._min is not None else None),
            max_c=(round(self._max, 4) if self._max is not None else None),
            first_c=(round(self._first, 4) if self._first is not None else None),
            last_c=(round(self._last, 4) if self._last is not None else None),
            sample_count=self._sample_count,
            usable_count=self._usable_count,
            coverage=coverage,
            duration_s=round(max(0.0, float(duration_s)), 3),
            quality=quality,
            freshness_known=False,
            source=resolved_source,
            source_changed=source_changed,
        )

    # -- snapshot / restore ---------------------------------------------

    def serialize(self) -> dict:
        return {
            "v": OUTDOOR_EVIDENCE_SCHEMA_VERSION,
            "n": self._sample_count,
            "u": self._usable_count,
            "min": self._min,
            "max": self._max,
            "first": self._first,
            "last": self._last,
            "stale": self._any_stale,
            "sources": sorted(self._sources),
            "wsum": self._wsum,
            "vwsum": self._vwsum,
            "ssum": self._simple_sum,
            "luo": self._last_usable_offset,
            "luv": self._last_usable_value,
        }

    @classmethod
    def restore(cls, data: object) -> "OutdoorEvidenceAccumulator":
        """Rebuild defensively from a snapshot. Any corruption yields a fresh,
        empty accumulator rather than raising — a lost outdoor summary must never
        break episode restore or learning."""
        acc = cls()
        if not isinstance(data, dict):
            return acc
        try:
            acc._sample_count = max(0, int(data.get("n", 0) or 0))
            acc._usable_count = max(0, int(data.get("u", 0) or 0))
            acc._min = _finite_or_none(data.get("min"))
            acc._max = _finite_or_none(data.get("max"))
            acc._first = _finite_or_none(data.get("first"))
            acc._last = _finite_or_none(data.get("last"))
            acc._any_stale = bool(data.get("stale", False))
            srcs = data.get("sources", [])
            acc._sources = {str(s) for s in srcs if isinstance(s, str) and s} \
                if isinstance(srcs, (list, tuple, set)) else set()
            acc._wsum = _finite_float(data.get("wsum"), 0.0)
            acc._vwsum = _finite_float(data.get("vwsum"), 0.0)
            acc._simple_sum = _finite_float(data.get("ssum"), 0.0)
            luo = data.get("luo")
            acc._last_usable_offset = int(luo) if isinstance(luo, (int, float)) \
                and math.isfinite(luo) else None
            acc._last_usable_value = _finite_or_none(data.get("luv"))
            # defensive invariants: usable count cannot exceed total; negative sums reset
            if acc._usable_count > acc._sample_count:
                return cls()
            if acc._wsum < 0.0 or acc._vwsum < 0.0 and acc._wsum == 0.0:
                acc._wsum = max(0.0, acc._wsum)
        except (TypeError, ValueError):
            return cls()
        return acc


def _finite_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _finite_float(value, default: float) -> float:
    f = _finite_or_none(value)
    return f if f is not None else default
