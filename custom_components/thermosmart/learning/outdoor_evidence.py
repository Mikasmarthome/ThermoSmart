"""Outdoor-temperature evidence for Learning episodes (pure Python, no HA).

Phase D2-A / D2-A.1. This module captures *trustworthy* outdoor-temperature
evidence over the life of a heating or passive-cooling episode and exposes it as
an immutable, optional sidecar (:class:`OutdoorEvidence`). It is STRICTLY
observational:

* It never changes any model prediction, model update, regime, TPI, Preheat,
  Boost or TRV output. Nothing in the control or learning path reads it.
* It never treats missing, invalid, implausible or corrupted data as valid
  evidence — only a present, plausible value with a usable :class:`DataQuality`
  contributes to the aggregates. Everything else is counted toward the sample
  total (so sample coverage is honest) and otherwise ignored.

Sample coverage vs temporal coverage
------------------------------------
Two distinct coverage metrics are reported:

* :attr:`OutdoorEvidence.coverage` — *sample* coverage: usable samples / total
  samples observed. A per-reading fraction.
* :attr:`OutdoorEvidence.temporal_coverage` — *time* coverage: the fraction of
  the episode duration that is actually backed by a fresh usable reading. Each
  usable reading is credited at most ``_MAX_HOLD_MS`` of hold time (see the
  cadence policy below), so a long sensor dropout cannot be silently
  extrapolated as fully observed. :attr:`OutdoorEvidence.conditioning_ready`
  summarises whether the evidence is dense enough to be trusted by a future
  conditioned learner (D2-B); D2-A.1 only exposes the flag, it consumes nothing.

Cadence / interval policy
-------------------------
The runtime observes each zone on a ~5-minute cycle, so a single usable reading
is credited with holding for up to ``_MAX_HOLD_MS`` = two cadence intervals
(600 s). Two intervals (not one) tolerate a single skipped/late poll without
penalising a healthy sensor, while a genuine multi-cycle dropout leaves the
intervening time uncovered. A value that merely *did not change* across cycles
is NOT treated as stale: as long as a fresh reading arrives each cycle its time
is fully covered.

Freshness policy
----------------
Home Assistant state timestamps cannot establish outdoor-reading freshness
reliably at this layer (the value arrives without ``last_updated``/``last_changed``,
and ``last_changed`` only moves when the value changes). We therefore do NOT
invent a freshness certainty: :attr:`OutdoorEvidence.freshness_known` is always
``False`` and quality is derived only from presence, plausibility and the
measurement's own :class:`DataQuality`. Temporal coverage, not a per-reading age,
is what bounds over-extrapolation.

Source policy
-------------
A per-sample source identity (e.g. ``"sensor"`` vs ``"weather_entity"``) is
tracked only for USABLE samples. If the usable source changes within one episode,
:attr:`OutdoorEvidence.source_changed` is set and :attr:`OutdoorEvidence.source`
is left ``None`` — the aggregate is still reported but the change is explicit, so
a consumer never silently combines incompatible sources. An unusable reading
never affects the source identity. When no source is provided the evidence
records ``source=None`` / ``source_changed=False`` (an explicit unknown).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .contracts import DataQuality, Measurement

OUTDOOR_EVIDENCE_SCHEMA_VERSION = 2
# v1 = D2-A (no temporal-coverage state); v2 = D2-A.1. Both are restorable.
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})

# Qualities whose value may be used as outdoor evidence. A STALE reading is still
# a real (if older) observation; UNAVAILABLE / INVALID / UNKNOWN never are.
_USABLE_QUALITIES = (DataQuality.OK, DataQuality.STALE)

# Reuse the integration-wide plausibility band without importing Home Assistant.
# Kept intentionally in sync with temperature_units.PLAUSIBLE_TEMPERATURE_* ; a
# pure-Python copy avoids pulling the HA-dependent module into the learning core.
_PLAUSIBLE_MIN_C = -30.0
_PLAUSIBLE_MAX_C = 50.0

# Interval policy (see module docstring). The runtime cadence is ~5 min.
_CADENCE_MS = 300_000
_MAX_HOLD_MS = 2 * _CADENCE_MS            # 600 s: one skipped poll tolerated
# Minimum temporal coverage for the evidence to be a candidate for conditioned
# learning. Observational hint only (D2-A.1/D2-A.2 consume nothing).
_MIN_TEMPORAL_COVERAGE = 0.5
_MIN_USABLE_FOR_CONDITIONING = 2

# Floating-point tolerances for restore/finalize consistency checks. JSON
# round-trips float64 exactly, so these only absorb accumulation rounding: a
# small absolute floor plus a relative term that scales with the magnitude of
# the (millisecond-scale) weight sums being compared — never a broad allowance.
_ABS_TOL = 1e-6
_REL_TOL = 1e-9


def _tol(*magnitudes: float) -> float:
    return _ABS_TOL + _REL_TOL * max((abs(m) for m in magnitudes), default=0.0)


def _in_range(value: float, lo: float, hi: float) -> bool:
    t = _tol(lo, hi, value)
    return (lo - t) <= value <= (hi + t)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _tol(a, b)


def _is_number(value) -> bool:
    """A real, non-bool int/float. ``bool`` is explicitly excluded so a persisted
    or live ``True``/``False`` can never read as 1.0/0.0 °C."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_plausible(value) -> bool:
    if not _is_number(value):
        return False
    v = float(value)
    return math.isfinite(v) and _PLAUSIBLE_MIN_C <= v <= _PLAUSIBLE_MAX_C


@dataclass(frozen=True)
class OutdoorEvidence:
    """Immutable summary of the outdoor temperature over one episode.

    All temperature fields are ``None`` when the episode produced no usable
    outdoor sample. ``coverage`` is SAMPLE coverage (usable / total samples);
    ``temporal_coverage`` is TIME coverage (fraction of the duration backed by a
    fresh reading). ``conditioning_ready`` is an observational suitability hint.
    """

    time_weighted_mean_c: Optional[float]
    min_c: Optional[float]
    max_c: Optional[float]
    first_c: Optional[float]
    last_c: Optional[float]
    sample_count: int
    usable_count: int
    coverage: float
    temporal_coverage: float
    conditioning_ready: bool
    duration_s: float
    quality: DataQuality
    freshness_known: bool
    source: Optional[str]
    source_changed: bool
    schema_version: int = OUTDOOR_EVIDENCE_SCHEMA_VERSION

    @property
    def has_usable_evidence(self) -> bool:
        return self.usable_count > 0 and self.time_weighted_mean_c is not None

    @property
    def sample_coverage(self) -> float:
        """Alias for :attr:`coverage` (usable / total samples)."""
        return self.coverage


class OutdoorEvidenceAccumulator:
    """Incremental, bounded accumulator of per-cycle outdoor readings.

    State is a fixed set of scalars (no growing sample list), so serialising it
    into a builder snapshot stays tiny and survives snapshot/restore mid-episode.
    Time-weighted mean uses a left-step integration over *usable* samples, with
    each held interval capped at ``_MAX_HOLD_MS``; an unusable reading suspends
    the previous valid interval so a dropout is not extrapolated.
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
        # time-weighted-mean + temporal-coverage running state
        self._wsum = 0.0            # Σ capped weights closed out (ms)
        self._vwsum = 0.0           # Σ value * capped weight closed out
        self._simple_sum = 0.0      # Σ usable values (fallback mean)
        self._covered_ms = 0.0      # Σ capped hold time (temporal coverage, ms)
        self._last_usable_offset: Optional[int] = None
        self._last_usable_value: Optional[float] = None
        self._last_offset: Optional[int] = None   # monotonic / dedup guard

    # -- observation ----------------------------------------------------

    def observe(self, offset_ms: int, measurement: Optional[Measurement],
                source: Optional[str] = None) -> None:
        """Record one cycle's outdoor reading. Duplicate / out-of-order offsets are
        ignored (not counted). Always counts toward the sample total when accepted;
        contributes to the value aggregates and temporal coverage only when usable."""
        if not _is_number(offset_ms) or not math.isfinite(offset_ms):
            return
        off = int(offset_ms)
        if off < 0:
            return
        # strictly-increasing guard: drop duplicate / out-of-order timestamps so a
        # replayed or reordered cycle can never create a negative or double-counted
        # interval.
        if self._last_offset is not None and off <= self._last_offset:
            return
        self._last_offset = off
        self._sample_count += 1

        # close out the previous usable sample's holding interval, capped so a gap
        # cannot extrapolate the last value across unobserved time.
        if self._last_usable_offset is not None:
            held = min(float(off - self._last_usable_offset), float(_MAX_HOLD_MS))
            if held > 0.0:
                self._covered_ms += held
                self._wsum += held
                self._vwsum += (self._last_usable_value or 0.0) * held

        value = getattr(measurement, "value", None) if measurement is not None else None
        quality = getattr(measurement, "quality", None) if measurement is not None else None
        usable = (
            measurement is not None
            and quality in _USABLE_QUALITIES
            and _is_plausible(value)
        )
        if not usable:
            # an unusable reading suspends the previous valid interval
            self._last_usable_offset = None
            self._last_usable_value = None
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
        if isinstance(source, str) and source:
            self._sources.add(source)
        self._last_usable_offset = off
        self._last_usable_value = v

    # -- finalisation ---------------------------------------------------

    def finalize(self, *, duration_s: float, end_offset_ms: Optional[int] = None,
                 source: Optional[str] = None) -> Optional[OutdoorEvidence]:
        """Produce the immutable summary, or ``None`` when no sample was seen."""
        if self._sample_count == 0:
            return None

        wsum, vwsum, covered = self._wsum, self._vwsum, self._covered_ms
        if self._last_usable_offset is not None and end_offset_ms is not None \
                and _is_number(end_offset_ms) and math.isfinite(end_offset_ms):
            tail = min(float(end_offset_ms) - float(self._last_usable_offset),
                       float(_MAX_HOLD_MS))
            if tail > 0.0:
                wsum += tail
                vwsum += (self._last_usable_value or 0.0) * tail
                covered += tail

        # Overflow-safe mean. On any arithmetic error, or a non-finite / out-of-
        # [min,max] result, the mean fails closed to None (unusable), never to a
        # fabricated value.
        try:
            if self._usable_count == 0:
                mean = None
            elif wsum > 0.0:
                mean = vwsum / wsum
            else:
                mean = self._simple_sum / self._usable_count
            if mean is not None and not math.isfinite(mean):
                mean = None
        except (OverflowError, ZeroDivisionError, ValueError):
            mean = None
        if mean is not None and self._min is not None and self._max is not None \
                and not _in_range(mean, self._min, self._max):
            mean = None                                  # finalized mean-range validation
        if mean is not None:
            mean = round(mean, 4)

        # a resolved source hint from the final cycle participates only when it is
        # a usable source we already recorded; it never fabricates a new identity.
        known_sources = set(self._sources)
        source_changed = len(known_sources) > 1
        resolved_source = next(iter(known_sources)) if len(known_sources) == 1 else None

        if self._usable_count == 0:
            quality = DataQuality.UNKNOWN
        elif self._any_stale:
            quality = DataQuality.STALE
        else:
            quality = DataQuality.OK

        duration_s = float(duration_s)
        duration_s = max(0.0, duration_s) if math.isfinite(duration_s) else 0.0
        duration_ms = duration_s * 1000.0
        sample_cov = round(self._usable_count / self._sample_count, 4) if self._sample_count else 0.0
        try:
            temporal_cov = round(min(1.0, covered / duration_ms), 4) if duration_ms > 0.0 else 0.0
        except (OverflowError, ZeroDivisionError, ValueError):
            temporal_cov = 0.0
        # Weights that exceed the episode duration (e.g. a legacy v1 uncapped
        # snapshot restored and finalized) mean the temporal basis is not
        # verifiable against this episode — do not treat it as trustworthy
        # conditioned evidence even though temporal_cov is clamped to 1.0.
        coverage_trustworthy = duration_ms > 0.0 and covered <= duration_ms + _tol(covered, duration_ms)
        conditioning_ready = (
            self._usable_count >= _MIN_USABLE_FOR_CONDITIONING
            and mean is not None
            and temporal_cov >= _MIN_TEMPORAL_COVERAGE
            and coverage_trustworthy
            and not source_changed
        )
        return OutdoorEvidence(
            time_weighted_mean_c=mean,
            min_c=(round(self._min, 4) if self._min is not None else None),
            max_c=(round(self._max, 4) if self._max is not None else None),
            first_c=(round(self._first, 4) if self._first is not None else None),
            last_c=(round(self._last, 4) if self._last is not None else None),
            sample_count=self._sample_count,
            usable_count=self._usable_count,
            coverage=sample_cov,
            temporal_coverage=temporal_cov,
            conditioning_ready=conditioning_ready,
            duration_s=round(duration_s, 3),
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
            "cov": self._covered_ms,
            "luo": self._last_usable_offset,
            "luv": self._last_usable_value,
            "lo": self._last_offset,
        }

    @classmethod
    def restore(cls, data: object) -> "OutdoorEvidenceAccumulator":
        """Rebuild defensively from a snapshot. ANY corruption yields a fresh,
        empty accumulator (unusable evidence) rather than a fabricated measurement
        and never raises — a lost outdoor summary must not break episode restore.

        Rejected (→ empty): non-dict; unsupported/absent/inconsistent schema
        version (v2 missing its ``cov``/``lo`` state); a missing or malformed
        required numeric field; NaN / Infinity; a bool where a number is expected;
        a negative count/weight/coverage/offset; usable > total; a usable episode
        missing min/max/first/last; any persisted temperature outside the
        plausibility contract; min > max or a first/last/last-value/reconstructed
        mean outside [min, max]; ``wsum == 0`` with a non-zero ``vwsum``; a v2
        ``cov`` that disagrees with ``wsum`` (the serializer increments them in
        lockstep); and inconsistent source metadata. OverflowError (and
        ValueError/TypeError/ZeroDivisionError) during conversion is caught.

        Duration is NOT known at restore, so offsets/weights are validated only
        for finiteness, sign and internal ordering here; their consistency with
        the actual episode duration is enforced in :meth:`finalize`
        (``coverage_trustworthy``), and unverifiable state is never surfaced as
        conditioning-ready.
        """
        acc = cls()
        if not isinstance(data, dict):
            return acc
        try:
            version = data.get("v")
            if version not in _SUPPORTED_SCHEMA_VERSIONS:
                return cls()
            # v2 must carry its explicit temporal-coverage state; its absence is an
            # inconsistent/forged v2 record.
            if version == 2 and ("cov" not in data or "lo" not in data):
                return cls()

            n = _req_int(data, "n")
            u = _req_int(data, "u")
            wsum = _req_float(data, "wsum")
            vwsum = _req_float(data, "vwsum")
            ssum = _req_float(data, "ssum")
            # v1 (D2-A) lacks explicit temporal-coverage state: fall back to the
            # recorded weight sum as a best-effort covered time, and to the last
            # usable offset for the ordering guard.
            covered = _opt_float(data, "cov") if "cov" in data else wsum
            if covered is _INVALID or (version == 2 and covered is None):
                return cls()

            vmin = _opt_float(data, "min")
            vmax = _opt_float(data, "max")
            vfirst = _opt_float(data, "first")
            vlast = _opt_float(data, "last")
            luv = _opt_float(data, "luv")
            if _INVALID in (vmin, vmax, vfirst, vlast, luv):
                return cls()

            luo = _opt_offset(data, "luo")
            lo = _opt_offset(data, "lo") if "lo" in data else luo
            if luo is _INVALID or lo is _INVALID:
                return cls()

            stale = data.get("stale", False)
            if not isinstance(stale, bool):
                return cls()

            srcs = data.get("sources", [])
            if not isinstance(srcs, (list, tuple, set)):
                return cls()
            sources = set()
            for s in srcs:
                if not isinstance(s, str) or not s:
                    return cls()
                sources.add(s)

            # ---- invariants ----
            if n < 0 or u < 0 or u > n:
                return cls()
            if wsum < 0.0 or (covered is not None and covered < 0.0):
                return cls()
            # wsum == 0 means no interval was ever closed out, so the weighted sum
            # must be exactly 0 too; a non-zero vwsum is corruption.
            if wsum == 0.0 and abs(vwsum) > _ABS_TOL:
                return cls()
            # v2: the serializer writes cov and wsum from the same lockstep
            # increments, so they must be equal (within fp rounding).
            if version == 2 and covered is not None and not _close(covered, wsum):
                return cls()
            if luo is not None and luo < 0:
                return cls()
            if lo is not None and lo < 0:
                return cls()
            if luo is not None and lo is not None and luo > lo:
                return cls()
            # every persisted temperature must satisfy the plausibility contract
            for t in (vmin, vmax, vfirst, vlast, luv):
                if t is not None and not (_PLAUSIBLE_MIN_C <= t <= _PLAUSIBLE_MAX_C):
                    return cls()
            # a usable value implies at least one usable sample and vice-versa
            if u == 0 and (sources or vmin is not None or vmax is not None
                           or vfirst is not None or vlast is not None):
                return cls()
            # a usable episode must carry a complete, ordered temperature envelope
            if u > 0 and (vmin is None or vmax is None or vfirst is None or vlast is None):
                return cls()
            if vmin is not None and vmax is not None and vmin > vmax:
                return cls()
            for edge in (vfirst, vlast, luv):
                if edge is not None and vmin is not None and vmax is not None \
                        and not _in_range(edge, vmin, vmax):
                    return cls()
            if luo is not None and luv is None:
                return cls()
            # reconstructed mean (restored) must lie within [min, max]
            if u > 0:
                mean = (vwsum / wsum) if wsum > 0.0 else (ssum / u)
                if not math.isfinite(mean):
                    return cls()
                if vmin is not None and vmax is not None and not _in_range(mean, vmin, vmax):
                    return cls()
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            return cls()

        acc._sample_count = n
        acc._usable_count = u
        acc._min = vmin
        acc._max = vmax
        acc._first = vfirst
        acc._last = vlast
        acc._any_stale = stale
        acc._sources = sources
        acc._wsum = wsum
        acc._vwsum = vwsum
        acc._simple_sum = ssum
        acc._covered_ms = float(covered if covered is not None else 0.0)
        acc._last_usable_offset = luo
        acc._last_usable_value = luv
        acc._last_offset = lo
        return acc


# -- restore helpers ----------------------------------------------------------

class _Invalid:
    __slots__ = ()


_INVALID = _Invalid()


def _req_int(d: dict, key: str) -> int:
    """Required non-negative-ish integer. Raises on missing / bool / non-int /
    non-finite so the caller's except maps it to an empty accumulator."""
    if key not in d:
        raise ValueError(f"missing {key}")
    raw = d[key]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"bad int {key}")
    return int(raw)


def _req_float(d: dict, key: str) -> float:
    if key not in d:
        raise ValueError(f"missing {key}")
    raw = d[key]
    if not _is_number(raw):
        raise ValueError(f"bad float {key}")
    f = float(raw)
    if not math.isfinite(f):
        raise ValueError(f"non-finite {key}")
    return f


def _opt_float(d: dict, key: str):
    """Optional finite float: None stays None; a present value must be a finite
    non-bool number, else :data:`_INVALID`."""
    raw = d.get(key)
    if raw is None:
        return None
    if not _is_number(raw):
        return _INVALID
    f = float(raw)
    return f if math.isfinite(f) else _INVALID


def _opt_offset(d: dict, key: str):
    """Optional non-negative integer offset (ms). None stays None; bool / float /
    non-finite / non-int -> :data:`_INVALID`."""
    raw = d.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        return _INVALID
    return int(raw)
