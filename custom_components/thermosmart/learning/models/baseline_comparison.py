"""Trustworthy baseline-versus-boost comparison (B2b-3, pure Python).

ThermoSmart can never observe the same situation both with and without boost, so
``expected_non_boost_duration_s`` is an *estimate* derived from comparable historical
authoritative NON-boost episodes — never from the boost episode's own duration or
outcome, never look-ahead, never cross-zone, never from legacy/partial/failed/
confounded/interrupted episodes.

Pipeline:
    authoritative non-boost OutcomeEpisode + dispatch record
      -> NonBoostBaselineSample (build_non_boost_sample)
      -> BaselineComparisonStore (per-zone, bounded, pruned, causal)
      -> estimate_non_boost_duration(query) -> BaselineEstimate (gated, robust)

This module performs NO control and NO model mutation; it only produces a typed,
provenance-safe duration estimate consumed by the existing B2b-1 boost context path.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

BASELINE_CONTEXT_VERSION = 1


class BaselineSource(Enum):
    HISTORICAL_ONLY = "historical_only"
    SHADOW_ONLY = "shadow_only"        # not used in B2b-3 (no calibrated shadow yet)
    BLENDED = "blended"               # not used in B2b-3
    UNAVAILABLE = "unavailable"


class BaselineRejection(Enum):
    OK = "ok"
    NO_CANDIDATES = "no_candidates"
    TOO_FEW_SAMPLES = "too_few_samples"
    LOW_EFFECTIVE_WEIGHT = "low_effective_weight"
    LOW_SIMILARITY = "low_similarity"
    LOW_RELIABILITY = "low_reliability"
    HIGH_DISPERSION = "high_dispersion"
    ALL_STALE = "all_stale"
    INCOMPATIBLE_CONTEXT = "incompatible_context"
    NO_EFFECTIVE_ONSET = "no_effective_onset"


class BaselineTimingBasis(Enum):
    """Which time axis a thermal baseline is built on (no free string semantics)."""
    EFFECTIVE_ONSET = "effective_onset"        # from effective heating onset (preferred)
    COMMAND_TO_TARGET = "command_to_target"    # command axis (reduced reliability; B2b later)
    UNAVAILABLE = "unavailable"


# ── the typed comparison-sample contract (§2) ───────────────────────────────
@dataclass(frozen=True)
class NonBoostBaselineSample:
    episode_id: str
    decision_id: str
    learning_zone_id: str
    started_at: str                       # ISO; command/decision time
    effective_heating_onset_at: Optional[str]
    target_reached_at: Optional[str]
    command_to_target_duration_s: Optional[float]
    onset_to_target_duration_s: Optional[float]
    start_deficit_c: float
    target_temperature_c: float
    start_temperature_c: float
    external_temperature_c: Optional[float]
    heat_rate_used: Optional[float]
    heat_loss_used: Optional[float]
    tpi_duty: Optional[float]
    baseline_setpoint_c: Optional[float]
    comfort_time_utc: Optional[str]
    time_of_day_bucket: int               # 0..23 (hour); -1 unknown
    presence_context: Optional[str]
    device_control_type: str              # "setpoint" | "direct_valve" | "unknown"
    dispatch_quality: str                 # "full" (only authoritative)
    sensor_reliability: float             # [0,1]
    confounder_flags: tuple[str, ...]
    outcome_reliability: float            # [0,1]
    boost_applied_c: float                # MUST be 0.0
    authoritative: bool                   # MUST be True to be usable
    schema_version: int = BASELINE_CONTEXT_VERSION

    def provenance_hash(self) -> str:
        raw = f"{self.learning_zone_id}|{self.decision_id}|{self.episode_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {k: getattr(self, k) if not isinstance(getattr(self, k), tuple)
                else list(getattr(self, k)) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NonBoostBaselineSample":
        kw = {k: d.get(k) for k in cls.__dataclass_fields__ if k != "schema_version"}
        kw["confounder_flags"] = tuple(d.get("confounder_flags", []) or ())
        kw["schema_version"] = d.get("schema_version", 0)
        return cls(**kw)


@dataclass(frozen=True)
class BaselineParams:
    # hard exclusion bounds (similarity 0 outside these)
    max_deficit_diff_c: float = 1.5
    max_target_diff_c: float = 1.0
    max_outdoor_diff_c: float = 6.0
    max_heat_rate_ratio: float = 2.0          # factor difference allowed
    # similarity weights
    w_deficit: float = 0.30
    w_target: float = 0.15
    w_outdoor: float = 0.20
    w_heat_rate: float = 0.15
    w_tpi: float = 0.10
    w_time: float = 0.10
    # gates
    min_samples: int = 3
    min_effective_weight: float = 1.0
    min_similarity: float = 0.5
    min_reliability: float = 0.4
    max_dispersion_ratio: float = 0.6         # MAD/median
    max_age_s: float = 30 * 24 * 3600.0
    recency_halflife_s: float = 10 * 24 * 3600.0
    min_sensor_reliability: float = 0.3
    trim_fraction: float = 0.1                # trimmed weighted aggregation
    full_reliability_effective_n: float = 6.0
    # retention
    count_cap: int = 200


@dataclass(frozen=True)
class BaselineQuery:
    """The boost episode asking for an estimate. All matching features are FROZEN at the
    authoritative boost decision (bound episode start), never reconstructed from the end
    state. ``baseline_cutoff_ts`` is the decision timestamp: only samples completed
    strictly before it are usable, making the estimate reconstructable from data known at
    boost start (and identical on retry)."""
    learning_zone_id: str
    decision_id: str
    episode_id: str
    baseline_cutoff_ts: str               # authoritative boost-decision time (== bound start)
    start_deficit_c: float
    target_temperature_c: float
    external_temperature_c: Optional[float]
    heat_rate_used: Optional[float]
    tpi_duty: Optional[float]
    time_of_day_bucket: int
    device_control_type: str
    query_source: str = "live_decision"
    query_context_version: int = BASELINE_CONTEXT_VERSION


@dataclass(frozen=True)
class BaselineEstimate:
    expected_non_boost_duration_s: Optional[float]            # thermal (effective-onset)
    expected_non_boost_command_duration_s: Optional[float]    # command axis (diagnostics)
    sample_count: int
    effective_sample_count: float
    effective_weight: float
    similarity: float
    reliability: float
    dispersion_s: float
    source: str
    context_version: int
    rejection_reason: str
    sample_refs: tuple[str, ...] = ()
    # B2b-3a provenance + timing basis (privacy-safe)
    baseline_cutoff_ts: Optional[str] = None
    decision_id: Optional[str] = None
    episode_id: Optional[str] = None
    query_source: str = "live_decision"
    query_context_version: int = BASELINE_CONTEXT_VERSION
    timing_basis: str = BaselineTimingBasis.UNAVAILABLE.value
    effective_onset_available: bool = False
    thermal_baseline_available: bool = False
    command_baseline_available: bool = False
    missing_thermal_reason: Optional[str] = None


# ── matching / similarity ───────────────────────────────────────────────────
def _iso_lt(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return a < b


def _parse_age_s(sample_ts: Optional[str], now_ts: str) -> Optional[float]:
    from datetime import datetime
    if sample_ts is None:
        return None
    try:
        a = datetime.fromisoformat(sample_ts)
        b = datetime.fromisoformat(now_ts)
        return max(0.0, (b - a).total_seconds())
    except (ValueError, TypeError):
        return None


def _tri(diff: float, span: float) -> float:
    if span <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(diff) / span)


def similarity(sample: NonBoostBaselineSample, q: BaselineQuery,
               p: BaselineParams) -> float:
    """Bounded deterministic similarity in [0,1]; 0 outside hard exclusion bounds."""
    if sample.learning_zone_id != q.learning_zone_id:
        return 0.0
    if sample.device_control_type != q.device_control_type:
        return 0.0
    if abs(sample.start_deficit_c - q.start_deficit_c) > p.max_deficit_diff_c:
        return 0.0
    if abs(sample.target_temperature_c - q.target_temperature_c) > p.max_target_diff_c:
        return 0.0
    if (sample.external_temperature_c is not None and q.external_temperature_c is not None
            and abs(sample.external_temperature_c - q.external_temperature_c)
            > p.max_outdoor_diff_c):
        return 0.0
    if (sample.heat_rate_used and q.heat_rate_used
            and sample.heat_rate_used > 0 and q.heat_rate_used > 0):
        ratio = max(sample.heat_rate_used, q.heat_rate_used) / min(
            sample.heat_rate_used, q.heat_rate_used)
        if ratio > p.max_heat_rate_ratio:
            return 0.0
    s = 0.0
    w = 0.0
    s += p.w_deficit * _tri(sample.start_deficit_c - q.start_deficit_c, p.max_deficit_diff_c)
    w += p.w_deficit
    s += p.w_target * _tri(sample.target_temperature_c - q.target_temperature_c,
                           p.max_target_diff_c)
    w += p.w_target
    if sample.external_temperature_c is not None and q.external_temperature_c is not None:
        s += p.w_outdoor * _tri(sample.external_temperature_c - q.external_temperature_c,
                                p.max_outdoor_diff_c)
        w += p.w_outdoor
    if sample.heat_rate_used and q.heat_rate_used:
        s += p.w_heat_rate * _tri(sample.heat_rate_used - q.heat_rate_used,
                                  max(q.heat_rate_used, 1e-6))
        w += p.w_heat_rate
    if sample.tpi_duty is not None and q.tpi_duty is not None:
        s += p.w_tpi * _tri(sample.tpi_duty - q.tpi_duty, 1.0)
        w += p.w_tpi
    if sample.time_of_day_bucket >= 0 and q.time_of_day_bucket >= 0:
        hour_diff = min(abs(sample.time_of_day_bucket - q.time_of_day_bucket),
                        24 - abs(sample.time_of_day_bucket - q.time_of_day_bucket))
        s += p.w_time * _tri(hour_diff, 6.0)
        w += p.w_time
    return round(s / w, 6) if w > 0 else 0.0


# ── robust aggregation ──────────────────────────────────────────────────────
def _weighted_median(pairs: Sequence[tuple[float, float]]) -> float:
    items = sorted(pairs, key=lambda x: x[0])
    total = sum(w for _, w in items)
    acc = 0.0
    for v, w in items:
        acc += w
        if acc >= total / 2.0:
            return v
    return items[-1][0]


def _weighted_mad(pairs: Sequence[tuple[float, float]], center: float) -> float:
    dev = [(abs(v - center), w) for v, w in pairs]
    return _weighted_median(dev)


def estimate_non_boost_duration(query: BaselineQuery, samples: Sequence[NonBoostBaselineSample],
                                *, now_ts: Optional[str] = None,
                                params: Optional[BaselineParams] = None) -> BaselineEstimate:
    """Estimate the expected non-boost thermal duration from comparable history.

    Causality + age are evaluated against ``query.baseline_cutoff_ts`` (the boost
    decision time), NOT a later 'now' — so the result is reconstructable and identical
    on retry. The thermal axis uses ONLY effective-onset durations; the command axis is
    aggregated separately for diagnostics and is never silently substituted as thermal.
    """
    p = params or BaselineParams()
    cutoff = query.baseline_cutoff_ts

    def out(reason: BaselineRejection, n: int = 0, *, cmd: Optional[float] = None,
            basis: BaselineTimingBasis = BaselineTimingBasis.UNAVAILABLE,
            onset_seen: bool = False, missing: Optional[str] = None) -> BaselineEstimate:
        return BaselineEstimate(
            expected_non_boost_duration_s=None, expected_non_boost_command_duration_s=cmd,
            sample_count=n, effective_sample_count=0.0, effective_weight=0.0, similarity=0.0,
            reliability=0.0, dispersion_s=0.0, source=BaselineSource.UNAVAILABLE.value,
            context_version=BASELINE_CONTEXT_VERSION, rejection_reason=reason.value,
            baseline_cutoff_ts=cutoff, decision_id=query.decision_id, episode_id=query.episode_id,
            query_source=query.query_source, query_context_version=query.query_context_version,
            timing_basis=basis.value, effective_onset_available=onset_seen,
            thermal_baseline_available=False, command_baseline_available=cmd is not None,
            missing_thermal_reason=(missing or reason.value))

    onset_w: list[tuple[float, float, str]] = []   # (onset_dur, weight, ref) thermal axis
    cmd_w: list[tuple[float, float]] = []           # (cmd_dur, weight) diagnostics axis
    sims: list[float] = []
    onset_seen = False
    for sm in samples:
        # causality: completed strictly before the boost decision; never the current episode.
        if sm.decision_id == query.decision_id or sm.episode_id == query.episode_id:
            continue
        if not _iso_lt(sm.target_reached_at or sm.started_at, cutoff):
            continue
        if not sm.authoritative or sm.boost_applied_c != 0.0:
            continue
        if sm.sensor_reliability < p.min_sensor_reliability:
            continue
        age = _parse_age_s(sm.target_reached_at or sm.started_at, cutoff)  # age relative to cutoff
        if age is not None and age > p.max_age_s:
            continue
        sim = similarity(sm, query, p)
        if sim < p.min_similarity:
            continue
        recency = 0.5 ** ((age or 0.0) / p.recency_halflife_s)
        weight = sim * max(sm.outcome_reliability, 0.0) * max(recency, 0.0)
        if weight <= 0.0:
            continue
        if sm.command_to_target_duration_s is not None:
            cmd_w.append((float(sm.command_to_target_duration_s), weight))
        # THERMAL axis: only effective-onset durations (no silent command substitution)
        if sm.onset_to_target_duration_s is not None:
            onset_seen = True
            onset_w.append((float(sm.onset_to_target_duration_s), weight, sm.provenance_hash()))
            sims.append(sim)

    cmd_median = _weighted_median([(c, w) for c, w in cmd_w]) if cmd_w else None
    cmd_round = round(cmd_median, 1) if cmd_median is not None else None

    n = len(onset_w)
    if n == 0:
        reason = (BaselineRejection.NO_EFFECTIVE_ONSET if cmd_w
                  else BaselineRejection.NO_CANDIDATES)
        return out(reason, n, cmd=cmd_round, onset_seen=onset_seen,
                   missing=reason.value)
    if n < p.min_samples:
        return out(BaselineRejection.TOO_FEW_SAMPLES, n, cmd=cmd_round, onset_seen=True)

    onset_w.sort(key=lambda x: x[0])
    k = int(n * p.trim_fraction)
    trimmed = onset_w[k: n - k] if n - 2 * k >= p.min_samples else onset_w
    eff_weight = sum(w for _, w, _ in trimmed)
    eff_n = (eff_weight ** 2) / sum(w * w for _, w, _ in trimmed)
    if eff_weight < p.min_effective_weight:
        return out(BaselineRejection.LOW_EFFECTIVE_WEIGHT, n, cmd=cmd_round, onset_seen=True)

    onset_pairs = [(v, w) for v, w, _ in trimmed]
    median = _weighted_median(onset_pairs)
    dispersion = _weighted_mad(onset_pairs, median)
    mean_sim = sum(sims) / len(sims)
    disp_ratio = dispersion / median if median > 0 else 1.0
    if disp_ratio > p.max_dispersion_ratio:
        return out(BaselineRejection.HIGH_DISPERSION, n, cmd=cmd_round, onset_seen=True)

    reliability = _baseline_reliability(eff_n, mean_sim, disp_ratio, p)
    if reliability < p.min_reliability:
        return out(BaselineRejection.LOW_RELIABILITY, n, cmd=cmd_round, onset_seen=True)

    return BaselineEstimate(
        expected_non_boost_duration_s=round(median, 1),
        expected_non_boost_command_duration_s=cmd_round,
        sample_count=n, effective_sample_count=round(eff_n, 3),
        effective_weight=round(eff_weight, 4), similarity=round(mean_sim, 4),
        reliability=round(reliability, 4), dispersion_s=round(dispersion, 1),
        source=BaselineSource.HISTORICAL_ONLY.value,
        context_version=BASELINE_CONTEXT_VERSION, rejection_reason=BaselineRejection.OK.value,
        sample_refs=tuple(r for _, _, r in trimmed),
        baseline_cutoff_ts=cutoff, decision_id=query.decision_id, episode_id=query.episode_id,
        query_source=query.query_source, query_context_version=query.query_context_version,
        timing_basis=BaselineTimingBasis.EFFECTIVE_ONSET.value, effective_onset_available=True,
        thermal_baseline_available=True, command_baseline_available=cmd_round is not None)


def _baseline_reliability(eff_n: float, mean_sim: float, disp_ratio: float,
                          p: BaselineParams) -> float:
    n_term = min(eff_n / p.full_reliability_effective_n, 1.0)
    disp_term = max(0.0, 1.0 - disp_ratio / max(p.max_dispersion_ratio, 1e-6))
    return max(0.0, min(1.0, n_term * mean_sim * (0.5 + 0.5 * disp_term)))


# ── per-zone bounded store ──────────────────────────────────────────────────
class BaselineComparisonStore:
    """Per-zone, bounded, causal, corruption-isolating store of non-boost samples.

    Retention: ``count_cap`` newest samples per zone (oldest pruned), plus age-cap
    pruning on load and before save. Deterministic order (by target_reached/started).
    Never holds cross-zone data; a corrupt single sample is dropped, not the store.
    """

    def __init__(self, learning_zone_id: str, params: Optional[BaselineParams] = None) -> None:
        if not learning_zone_id:
            raise ValueError("learning_zone_id must be non-empty")
        self._zone = learning_zone_id
        self._p = params or BaselineParams()
        self._samples: list[NonBoostBaselineSample] = []
        self._seen: set[str] = set()           # dedup by decision_id
        self._rejection_counts: dict[str, int] = {}   # B2b-3b diagnostics (privacy-safe)

    @property
    def size(self) -> int:
        return len(self._samples)

    def record_rejection(self, reason: Optional[str]) -> None:
        if reason:
            self._rejection_counts[reason] = self._rejection_counts.get(reason, 0) + 1

    def rejection_counts(self) -> dict:
        return dict(self._rejection_counts)

    def _key(self, s: NonBoostBaselineSample) -> str:
        return s.target_reached_at or s.started_at or ""

    def add(self, sample: Optional[NonBoostBaselineSample]) -> bool:
        """Add one authoritative non-boost sample. Idempotent per decision_id."""
        if sample is None:
            return False
        if sample.learning_zone_id != self._zone:
            return False
        if not sample.authoritative or sample.boost_applied_c != 0.0:
            return False
        if sample.decision_id in self._seen:
            return False
        self._seen.add(sample.decision_id)
        self._samples.append(sample)
        self._samples.sort(key=self._key)
        self._enforce_caps(now_ts=None)
        return True

    def _enforce_caps(self, *, now_ts: Optional[str]) -> None:
        if now_ts is not None:
            kept = []
            for s in self._samples:
                age = _parse_age_s(s.target_reached_at or s.started_at, now_ts)
                if age is None or age <= self._p.max_age_s:
                    kept.append(s)
                else:
                    self._seen.discard(s.decision_id)
            self._samples = kept
        if len(self._samples) > self._p.count_cap:
            for victim in self._samples[: len(self._samples) - self._p.count_cap]:
                self._seen.discard(victim.decision_id)
            self._samples = self._samples[-self._p.count_cap:]

    def prune(self, now_ts: str) -> None:
        self._enforce_caps(now_ts=now_ts)

    def samples(self) -> tuple[NonBoostBaselineSample, ...]:
        return tuple(self._samples)

    def estimate(self, query: BaselineQuery, *, now_ts: Optional[str] = None) -> BaselineEstimate:
        # cutoff-based (reconstructable); now_ts is accepted but not used for the estimate.
        return estimate_non_boost_duration(query, self._samples, params=self._p)

    def serialize(self) -> dict:
        return {"learning_zone_id": self._zone, "schema_version": BASELINE_CONTEXT_VERSION,
                "samples": [s.to_dict() for s in self._samples],
                "rejection_counts": dict(self._rejection_counts)}

    def restore(self, data: Mapping[str, Any], *, now_ts: Optional[str] = None) -> None:
        self._samples = []
        self._seen = set()
        self._rejection_counts = dict(data.get("rejection_counts", {}) or {})
        for raw in data.get("samples", []):
            try:
                s = NonBoostBaselineSample.from_dict(raw)
            except Exception:
                continue  # corrupt single sample isolated, not the whole store
            if s.learning_zone_id != self._zone:
                continue
            if s.schema_version != BASELINE_CONTEXT_VERSION:
                continue  # unknown/old version neutrally discarded
            if s.decision_id in self._seen:
                continue
            self._seen.add(s.decision_id)
            self._samples.append(s)
        self._samples.sort(key=self._key)
        self._enforce_caps(now_ts=now_ts)


# ── authoritative non-boost sample creation (§4) ────────────────────────────
_BASELINE_EXCLUDE_CONFOUNDERS = frozenset({
    "window_open", "manual_override", "heating_failure", "solar_gain", "external_heat",
    "additional_radiator", "reheating", "mode_change", "schedule_change", "target_change",
    "sensor_gap", "restart_gap", "supply_delay", "multi_trv_ambiguous"})


class BaselineSampleRejection(Enum):
    """Why an episode did NOT become an adaptive non-boost baseline (diagnostics only)."""
    BOOST_APPLIED = "boost_applied"                       # a real boost was applied
    BOOST_APPLICATION_UNKNOWN = "boost_application_unknown"  # applied is None (not provable)
    DISPATCH_NOT_FULLY_SUCCEEDED = "dispatch_not_fully_succeeded"
    NOT_REACHED = "not_reached"
    CONFOUNDED = "confounded"
    LOW_RELIABILITY = "low_reliability"
    NO_EFFECTIVE_ONSET = "no_effective_onset"


def _onset_and_target(points: Sequence[Any], target: float, start_temp: float,
                      tolerance: float) -> tuple[Optional[float], Optional[float]]:
    """Return (effective_onset_ms, target_reached_ms) from a trajectory. Pure scan."""
    onset_ms = None
    prev = start_temp
    target_ms = None
    band = target - tolerance
    for pt in points:
        if pt.gap or pt.value is None:
            continue
        if onset_ms is None and pt.value > prev + 0.05:
            onset_ms = pt.offset_ms
        if target_ms is None and pt.value >= band:
            target_ms = pt.offset_ms
        prev = pt.value
    return onset_ms, target_ms


def build_non_boost_sample(episode: Any, dispatch: Any, *,
                           external_temperature_c: Optional[float] = None,
                           heat_rate_used: Optional[float] = None,
                           heat_loss_used: Optional[float] = None,
                           tpi_duty: Optional[float] = None,
                           presence_context: Optional[str] = None,
                           device_control_type: str = "setpoint",
                           sensor_reliability: float = 1.0,
                           reject_out: Optional[list] = None) -> Optional[NonBoostBaselineSample]:
    """Build an authoritative non-boost baseline sample, or None if not eligible.

    B2b-3b strict zero-boost authority: ``boost_applied_c`` must be EXPLICITLY ``0.0``
    (authoritatively "no boost applied"). ``None`` means unknown/unprovable and is NEVER
    interpreted as 0.0 — it is rejected with ``boost_application_unknown``. Old/restored
    records lacking the field degrade safely (treated as None). On rejection, the reason is
    appended to ``reject_out`` (if given) for privacy-safe diagnostics counting.
    """
    from ..episode_schemas import EpisodeReason

    def reject(reason: "BaselineSampleRejection") -> None:
        if reject_out is not None:
            reject_out.append(reason.value)

    if dispatch is None:
        reject(BaselineSampleRejection.BOOST_APPLICATION_UNKNOWN)
        return None
    # ── §7/§3b strict authority ──────────────────────────────────────────────
    # Missing field on an old/restored record -> None (no auto-migration to 0.0).
    if not hasattr(dispatch, "boost_applied_c"):
        reject(BaselineSampleRejection.BOOST_APPLICATION_UNKNOWN)
        return None
    applied = dispatch.boost_applied_c
    if applied is None:                     # unknown / unprovable -> never a baseline
        reject(BaselineSampleRejection.BOOST_APPLICATION_UNKNOWN)
        return None
    if applied != 0.0:                      # a real boost was applied -> not a baseline
        reject(BaselineSampleRejection.BOOST_APPLIED)
        return None
    # a real heating control must have been EXECUTED (authoritative dispatch).
    if getattr(dispatch, "dispatch_status", None) != "fully_succeeded":
        reject(BaselineSampleRejection.DISPATCH_NOT_FULLY_SUCCEEDED)
        return None
    if episode.reason is not EpisodeReason.REACHED:      # complete thermal outcome
        reject(BaselineSampleRejection.NOT_REACHED)
        return None
    # B2b-4: SAME central confounder evaluator as the boost outcome path — a state can never
    # be admissible for the baseline yet blocking for the boost (or vice versa). Blocking and
    # interrupting confounders exclude a baseline sample.
    from .boost_reliability import evaluate_confounders
    cf = set(episode.confounder_flags or ())
    cf_eval = evaluate_confounders(list(cf))
    if cf_eval.is_blocking or cf_eval.is_interrupting:
        reject(BaselineSampleRejection.CONFOUNDED)
        return None
    if sensor_reliability < 0.3 or episode.reliability < 0.3:
        reject(BaselineSampleRejection.LOW_RELIABILITY)
        return None
    tol = getattr(episode, "comfort_tolerance_at_start", 0.5)
    onset_ms, target_ms = _onset_and_target(
        episode.trajectory.points, episode.target, episode.start_temp, tol)
    # §4/§7: a thermal baseline sample REQUIRES a detected effective heating onset — this
    # proves real heating actually happened and avoids a silent command-duration thermal
    # fallback. Episodes without a clean onset are not stored as thermal baselines.
    if target_ms is None or onset_ms is None or target_ms <= onset_ms:
        reject(BaselineSampleRejection.NO_EFFECTIVE_ONSET)
        return None
    cmd_dur = target_ms / 1000.0
    onset_dur = (target_ms - onset_ms) / 1000.0 if onset_ms is not None else None
    start_iso = episode.start_ts.isoformat()
    onset_iso = (episode.start_ts.fromtimestamp(
        episode.start_ts.timestamp() + onset_ms / 1000.0, tz=episode.start_ts.tzinfo).isoformat()
        if onset_ms is not None else None)
    reached_iso = episode.start_ts.fromtimestamp(
        episode.start_ts.timestamp() + target_ms / 1000.0,
        tz=episode.start_ts.tzinfo).isoformat()
    hour = episode.start_ts.hour
    return NonBoostBaselineSample(
        episode_id=episode.episode_id, decision_id=episode.decision_id,
        learning_zone_id=episode.learning_zone_id, started_at=start_iso,
        effective_heating_onset_at=onset_iso, target_reached_at=reached_iso,
        command_to_target_duration_s=round(cmd_dur, 1),
        onset_to_target_duration_s=round(onset_dur, 1) if onset_dur is not None else None,
        start_deficit_c=round(episode.target - episode.start_temp, 4),
        target_temperature_c=episode.target, start_temperature_c=episode.start_temp,
        external_temperature_c=external_temperature_c, heat_rate_used=heat_rate_used,
        heat_loss_used=heat_loss_used, tpi_duty=tpi_duty,
        baseline_setpoint_c=getattr(dispatch, "baseline_setpoint_c", None),
        comfort_time_utc=None, time_of_day_bucket=hour, presence_context=presence_context,
        device_control_type=device_control_type, dispatch_quality="full",
        sensor_reliability=sensor_reliability, confounder_flags=tuple(sorted(cf)),
        outcome_reliability=float(episode.reliability), boost_applied_c=0.0,
        authoritative=True)

