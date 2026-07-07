"""OnsetDelayModel for LE 2.0 (pure Python).

Learns the effective heating onset delay for a zone: the elapsed time between
the TRV command (episode.start_ts) and the first measurable, sustained room-
temperature response. This quantity is distinct from the heating rate and must
NEVER be folded into the rate learning.

Authority chain (command_lead_time = effective_onset_delay + room_heating_duration):
  A  command_lead_time         ← returned to coordinator (sum)
  B  effective_onset_delay     ← THIS model (adaptive; prior = 5 min)
  C  room_heating_duration     ← HeatRateModel (never includes B)

Onset detection:
  The trajectory in a HeatingEpisode captures temperature readings relative to
  a monotonic t0. Onset delay = elapsed time from the first trajectory point
  until the temperature first diverges meaningfully from start_temp and then
  holds that direction across a confirmation window (≥ 2 consecutive points
  rising).  Passive solar, neighbour heat, sensor noise, and window-cooling
  are excluded via confounder_flags and minimum-rise thresholds.

5-level fallback hierarchy (most-specific → prior):
  L1  time_bucket + setback_class + device_profile  (≥ bucket_min_samples)
  L2  time_bucket + setback_class                   (≥ bucket_min_samples)
  L3  time_bucket                                   (≥ bucket_min_samples)
  L4  general                                       (any evidence)
  L5  cold-start prior                              (always available)

Context features actively used for conditioning:
  time_bucket        – "morning" | "afternoon" | "evening" | "night"
  setback_class      – derived from setback_depth_c + setback_duration_h
                       ("none" | "shallow" | "moderate" | "deep")
  device_profile_key – TRV profile slug (L1 only, with enough evidence)
  outdoor_temp_c     – not a bucket dimension; used as confidence/match factor

No HA, no I/O, no state-sharing across models.  Mutates no input.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from ..contracts import (
    ConfidenceContribution,
    DataQuality,
    ExportScope,
    ModelRebuildResult,
    OutlierStatus,
    Prediction,
    PredictionType,
    Regime,
    RejectionReason,
    UpdateEligibilityResult,
)
from ..episode_schemas import EpisodeType, HeatingEpisode
from .base import clamp, freshness_from_age, is_finite, outlier_z, robust_ema_update

MODEL_VERSION = 1
PARAMETER_VERSION = 1
MODEL_NAME = "onset_delay"

# Minimum measurable temperature rise that constitutes a real response (°C).
# Below this threshold a point is treated as sensor noise.
_ONSET_RISE_THRESHOLD_C = 0.08

# Confirmation: require at least this many consecutive rising points after
# the first candidate point before declaring onset confirmed.
_ONSET_CONFIRMATION_POINTS = 2

# Setback depth above which a schedule transition is classified as "significant"
# (overnight heating setback → typically longer supply/onset delay).
_SETBACK_THRESHOLD_C = 0.5

# Setback-class classifier thresholds (versioned for reproducible classification).
_SETBACK_CLASS_VERSION = 1
_SETBACK_CLASS_SHALLOW_MIN_C = 0.5   # depth ≥ 0.5°C qualifies as any setback
_SETBACK_CLASS_MODERATE_MIN_C = 1.5  # depth ≥ 1.5°C → moderate
_SETBACK_CLASS_DEEP_MIN_C = 3.0      # depth ≥ 3.0°C → deep
_SETBACK_CLASS_LONG_H = 6.0          # duration ≥ 6h → promoted to deep regardless of depth

# Outdoor band breakpoints (°C).  Three bands: cold < 5, mild 5–15, warm ≥ 15.
# Used as a confidence/match factor, NOT as an independent bucket dimension.
_OUTDOOR_BAND_BREAKPOINTS = (5.0, 15.0)

# Hard cap on total specific bucket count per model instance.
# Prevents explosion when many combinations of time/setback/device are encountered.
# 4 times × 4 setback_classes = 16 L2 buckets + 4 L3 + any L1 → 24 leaves comfortable headroom.
_BUCKET_COUNT_CAP = 24


# ---------------------------------------------------------------------------
# Rejection reasons
# ---------------------------------------------------------------------------

class OnsetDelayRejection(Enum):
    WRONG_EPISODE_TYPE = "wrong_episode_type"
    WRONG_ZONE = "wrong_zone"
    DISTURBED = "disturbed"
    WINDOW_CONFOUNDER = "window_confounder"
    HEATING_FAILURE = "heating_failure"
    AFTERHEAT_CONTAMINATION = "afterheat_contamination"
    LOW_RELIABILITY = "low_reliability"
    INSUFFICIENT_DURATION = "insufficient_duration"
    MISSING_TRAJECTORY = "missing_trajectory"
    INSUFFICIENT_POINTS = "insufficient_points"
    NO_RESPONSE = "no_response"          # No measurable temperature rise
    PASSIVE_WARMING = "passive_warming"  # Rise without confirmed active heating
    IMPLAUSIBLE_DELAY = "implausible_delay"
    DUPLICATE_EPISODE = "duplicate_episode"
    VERSION_MISMATCH = "version_mismatch"
    SEVERE_OUTLIER = "severe_outlier"
    CONTEXT_EPISODE_MISMATCH = "context_episode_mismatch"


_TO_CONTRACT: Mapping[OnsetDelayRejection, RejectionReason] = {
    OnsetDelayRejection.WRONG_EPISODE_TYPE: RejectionReason.NOT_ELIGIBLE,
    OnsetDelayRejection.WRONG_ZONE: RejectionReason.NOT_ELIGIBLE,
    OnsetDelayRejection.DISTURBED: RejectionReason.DISTURBED_REGIME,
    OnsetDelayRejection.WINDOW_CONFOUNDER: RejectionReason.DISTURBED_REGIME,
    OnsetDelayRejection.HEATING_FAILURE: RejectionReason.DISTURBED_REGIME,
    OnsetDelayRejection.AFTERHEAT_CONTAMINATION: RejectionReason.NOT_ELIGIBLE,
    OnsetDelayRejection.LOW_RELIABILITY: RejectionReason.LOW_RELIABILITY,
    OnsetDelayRejection.INSUFFICIENT_DURATION: RejectionReason.NOT_ELIGIBLE,
    OnsetDelayRejection.MISSING_TRAJECTORY: RejectionReason.MISSING_REQUIRED_DATA,
    OnsetDelayRejection.INSUFFICIENT_POINTS: RejectionReason.NOT_ELIGIBLE,
    OnsetDelayRejection.NO_RESPONSE: RejectionReason.NOT_ELIGIBLE,
    OnsetDelayRejection.PASSIVE_WARMING: RejectionReason.DISTURBED_REGIME,
    OnsetDelayRejection.IMPLAUSIBLE_DELAY: RejectionReason.OUTLIER,
    OnsetDelayRejection.DUPLICATE_EPISODE: RejectionReason.DUPLICATE,
    OnsetDelayRejection.VERSION_MISMATCH: RejectionReason.NOT_ELIGIBLE,
    OnsetDelayRejection.SEVERE_OUTLIER: RejectionReason.OUTLIER,
    OnsetDelayRejection.CONTEXT_EPISODE_MISMATCH: RejectionReason.NOT_ELIGIBLE,
}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OnsetDelayParameters:
    """Versioned, conservative thresholds for onset-delay learning."""

    # Eligibility
    min_episode_reliability: float = 0.4
    min_duration_s: float = 300.0        # episode must last ≥ 5 min
    min_trajectory_points: int = 4       # need enough points to confirm rise
    min_onset_delay_min: float = 0.0     # immediate response is valid
    # Raised from 30 → 45 min: real central-heating supply delays can reach 40+ min
    # after deep overnight setbacks (cold pipes, slow TRV response). 45 min gives a
    # 5-min margin above the worst observed real-world case; the total preheat cap
    # (120 min) remains the binding control boundary.
    max_onset_delay_min: float = 45.0
    min_temp_rise_c: float = 0.08        # minimum per-point rise to detect response

    # Outlier rejection
    outlier_mad_k_mild: float = 3.0
    outlier_mad_k_severe: float = 6.0
    min_samples_for_outlier: int = 5

    # Sample weighting
    max_sample_weight: float = 1.0

    # Bucket activation
    bucket_min_samples: int = 5
    bucket_seed_evidence: float = 1.0

    # Deduplication / retention
    dedup_max_ids: int = 500
    research_sample_cap: int = 50

    # Cold-start prior (versioned; 5 min is the conservative starting assumption)
    cold_start_prior_min: float = 5.0
    cold_start_confidence_cap: float = 0.35

    # Confidence ramp
    full_confidence_samples: float = 20.0

    parameter_version: int = PARAMETER_VERSION


# ---------------------------------------------------------------------------
# Sample / Bucket / State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OnsetDelaySample:
    """Immutable record of one learned onset-delay observation."""
    source_episode_id: str
    learning_zone_id: str
    onset_delay_min: float          # observed delay (minutes)
    episode_duration_s: float       # context: total episode length
    start_temp_c: float             # temp at command time
    target_temp_c: float            # demanded setpoint
    episode_reliability: float
    effective_weight: float
    bucket: str                     # "general" initially
    time_bucket: Optional[str]      # "morning"|"afternoon"|"evening"|"night" (future)
    outdoor_band: Optional[int]     # future conditioning index
    profile_id: Optional[str]
    builder_version: int
    classifier_version: int
    data_quality: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class OnsetDelayBucket:
    key: str
    onset_delay_min: float      # robust EMA of observed onset delays
    effective_n: float          # shrinking learning rate denominator
    dispersion: float           # EMA of absolute deviation
    sample_count: int
    last_update_ts: Optional[str] = None

    @property
    def has_evidence(self) -> bool:
        return self.sample_count > 0 and self.effective_n > 0.0


def _empty_bucket(key: str) -> OnsetDelayBucket:
    return OnsetDelayBucket(key=key, onset_delay_min=0.0, effective_n=0.0,
                            dispersion=0.0, sample_count=0)


@dataclass(frozen=True)
class OnsetDelayState:
    learning_zone_id: str
    general: OnsetDelayBucket
    buckets: Mapping[str, OnsetDelayBucket] = field(default_factory=dict)
    processed_ids: tuple[str, ...] = ()
    recent_samples: tuple[OnsetDelaySample, ...] = ()
    aggregate_reliability: float = 0.0
    last_update_ts: Optional[str] = None
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    outlier_counts: Mapping[str, int] = field(default_factory=dict)
    dedup_count: int = 0
    model_version: int = MODEL_VERSION
    parameter_version: int = PARAMETER_VERSION


@dataclass(frozen=True)
class OnsetDelayDiagnostics:
    general_onset_delay_min: float
    bucket_delays: Mapping[str, float]
    sample_counts: Mapping[str, int]
    effective_sample_counts: Mapping[str, float]
    dispersion: Mapping[str, float]
    confidence: float
    last_update_ts: Optional[str]
    rejection_counts: Mapping[str, int]
    outlier_counts: Mapping[str, int]
    dedup_count: int
    fallback_used: bool
    cold_start_prior_min: float
    model_version: int
    parameter_version: int


@dataclass(frozen=True)
class OnsetDelayPredictionContext:
    """Context for predicting the onset delay.

    Prediction follows a 5-level fallback hierarchy:
      L1  time_bucket + setback_class + device_profile   (most specific)
      L2  time_bucket + setback_class
      L3  time_bucket
      L4  general                                        (any evidence)
      L5  cold-start prior                               (always available)

    ``outdoor_temp_c`` is NOT a bucket dimension — it acts as a confidence/
    match-quality factor that can reduce confidence when missing or mismatched.
    Missing outdoor context never blocks the prediction.
    """
    profile_id: Optional[str] = None               # device profile key (L1)
    outdoor_temp_c: Optional[float] = None         # match-quality factor only
    time_bucket: Optional[str] = None              # primary bucket discriminator (L2/L3)
    setback_depth_c: Optional[float] = None        # setback_class input
    setback_duration_h: Optional[float] = None     # setback_class input (long → deep)


@dataclass(frozen=True)
class OnsetDelayUpdateContext:
    """Persistable, episode-bound conditioning evidence.

    JSON-serializable; stored alongside the episode for deterministic rebuild.
    All optional context features are for future conditioning; the model is
    correct without them (general-bucket fallback).
    """
    source_episode_id: str
    time_bucket: Optional[str] = None
    outdoor_temperature_c: Optional[float] = None
    device_profile_key: Optional[str] = None
    setback_depth_c: Optional[float] = None     # comfort_temp − previous_setpoint
    setback_duration_h: Optional[float] = None  # how long the setback lasted
    source_feature_version: int = 1
    source_refs: tuple[str, ...] = ()
    quality: DataQuality = DataQuality.OK

    def __post_init__(self) -> None:
        if not self.source_episode_id:
            raise ValueError("source_episode_id must be non-empty")

    def to_dict(self) -> dict:
        return {
            "source_episode_id": self.source_episode_id,
            "time_bucket": self.time_bucket,
            "outdoor_temperature_c": self.outdoor_temperature_c,
            "device_profile_key": self.device_profile_key,
            "setback_depth_c": self.setback_depth_c,
            "setback_duration_h": self.setback_duration_h,
            "source_feature_version": self.source_feature_version,
            "source_refs": list(self.source_refs),
            "quality": self.quality.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OnsetDelayUpdateContext":
        return cls(
            source_episode_id=d["source_episode_id"],
            time_bucket=d.get("time_bucket"),
            outdoor_temperature_c=d.get("outdoor_temperature_c"),
            device_profile_key=d.get("device_profile_key"),
            setback_depth_c=d.get("setback_depth_c"),
            setback_duration_h=d.get("setback_duration_h"),
            source_feature_version=d.get("source_feature_version", 1),
            source_refs=tuple(d.get("source_refs", [])),
            quality=DataQuality(d.get("quality", DataQuality.OK.value)),
        )


@dataclass(frozen=True)
class OnsetDelayRebuildItem:
    episode: Any
    context: Optional[OnsetDelayUpdateContext] = None


# ---------------------------------------------------------------------------
# Onset detection (authoritative, single location)
# ---------------------------------------------------------------------------

def _detect_onset_delay(episode: HeatingEpisode,
                        *,
                        rise_threshold: float = _ONSET_RISE_THRESHOLD_C,
                        confirmation_points: int = _ONSET_CONFIRMATION_POINTS,
                        max_delay_min: float = 30.0,
                        ) -> tuple:
    """Extract onset delay (minutes) from a HeatingEpisode trajectory.

    Returns (onset_delay_min, rejection_reason, n_valid_points):
      onset_delay_min  – float or None if not detectable
      rejection_reason – OnsetDelayRejection or None if accepted
      n_valid_points   – int count of usable trajectory points

    Algorithm:
      1. Collect trajectory points that are not gaps and have finite values.
      2. Record t0_ms = first point's offset_ms (reference for elapsed time).
      3. Walk forward; when a point exceeds start_temp + rise_threshold, check
         that the NEXT `confirmation_points` non-gap points all continue to
         rise above start_temp (confirming sustained response, not noise).
      4. onset_delay_min = (candidate_offset_ms - t0_ms) / 60_000.
      5. Clamp to [0, max_delay_min]; reject if outside.

    A single spike that immediately drops back does NOT trigger onset.
    Passive solar and sensor noise cannot produce `confirmation_points`
    consecutive rising samples before the trajectory ends.
    """
    traj = episode.trajectory
    if traj is None or not traj.points:
        return None, OnsetDelayRejection.MISSING_TRAJECTORY, 0

    # Collect usable points
    valid_pts: list[tuple[int, float]] = []
    for pt in traj.points:
        if pt.gap or pt.value is None:
            continue
        if not is_finite(pt.value):
            continue
        valid_pts.append((pt.offset_ms, pt.value))

    n = len(valid_pts)
    if n < confirmation_points + 2:
        return None, OnsetDelayRejection.INSUFFICIENT_POINTS, n

    t0_ms = valid_pts[0][0]
    start_temp = episode.start_temp

    # Causal onset guard: if the temperature at t0 (command time) is already
    # above the rise threshold, warming predates the heating command (passive
    # solar, neighbour heat, internal loads).  This is not a heating onset.
    if valid_pts[0][1] >= start_temp + rise_threshold:
        return None, OnsetDelayRejection.PASSIVE_WARMING, n

    # Walk forward looking for the first confirmed onset
    for i, (offset_ms, temp) in enumerate(valid_pts):
        if temp < start_temp + rise_threshold:
            continue  # still at baseline

        # Candidate onset found; verify confirmation window
        confirmed = 0
        for j in range(i + 1, min(i + 1 + confirmation_points, n)):
            _, next_temp = valid_pts[j]
            if next_temp > start_temp:
                confirmed += 1
            else:
                break

        if confirmed < confirmation_points:
            continue  # spike without sustained follow-through

        # Confirmed onset
        onset_ms = offset_ms - t0_ms
        onset_min = onset_ms / 60_000.0
        onset_min = clamp(max(0.0, onset_min), 0.0, max_delay_min)
        return onset_min, None, n

    return None, OnsetDelayRejection.NO_RESPONSE, n


# ---------------------------------------------------------------------------
# Context / bucket helpers (activated conditioning paths)
# ---------------------------------------------------------------------------

def _time_bucket_key(time_bucket: str) -> str:
    """Stable L3 bucket key (time only). Max 4 values: morning/afternoon/evening/night."""
    return time_bucket


def _setback_class(depth_c: Optional[float],
                   duration_h: Optional[float] = None) -> str:
    """Classify a schedule setback into one of four stable categories.

    Classification version: _SETBACK_CLASS_VERSION = 1.
    Thresholds versioned; future bump uses migrate_state().

      none     – no relevant setback (depth < 0.5°C)
      shallow  – 0.5°C ≤ depth < 1.5°C (and duration < 6h)
      moderate – 1.5°C ≤ depth < 3.0°C (and duration < 6h)
      deep     – depth ≥ 3.0°C OR duration ≥ 6h (long cold-soak regardless of depth)
    """
    if depth_c is None or depth_c < _SETBACK_CLASS_SHALLOW_MIN_C:
        return "none"
    if duration_h is not None and duration_h >= _SETBACK_CLASS_LONG_H:
        return "deep"   # long cold-soak → deep regardless of temperature depth
    if depth_c >= _SETBACK_CLASS_DEEP_MIN_C:
        return "deep"
    if depth_c >= _SETBACK_CLASS_MODERATE_MIN_C:
        return "moderate"
    return "shallow"


def _bucket_key_l2(time_bucket: str, setback_cls: str) -> str:
    """Stable L2 bucket key: time_bucket:setback_class. Max 4 × 4 = 16 per zone."""
    return f"{time_bucket}:{setback_cls}"


def _bucket_key_l1(time_bucket: str, setback_cls: str,
                   device_key: Optional[str]) -> Optional[str]:
    """Stable L1 bucket key: time_bucket:setback_class:device_profile.

    Returns None when no device_profile available (skips L1 path).
    The device_key must be a stable non-identifying slug, not an entity name.
    """
    if not device_key:
        return None
    return f"{time_bucket}:{setback_cls}:{device_key}"


def _outdoor_confidence_factor(outdoor_temp_c: Optional[float]) -> float:
    """Confidence multiplier for outdoor temperature availability.

    Outdoor temperature is NOT a bucket dimension.  When missing it reduces
    confidence to reflect slightly weaker context match (0.9), otherwise 1.0.
    This never blocks a prediction — it only modulates confidence.
    """
    return 1.0 if outdoor_temp_c is not None else 0.9


def _setback_weight_modifier(setback_depth_c: Optional[float],
                              setback_duration_h: Optional[float] = None) -> float:
    """EMA weight multiplier based on setback class.

    Deep overnight setbacks are more informative evidence; they receive a
    slightly higher learning weight.  Modifier stays close to 1.0 (never > 1.2).
    """
    sc = _setback_class(setback_depth_c, setback_duration_h)
    if sc == "none":
        return 1.0
    if sc == "shallow":
        return 1.03
    if sc == "moderate":
        return 1.08
    return 1.15   # deep


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class OnsetDelayModel:
    """Adaptive onset-delay model following the LE 2.0 Model contract.

    Single authority for: time from TRV command to first sustained room
    temperature response.  HeatRate learning MUST start from after this delay
    — the two models are independent and must never share learning evidence.
    """

    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    parameter_version = PARAMETER_VERSION
    consumed_raw_tracks: tuple[str, ...] = ()
    consumed_episode_types = (EpisodeType.HEATING.value,)
    required_features = ("indoor_temp",)
    optional_features = ("outdoor_temp", "time_bucket", "device_profile")
    supported_prediction_types = (PredictionType.ONSET_DELAY,)
    min_tier = 0  # works with TRV-only setups

    def __init__(self, learning_zone_id: str,
                 params: Optional[OnsetDelayParameters] = None) -> None:
        if not learning_zone_id:
            raise ValueError("learning_zone_id must be non-empty")
        self._zone = learning_zone_id
        self._params = params or OnsetDelayParameters()
        self._state = self._empty_state()

    # -- cold-start prior -----------------------------------------------

    def cold_start_prior(self) -> Mapping[str, Any]:
        return {"onset_delay_min": self._params.cold_start_prior_min,
                "source": "cold_start_prior"}

    def _empty_state(self) -> OnsetDelayState:
        return OnsetDelayState(learning_zone_id=self._zone, general=_empty_bucket("general"))

    # -- onset detection (authoritative) --------------------------------

    def _extract_onset(self, episode: HeatingEpisode):
        """Return (onset_min, rejection_reason, n_valid_points)."""
        return _detect_onset_delay(
            episode,
            rise_threshold=self._params.min_temp_rise_c,
            confirmation_points=_ONSET_CONFIRMATION_POINTS,
            max_delay_min=self._params.max_onset_delay_min,
        )

    # -- eligibility ----------------------------------------------------

    def update_eligibility(self, episode: Any) -> UpdateEligibilityResult:
        p = self._params
        if not isinstance(episode, HeatingEpisode):
            return self._reject(OnsetDelayRejection.WRONG_EPISODE_TYPE)
        if episode.learning_zone_id != self._zone:
            return self._reject(OnsetDelayRejection.WRONG_ZONE)
        if episode.episode_schema_version != 1:
            return self._reject(OnsetDelayRejection.VERSION_MISMATCH)
        if episode.regime is not Regime.ACTIVE_HEATING:
            return self._reject(OnsetDelayRejection.DISTURBED, regime=episode.regime)
        cf = set(episode.confounder_flags)
        if "window_open" in cf:
            return self._reject(OnsetDelayRejection.WINDOW_CONFOUNDER)
        if "heating_failure" in cf:
            return self._reject(OnsetDelayRejection.HEATING_FAILURE)
        if "reheating" in cf:
            return self._reject(OnsetDelayRejection.AFTERHEAT_CONTAMINATION)
        if episode.episode_id in self._state.processed_ids:
            return self._reject(OnsetDelayRejection.DUPLICATE_EPISODE)
        if episode.reliability < p.min_episode_reliability:
            return self._reject(OnsetDelayRejection.LOW_RELIABILITY)
        if (episode.end_ts - episode.start_ts).total_seconds() < p.min_duration_s:
            return self._reject(OnsetDelayRejection.INSUFFICIENT_DURATION)
        onset_min, rejection, n = self._extract_onset(episode)
        if rejection is not None:
            return self._reject(rejection)
        if not is_finite(onset_min) or onset_min < 0 or onset_min > p.max_onset_delay_min:
            return self._reject(OnsetDelayRejection.IMPLAUSIBLE_DELAY)
        weight = clamp(episode.reliability, 0.0, p.max_sample_weight)
        return UpdateEligibilityResult(
            accepted=True, reliability=episode.reliability, weight=weight,
            outlier_status=OutlierStatus.NONE, regime=Regime.ACTIVE_HEATING,
            source_episode_id=episode.episode_id,
            feature_extractor_version=1)

    def _reject(self, reason: OnsetDelayRejection, *,
                regime=None) -> UpdateEligibilityResult:
        return UpdateEligibilityResult(
            accepted=False, reliability=0.0, weight=0.0,
            rejection_reason=_TO_CONTRACT[reason],
            confounder_flags=(reason.value,),
            regime=regime)

    # -- update ---------------------------------------------------------

    def update(self, episode: Any,
               context: Optional[OnsetDelayUpdateContext] = None) -> UpdateEligibilityResult:
        """Learn one onset delay from a HeatingEpisode."""
        if context is not None and context.source_episode_id != episode.episode_id:
            self._bump_rejection(OnsetDelayRejection.CONTEXT_EPISODE_MISMATCH.value)
            return self._reject(OnsetDelayRejection.CONTEXT_EPISODE_MISMATCH)
        elig = self.update_eligibility(episode)
        if not elig.accepted:
            self._bump_rejection(
                elig.confounder_flags[0] if elig.confounder_flags else "unknown")
            if elig.confounder_flags and \
                    elig.confounder_flags[0] == OnsetDelayRejection.DUPLICATE_EPISODE.value:
                self._state = replace(self._state, dedup_count=self._state.dedup_count + 1)
            return elig

        onset_min, _, n = self._extract_onset(episode)
        weight = elig.weight

        # Outlier scoring: use the MOST SPECIFIC reference bucket available for
        # this context, but only one with the SAME setback class.
        # Rationale: morning:deep (20 min) vs morning:shallow (8 min) are legitimate
        # distinct populations; using L3 "morning" (mixed) as reference would falsely
        # reject all shallow episodes as outliers against the deep distribution.
        # Resolution: L1 → L2 (same setback class) → L3 only if no specific setback
        # context exists → general when no time context.
        general = self._state.general
        outlier_status = OutlierStatus.NONE
        ref_bucket: Optional[OnsetDelayBucket] = None
        if context is not None and context.time_bucket is not None:
            sc = _setback_class(context.setback_depth_c, context.setback_duration_h)
            p_min = self._params.min_samples_for_outlier
            # L1 reference (exact match: time + setback_class + device)
            l1_key = _bucket_key_l1(context.time_bucket, sc, context.device_profile_key)
            if l1_key:
                l1 = self._state.buckets.get(l1_key)
                if l1 and l1.sample_count >= p_min:
                    ref_bucket = l1
            # L2 reference (same time + setback_class only)
            if ref_bucket is None:
                l2 = self._state.buckets.get(_bucket_key_l2(context.time_bucket, sc))
                if l2 and l2.sample_count >= p_min:
                    ref_bucket = l2
            # L3 is only used as reference when NO specific setback class is given,
            # because L3 mixes all setback classes — comparing a deep morning episode
            # against shallow morning data would be a false context comparison.
            if ref_bucket is None and context.setback_depth_c is None:
                l3 = self._state.buckets.get(_time_bucket_key(context.time_bucket))
                if l3 and l3.sample_count >= p_min:
                    ref_bucket = l3
            # No reference found → first episodes for this exact context: skip check
        elif general.sample_count >= self._params.min_samples_for_outlier:
            ref_bucket = general

        if ref_bucket is not None:
            z = outlier_z(onset_min, ref_bucket.onset_delay_min, ref_bucket.dispersion)
            if z > self._params.outlier_mad_k_severe:
                self._bump_outlier("severe")
                return self._reject(OnsetDelayRejection.SEVERE_OUTLIER)
            if z > self._params.outlier_mad_k_mild:
                outlier_status = OutlierStatus.MILD
                weight *= 0.25
                self._bump_outlier("mild")

        # Setback weight modifier: uses setback class (depth + duration → more informative)
        setback_mod = _setback_weight_modifier(
            context.setback_depth_c if context is not None else None,
            context.setback_duration_h if context is not None else None,
        )
        effective_weight = clamp(weight * setback_mod, 0.0, self._params.max_sample_weight)

        # Always update the general bucket (L4 — aggregate fallback for all contexts).
        new_general = _apply_bucket(general, onset_min, effective_weight)
        new_buckets = dict(self._state.buckets)
        bucket_name = "general"

        # Specific bucket updates (L3 → L2 → L1) when context is available.
        # Each level is only created when the total specific bucket count stays
        # below _BUCKET_COUNT_CAP (prevents explosion from rare context combos).
        if context is not None and context.time_bucket is not None:
            sc_ctx = _setback_class(context.setback_depth_c, context.setback_duration_h)

            # L3: time_bucket
            l3_key = _time_bucket_key(context.time_bucket)
            if l3_key in new_buckets or len(new_buckets) < _BUCKET_COUNT_CAP:
                seed = new_buckets.get(l3_key, _empty_bucket(l3_key))
                new_buckets[l3_key] = _apply_bucket(seed, onset_min, effective_weight)
            bucket_name = l3_key

            # L2: time_bucket + setback_class
            l2_key = _bucket_key_l2(context.time_bucket, sc_ctx)
            if l2_key in new_buckets or len(new_buckets) < _BUCKET_COUNT_CAP:
                seed2 = new_buckets.get(l2_key, _empty_bucket(l2_key))
                new_buckets[l2_key] = _apply_bucket(seed2, onset_min, effective_weight)
                bucket_name = l2_key  # primary for this sample

            # L1: time_bucket + setback_class + device_profile (when available)
            l1_key_ctx = _bucket_key_l1(
                context.time_bucket, sc_ctx, context.device_profile_key)
            if l1_key_ctx is not None:
                if l1_key_ctx in new_buckets or len(new_buckets) < _BUCKET_COUNT_CAP:
                    seed1 = new_buckets.get(l1_key_ctx, _empty_bucket(l1_key_ctx))
                    new_buckets[l1_key_ctx] = _apply_bucket(
                        seed1, onset_min, effective_weight)
                    bucket_name = l1_key_ctx  # primary for this sample

        sample = OnsetDelaySample(
            source_episode_id=episode.episode_id,
            learning_zone_id=self._zone,
            onset_delay_min=round(onset_min, 3),
            episode_duration_s=(episode.end_ts - episode.start_ts).total_seconds(),
            start_temp_c=episode.start_temp,
            target_temp_c=episode.target,
            episode_reliability=episode.reliability,
            effective_weight=effective_weight,
            bucket=bucket_name,
            time_bucket=context.time_bucket if context is not None else None,
            outdoor_band=None,
            profile_id=context.device_profile_key if context is not None else None,
            builder_version=episode.builder_version,
            classifier_version=episode.classifier_version,
            data_quality=episode.time_quality.value,
            reason_codes=(("mild_outlier",) if outlier_status is OutlierStatus.MILD else ()))

        processed = (self._state.processed_ids + (episode.episode_id,))[
            -self._params.dedup_max_ids:]
        recent = (self._state.recent_samples + (sample,))[-self._params.research_sample_cap:]
        sc = new_general.sample_count
        agg = (self._state.aggregate_reliability * (sc - 1) + episode.reliability) / sc
        self._state = replace(
            self._state, general=new_general, buckets=new_buckets,
            processed_ids=processed, recent_samples=recent,
            aggregate_reliability=agg, last_update_ts=_iso(episode.end_ts))
        return elig

    def _bump_rejection(self, code: str) -> None:
        rc = dict(self._state.rejection_counts)
        rc[code] = rc.get(code, 0) + 1
        self._state = replace(self._state, rejection_counts=rc)

    def _bump_outlier(self, kind: str) -> None:
        oc = dict(self._state.outlier_counts)
        oc[kind] = oc.get(kind, 0) + 1
        self._state = replace(self._state, outlier_counts=oc)

    # -- predictions ----------------------------------------------------

    def predict(self, context: Any) -> Prediction:
        return self.predict_onset_delay(
            context if isinstance(context, OnsetDelayPredictionContext)
            else OnsetDelayPredictionContext())

    def predict_onset_delay(self,
                            context: OnsetDelayPredictionContext) -> Prediction:
        """5-level fallback hierarchy for onset delay prediction.

        L1  time_bucket + setback_class + device_profile  (most specific)
        L2  time_bucket + setback_class
        L3  time_bucket
        L4  general (all episodes)
        L5  cold-start prior (5 min)

        Missing context features are recorded in missing_evidence but NEVER block
        the prediction — the model degrades gracefully to the best available level.
        Outdoor temperature acts as a confidence match-factor only (not a bucket).
        A thin specific bucket (< bucket_min_samples) is NEVER preferred over a
        well-populated general bucket; it simply falls through to the next level.
        """
        p = self._params
        missing: list[str] = []
        fallback_path: list[str] = []

        if context.outdoor_temp_c is None:
            missing.append("outdoor_temp")
        if context.time_bucket is None:
            missing.append("time_bucket")
        if context.setback_depth_c is None:
            missing.append("setback_depth")

        outdoor_factor = _outdoor_confidence_factor(context.outdoor_temp_c)
        g = self._state.general
        general_n = g.sample_count

        if context.time_bucket is not None:
            sc = _setback_class(context.setback_depth_c, context.setback_duration_h)

            # L1: time_bucket + setback_class + device_profile
            l1_key = _bucket_key_l1(context.time_bucket, sc, context.profile_id)
            if l1_key is not None:
                l1_b = self._state.buckets.get(l1_key)
                if l1_b is not None and l1_b.sample_count >= p.bucket_min_samples:
                    return self._onset_prediction(
                        l1_b.onset_delay_min, l1_b.sample_count,
                        learned=1.0, fallback=False, bucket=l1_key,
                        context_level="L1", outdoor_factor=outdoor_factor,
                        general_n=general_n, fallback_path=fallback_path,
                        reasons=("L1_specific",), missing=missing,
                        dispersion=l1_b.dispersion)
                fallback_path.append(f"L1:{l1_key}:thin")

            # L2: time_bucket + setback_class
            l2_key = _bucket_key_l2(context.time_bucket, sc)
            l2_b = self._state.buckets.get(l2_key)
            if l2_b is not None and l2_b.sample_count >= p.bucket_min_samples:
                return self._onset_prediction(
                    l2_b.onset_delay_min, l2_b.sample_count,
                    learned=1.0, fallback=False, bucket=l2_key,
                    context_level="L2", outdoor_factor=outdoor_factor,
                    general_n=general_n, fallback_path=fallback_path,
                    reasons=("L2_time_setback",), missing=missing,
                    dispersion=l2_b.dispersion)
            fallback_path.append(f"L2:{l2_key}:thin")

            # L3: time_bucket
            l3_key = _time_bucket_key(context.time_bucket)
            l3_b = self._state.buckets.get(l3_key)
            if l3_b is not None and l3_b.sample_count >= p.bucket_min_samples:
                return self._onset_prediction(
                    l3_b.onset_delay_min, l3_b.sample_count,
                    learned=1.0, fallback=False, bucket=l3_key,
                    context_level="L3", outdoor_factor=outdoor_factor,
                    general_n=general_n, fallback_path=fallback_path,
                    reasons=("L3_time_bucket",), missing=missing,
                    dispersion=l3_b.dispersion)
            fallback_path.append(f"L3:{l3_key}:thin")

        # L4: general
        if g.has_evidence:
            return self._onset_prediction(
                g.onset_delay_min, g.sample_count,
                learned=1.0, fallback=False, bucket="general",
                context_level="L4", outdoor_factor=outdoor_factor,
                general_n=general_n, fallback_path=fallback_path,
                reasons=("L4_general",), missing=missing,
                dispersion=g.dispersion)
        fallback_path.append("L4:general:no_evidence")

        # L5: cold-start prior
        return self._onset_prediction(
            p.cold_start_prior_min, 0,
            learned=0.0, fallback=True, bucket="cold_start_prior",
            context_level="L5", outdoor_factor=outdoor_factor,
            general_n=general_n, fallback_path=fallback_path,
            reasons=("L5_prior",), missing=missing,
            cap=p.cold_start_confidence_cap)

    def _onset_prediction(self, delay_min: float, sample_count: int, *,
                          learned: float, fallback: bool, bucket: str,
                          reasons: tuple, missing: list,
                          context_level: str = "L4",
                          outdoor_factor: float = 1.0,
                          general_n: int = 0,
                          fallback_path: Optional[list] = None,
                          cap: Optional[float] = None,
                          dispersion: float = 0.0) -> Prediction:
        """Build a Prediction with full context metadata encoded in reason_codes.

        reason_codes carries structured context info so callers can inspect:
          context_level:{L1|L2|L3|L4|L5}
          context_key:{bucket}
          match_quality:{factor:.2f}
          general_n:{count}
          fallback_path:{comma-separated tried levels}
          pre_conf:{raw confidence before outdoor factor}
          post_conf:{confidence after outdoor factor}
        """
        p = self._params
        delay_min = float(delay_min)
        if not is_finite(delay_min):
            delay_min = p.cold_start_prior_min
            fallback = True
        delay_min = clamp(delay_min, p.min_onset_delay_min, p.max_onset_delay_min)
        pre_conf = self._confidence_value(sample_count, fallback, dispersion, delay_min)
        post_conf = clamp(pre_conf * outdoor_factor, 0.0, 1.0)
        if cap is not None:
            post_conf = min(post_conf, cap)
        fp_str = ",".join(fallback_path) if fallback_path else "none"
        meta_codes = (
            f"context_level:{context_level}",
            f"context_key:{bucket}",
            f"match_quality:{outdoor_factor:.2f}",
            f"general_n:{general_n}",
            f"fallback_path:{fp_str}",
            f"pre_conf:{pre_conf:.3f}",
            f"post_conf:{post_conf:.3f}",
        )
        return Prediction(
            prediction_type=PredictionType.ONSET_DELAY,
            values={"onset_delay": round(delay_min, 2)},
            units={"onset_delay": "min"},
            confidence=post_conf,
            reliability=(self._state.aggregate_reliability if not fallback else 0.3),
            model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION,
            prior_contribution=1.0 - learned,
            learned_contribution=learned,
            fallback_used=fallback,
            evidence_count=sample_count,
            bucket=bucket,
            confidence_cap=cap,
            cap_reasons=("cold_start",) if cap is not None else (),
            missing_evidence=tuple(missing),
            reason_codes=meta_codes + tuple(reasons),
        )

    # -- confidence -----------------------------------------------------

    def _confidence_value(self, sample_count: int, fallback: bool,
                          dispersion: float = 0.0, delay_min: float = 1.0) -> float:
        if fallback or sample_count <= 0:
            return min(0.2, self._params.cold_start_confidence_cap)
        p = self._params
        coverage = min(sample_count / p.full_confidence_samples, 1.0)
        delay = max(abs(delay_min), 1e-6)
        spread_penalty = clamp(1.0 - (dispersion / delay), 0.3, 1.0)
        rel = max(self._state.aggregate_reliability, 0.1)
        return clamp(coverage * spread_penalty * rel, 0.0, 1.0)

    def confidence(self, *, now: Optional[str] = None) -> ConfidenceContribution:
        g = self._state.general
        reasons: list[str] = []
        if not g.has_evidence:
            reasons.append("cold_start")
        if g.sample_count < self._params.bucket_min_samples:
            reasons.append("few_samples")
        value = self._confidence_value(g.sample_count, not g.has_evidence,
                                       g.dispersion, g.onset_delay_min)
        freshness, status = freshness_from_age(now, self._state.last_update_ts)
        return ConfidenceContribution(value=value, evidence_count=g.sample_count,
                                      reasons=tuple(reasons), freshness=freshness, status=status)

    # -- diagnostics / export -------------------------------------------

    def diagnostics(self) -> OnsetDelayDiagnostics:
        s = self._state
        p = self._params
        return OnsetDelayDiagnostics(
            general_onset_delay_min=s.general.onset_delay_min,
            bucket_delays={k: b.onset_delay_min for k, b in s.buckets.items()},
            sample_counts={"general": s.general.sample_count,
                           **{k: b.sample_count for k, b in s.buckets.items()}},
            effective_sample_counts={"general": s.general.effective_n,
                                     **{k: b.effective_n for k, b in s.buckets.items()}},
            dispersion={"general": s.general.dispersion,
                        **{k: b.dispersion for k, b in s.buckets.items()}},
            confidence=self.confidence().value,
            last_update_ts=s.last_update_ts,
            rejection_counts=dict(s.rejection_counts),
            outlier_counts=dict(s.outlier_counts),
            dedup_count=s.dedup_count,
            fallback_used=not s.general.has_evidence,
            cold_start_prior_min=p.cold_start_prior_min,
            model_version=MODEL_VERSION,
            parameter_version=PARAMETER_VERSION,
        )

    def export(self, scope: ExportScope) -> Mapping[str, Any]:
        s = self._state
        if scope is ExportScope.SUPPORT:
            return {
                "general_onset_delay_min": s.general.onset_delay_min,
                "sample_count": s.general.sample_count,
                "confidence": self.confidence().value,
                "fallback": not s.general.has_evidence,
                "cold_start_prior_min": self._params.cold_start_prior_min,
                "model_version": MODEL_VERSION,
                "parameter_version": PARAMETER_VERSION,
                "rejection_summary": dict(s.rejection_counts),
            }
        if scope is ExportScope.RESEARCH:
            return {
                "samples": [
                    {
                        "onset_delay_min": x.onset_delay_min,
                        "start_temp_c": x.start_temp_c,
                        "target_temp_c": x.target_temp_c,
                        "bucket": x.bucket,
                        "time_bucket": x.time_bucket,
                        "reliability": x.episode_reliability,
                        "data_quality": x.data_quality,
                    }
                    for x in s.recent_samples
                ],
                "model_version": MODEL_VERSION,
            }
        return {"unsupported": "raw_export_not_provided_by_model"}

    # -- lifecycle -------------------------------------------------------

    def reset(self) -> None:
        self._state = self._empty_state()

    def rebuild(self, raw: Any = None,
                episodes: Optional[Sequence[Any]] = None) -> ModelRebuildResult:
        """Deterministically rebuild from a sequence of HeatingEpisode or
        OnsetDelayRebuildItem. Result equals sequential updates of same items."""
        self.reset()
        raw_items = list(episodes or [])
        items: list[tuple[Any, Optional[OnsetDelayUpdateContext]]] = []
        for it in raw_items:
            if isinstance(it, OnsetDelayRebuildItem):
                items.append((it.episode, it.context))
            else:
                items.append((it, None))
        items.sort(key=lambda pair: (
            _iso(getattr(pair[0], "start_ts", None)) or "",
            getattr(pair[0], "episode_id", "")))
        accepted = rejected = duplicates = 0
        rejection_counts: dict[str, int] = {}
        for episode, context in items:
            res = self.update(episode, context)
            if res.accepted:
                accepted += 1
            else:
                rejected += 1
                code = res.confounder_flags[0] if res.confounder_flags else "unknown"
                rejection_counts[code] = rejection_counts.get(code, 0) + 1
                if code == OnsetDelayRejection.DUPLICATE_EPISODE.value:
                    duplicates += 1
        return ModelRebuildResult(
            processed_count=len(items), accepted_count=accepted, rejected_count=rejected,
            duplicate_count=duplicates, error_count=0, rejection_counts=rejection_counts,
            started_from_prior=True, deterministic=True)

    # -- serialization ---------------------------------------------------

    def serialize_state(self) -> Mapping[str, Any]:
        s = self._state
        return {
            "model_version": MODEL_VERSION,
            "parameter_version": PARAMETER_VERSION,
            "learning_zone_id": s.learning_zone_id,
            "general": _bucket_dict(s.general),
            "buckets": {k: _bucket_dict(b) for k, b in s.buckets.items()},
            "processed_ids": list(s.processed_ids),
            "aggregate_reliability": s.aggregate_reliability,
            "last_update_ts": s.last_update_ts,
            "rejection_counts": dict(s.rejection_counts),
            "outlier_counts": dict(s.outlier_counts),
            "dedup_count": s.dedup_count,
            "recent_samples": [_sample_dict(x) for x in s.recent_samples],
        }

    def validate_state(self, state: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        if not isinstance(state, dict):
            return ["state is not a mapping"]
        if state.get("model_version") != MODEL_VERSION:
            errors.append("model_version mismatch")
        if state.get("learning_zone_id") != self._zone:
            errors.append("learning_zone_id mismatch")
        for key in ("general",):
            b = state.get(key, {})
            for fld in ("onset_delay_min", "effective_n", "dispersion"):
                v = b.get(fld)
                if v is not None and not is_finite(v):
                    errors.append(f"non-finite {key}.{fld}")
        return errors

    def deserialize_state(self, state: Mapping[str, Any]) -> None:
        errors = self.validate_state(state)
        if errors:
            raise ValueError(f"invalid onset-delay state: {errors}")
        buckets = {k: _bucket_from(b) for k, b in state.get("buckets", {}).items()}
        recent = tuple(_sample_from(x) for x in state.get("recent_samples", []))
        self._state = OnsetDelayState(
            learning_zone_id=state["learning_zone_id"],
            general=_bucket_from(state["general"]),
            buckets=buckets,
            processed_ids=tuple(state.get("processed_ids", [])),
            recent_samples=recent,
            aggregate_reliability=state.get("aggregate_reliability", 0.0),
            last_update_ts=state.get("last_update_ts"),
            rejection_counts=dict(state.get("rejection_counts", {})),
            outlier_counts=dict(state.get("outlier_counts", {})),
            dedup_count=state.get("dedup_count", 0),
        )

    def migrate_state(self, old_model_version: int, old_parameter_version: int,
                      state: Mapping[str, Any]) -> Mapping[str, Any]:
        if old_model_version == MODEL_VERSION:
            return state
        raise ValueError(
            f"no migration path from model_version {old_model_version} to {MODEL_VERSION}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_bucket(bucket: OnsetDelayBucket,
                  delay_min: float, weight: float) -> OnsetDelayBucket:
    upd = robust_ema_update(bucket.onset_delay_min, bucket.effective_n,
                            bucket.dispersion, delay_min, weight)
    return OnsetDelayBucket(
        key=bucket.key, onset_delay_min=upd.value, effective_n=upd.effective_n,
        dispersion=upd.dispersion, sample_count=bucket.sample_count + 1)


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


def _bucket_dict(b: OnsetDelayBucket) -> dict:
    return {
        "key": b.key,
        "onset_delay_min": b.onset_delay_min,
        "effective_n": b.effective_n,
        "dispersion": b.dispersion,
        "sample_count": b.sample_count,
        "last_update_ts": b.last_update_ts,
    }


def _bucket_from(d: Mapping[str, Any]) -> OnsetDelayBucket:
    return OnsetDelayBucket(
        key=d["key"],
        onset_delay_min=d["onset_delay_min"],
        effective_n=d["effective_n"],
        dispersion=d["dispersion"],
        sample_count=d["sample_count"],
        last_update_ts=d.get("last_update_ts"),
    )


def _sample_dict(x: OnsetDelaySample) -> dict:
    return {
        "source_episode_id": x.source_episode_id,
        "learning_zone_id": x.learning_zone_id,
        "onset_delay_min": x.onset_delay_min,
        "episode_duration_s": x.episode_duration_s,
        "start_temp_c": x.start_temp_c,
        "target_temp_c": x.target_temp_c,
        "episode_reliability": x.episode_reliability,
        "effective_weight": x.effective_weight,
        "bucket": x.bucket,
        "time_bucket": x.time_bucket,
        "outdoor_band": x.outdoor_band,
        "profile_id": x.profile_id,
        "builder_version": x.builder_version,
        "classifier_version": x.classifier_version,
        "data_quality": x.data_quality,
        "reason_codes": list(x.reason_codes),
    }


def _sample_from(d: Mapping[str, Any]) -> OnsetDelaySample:
    return OnsetDelaySample(
        source_episode_id=d["source_episode_id"],
        learning_zone_id=d["learning_zone_id"],
        onset_delay_min=d["onset_delay_min"],
        episode_duration_s=d["episode_duration_s"],
        start_temp_c=d["start_temp_c"],
        target_temp_c=d["target_temp_c"],
        episode_reliability=d["episode_reliability"],
        effective_weight=d["effective_weight"],
        bucket=d["bucket"],
        time_bucket=d.get("time_bucket"),
        outdoor_band=d.get("outdoor_band"),
        profile_id=d.get("profile_id"),
        builder_version=d["builder_version"],
        classifier_version=d["classifier_version"],
        data_quality=d["data_quality"],
        reason_codes=tuple(d.get("reason_codes", [])),
    )


# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------

def onset_delay_model_definition():
    """Build the registry ModelDefinition for OnsetDelay (no auto-registration)."""
    from ..registry import ModelDefinition

    return ModelDefinition(
        model_name=MODEL_NAME,
        model_factory=lambda: OnsetDelayModel("__definition__"),
        model_version=MODEL_VERSION,
        parameter_version=PARAMETER_VERSION,
        consumed_raw_tracks=(),
        consumed_episode_types=(EpisodeType.HEATING,),
        supported_prediction_types=(PredictionType.ONSET_DELAY,),
        required_features=("indoor_temp",),
        optional_features=("outdoor_temp", "time_bucket", "device_profile"),
        required_capability_flags=(),
        regime_dependent=True,
        advisory=False,
        control_relevant=True,
        rebuildable=True,
        can_be_not_available=False,
        min_trv_only=True,
    )
