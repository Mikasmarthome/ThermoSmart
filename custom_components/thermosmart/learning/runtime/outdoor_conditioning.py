"""Phase D2-B — pure outdoor-conditioning gate (no HA, no I/O, no control).

Turns a completed HeatRate / HeatLoss episode's D2-A ``OutdoorEvidence`` sidecar
into a typed, episode-bound ``HeatRateUpdateContext`` / ``HeatLossUpdateContext``
— but ONLY when the evidence is trustworthy enough to attribute to a single
outdoor bucket. This is LEARNING-ONLY: the context is consumed by the models'
existing conditioned-bucket update path; it never touches predictions or control.

Acceptance gate (ALL must hold, else no context -> unconditioned update):
  1. evidence exists and has a usable finite time-weighted mean;
  2. quality is DataQuality.OK (STALE / UNKNOWN rejected — see freshness note);
  3. source did not change within the episode;
  4. temporal_coverage >= 0.5;
  5. usable_count >= 2;
  6. min and max both fall in the SAME existing outdoor bucket (no cross-band
     ambiguity), which also guarantees the mean shares that band;
  7. the evidence is internally consistent (finite, min <= mean <= max).

``conditioning_ready`` is advisory and deliberately NOT used as the gate.

Freshness note: ``DataQuality.OK`` describes value validity, NOT recency.
``OutdoorEvidence.freshness_known`` is always False, so OK never asserts the
reading is fresh; the temporal-coverage gate (3)/(4) is what bounds observation
density. We require OK (not STALE) for conditioned attribution precisely because
a band assignment cannot be validated against an unknowable freshness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..contracts import DataQuality
from ..models.heat_loss import HeatLossUpdateContext
from ..models.heat_rate import HeatRateUpdateContext

# Canonical outdoor bucket edges. MUST match HeatRateParameters.outdoor_buckets
# and HeatLossParameters.outdoor_buckets (asserted by a test); duplicated here so
# this gate stays a pure, dependency-light helper.
OUTDOOR_BUCKET_EDGES: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0)

MIN_TEMPORAL_COVERAGE = 0.5
MIN_USABLE_COUNT = 2

# model_name values this gate applies to (all others -> no conditioning)
_CONDITIONED_MODELS = ("heat_rate", "heat_loss")


def _bucket_index(value: float, edges: Sequence[float]) -> int:
    """Identical banding to HeatRate/HeatLoss models' private ``_bucket_index``."""
    idx = 0
    for edge in edges:
        if value >= edge:
            idx += 1
        else:
            break
    return idx


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    reason: str                       # "ok" or the first failing criterion
    band: Optional[int] = None        # the outdoor band when accepted


def evaluate_gate(evidence: Any,
                  edges: Sequence[float] = OUTDOOR_BUCKET_EDGES) -> GateResult:
    """Pure gate decision for one episode's outdoor evidence. Never raises."""
    ev = evidence
    if ev is None:
        return GateResult(False, "no_evidence")
    mean = getattr(ev, "time_weighted_mean_c", None)
    if not getattr(ev, "has_usable_evidence", False) or mean is None \
            or not _finite(mean):
        return GateResult(False, "no_usable_mean")
    if getattr(ev, "quality", None) is not DataQuality.OK:
        return GateResult(False, "quality_not_ok")
    if getattr(ev, "source_changed", False):
        return GateResult(False, "source_changed")
    cov = getattr(ev, "temporal_coverage", 0.0)
    if not _finite(cov) or cov < MIN_TEMPORAL_COVERAGE:
        return GateResult(False, "low_temporal_coverage")
    if getattr(ev, "usable_count", 0) < MIN_USABLE_COUNT:
        return GateResult(False, "too_few_usable_samples")
    lo = getattr(ev, "min_c", None)
    hi = getattr(ev, "max_c", None)
    if lo is None or hi is None or not _finite(lo) or not _finite(hi):
        return GateResult(False, "incomplete_envelope")
    # internal consistency: ordered envelope containing the mean
    if lo > hi or not (lo - 1e-9 <= mean <= hi + 1e-9):
        return GateResult(False, "inconsistent_envelope")
    # single-band: min and max (hence the mean) share one existing bucket
    band = _bucket_index(mean, edges)
    if _bucket_index(lo, edges) != band or _bucket_index(hi, edges) != band:
        return GateResult(False, "cross_band_spread")
    return GateResult(True, "ok", band=band)


def build_context(model_name: str, episode: Any, evidence: Any,
                  edges: Sequence[float] = OUTDOOR_BUCKET_EDGES) -> Optional[Any]:
    """Return a typed update context for an accepted HeatRate/HeatLoss episode,
    or ``None`` when the model is not conditioned or the evidence is rejected.

    Pure and defensive: any unexpected error yields ``None`` (an unconditioned
    update), never an exception into the runtime cycle.
    """
    try:
        if model_name not in _CONDITIONED_MODELS:
            return None
        episode_id = getattr(episode, "episode_id", None)
        if not isinstance(episode_id, str) or not episode_id:
            return None
        gate = evaluate_gate(evidence, edges)
        if not gate.accepted:
            return None
        source = getattr(evidence, "source", None)
        source_refs = (source,) if isinstance(source, str) and source else ()
        mean = float(evidence.time_weighted_mean_c)
        if model_name == "heat_rate":
            return HeatRateUpdateContext(
                source_episode_id=episode_id, outdoor_temperature_c=mean,
                device_profile_key=None, source_refs=source_refs, quality=DataQuality.OK)
        return HeatLossUpdateContext(
            source_episode_id=episode_id, outdoor_temperature_c=mean,
            indoor_outdoor_delta_c=None,     # D2-B: normalized coefficient stays off
            device_profile_key=None, source_refs=source_refs, quality=DataQuality.OK)
    except Exception:
        return None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
