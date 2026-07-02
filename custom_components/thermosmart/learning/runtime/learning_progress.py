"""Pure, calibrated LE2 learning-progress calculation (0.0-100.0 %).

Replaces the previous additive per-model saturation formula, which could
produce an artificial plateau: a single 25%-weighted model (heat_rate,
heat_loss, or outcome) reaching its own saturation threshold alone — while
every other core model stayed at zero — contributed exactly its full weight
(25.0) to the raw score, and the maturity-cap layer did not clamp that value
down for several realistic zone states (e.g. outcome_updates in [8, 20) allows
up to 35%). Because "outcome" episodes complete far more often than
heat_rate/heat_loss/afterheat transitions in many real installations, several
zones independently reaching outcome-only saturation produced the exact same
25.0% value — regardless of situation diversity, regime coverage, data
quality, or confounder contamination. That is the root cause of the "several
zones stuck at exactly 25%" symptom this module was built to fix.

Design: six weighted components (summing to 1.0), each derived only from data
the models already track (no invented fields), followed by a confounder
penalty multiplier:

    data_volume_score              total accepted episode-driven updates,
                                    log-scaled across ALL models together
                                    (never saturates from one model alone)
    thermal_regime_coverage_score  fraction of core regime models
                                    (heat_rate/heat_loss/afterheat/outcome)
                                    with a MEANINGFUL amount of data (not
                                    just "any"), continuous — not a hard cap
    situation_diversity_score      distinct populated buckets across
                                    heat_rate/heat_loss/afterheat diagnostics
                                    (real outdoor-temperature/heating-duration/
                                    device-profile buckets each model already
                                    tracks internally)
    clean_episode_score            accepted vs. (accepted + rejected +
                                    outlier) ratio, averaged across active
                                    models
    outcome_validation_score       outcome sample count (saturating) AND
                                    outcome data quality (timeout/overshoot/
                                    general_data_quality) — reuses the exact
                                    thresholds the previous quality-cap layer
                                    used, expressed continuously
    model_confidence_score         average of each model's OWN confidence()
                                    value, but ONLY for models with at least
                                    one accepted episode-driven update (a
                                    bootstrap/prior-only model never
                                    contributes here — preserves the original
                                    "bootstrap/priors excluded" guarantee)

confounder_penalty then dampens the weighted sum multiplicatively (capped at
50% reduction), derived from the same rejection/outlier counts.

Regime-breadth ceiling (added after initial calibration review): the six
components above can still overweight a single well-saturated area — e.g. a
zone with only "outcome" episodes, or only "heat_rate" episodes, reached
~35-37% purely from data_volume + clean_episode + model_confidence, none of
which require breadth across regimes. A zone understanding only ONE regime
area (or having very low situation diversity, or no outcome validation yet)
should not read as "roughly a third understood". Three independent ceilings
are computed from the same real signals (regime breadth, situation
diversity, outcome availability) and combined via min(); the tightest one
is applied. The ceiling is enforced as a smooth asymptotic soft-cap
(threshold + width * (1 - exp(-excess/width))), NOT a hard min()-clamp:
scores at/below (ceiling - width) are untouched, and scores above approach
the ceiling but can mathematically never equal it exactly. This preserves
relative differentiation between differently-saturated zones (e.g. good vs.
poor outcome quality) instead of collapsing them onto one shared number —
which would just relocate the original "several zones stuck at exactly X%"
symptom rather than removing it.

recency_score is intentionally NOT computed in this step: doing so honestly
would require historical progress snapshots (stability over time), which are
not currently persisted anywhere. It is exposed in the result attributes as
``None`` with an explicit gap note rather than being faked from
last_update_ts recency (which measures a different thing).

No HA imports. No control modification. Pure and defensive: malformed/missing
input for any one model is skipped, never raises.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

# Core regime models whose combined coverage represents "does ThermoSmart
# understand this zone's heating AND cooling behaviour", not just one facet.
_CORE_MODELS: tuple[str, ...] = ("heat_rate", "heat_loss", "afterheat", "outcome")

# Models with real, model-tracked situation buckets (outdoor temperature /
# heating-duration / device-profile) usable as a genuine diversity signal.
_DIVERSITY_MODELS: tuple[str, ...] = ("heat_rate", "heat_loss", "afterheat")

# Minimum accepted updates for a core model to count as "covered" — a single
# lucky update should not count the same as sustained coverage.
_MIN_UPDATES_FOR_COVERAGE = 2

# Total accepted episode-driven updates (across ALL models) that saturates
# data_volume_score to 1.0. Roughly the sum of the previous per-model
# saturation thresholds (10+10+8+8+10=46), rounded up.
_DATA_VOLUME_SATURATION_TOTAL = 50

# Distinct populated situation buckets (summed across the three diversity
# models) that saturates situation_diversity_score to 1.0. Deliberately
# modest — these buckets are model-internal and don't require any optional
# sensor, so a TRV-only zone with real outdoor-temperature variation over
# time can still reach this.
_DIVERSITY_BUCKET_SATURATION = 6

# Outcome sample count that saturates the count portion of
# outcome_validation_score to 1.0.
_OUTCOME_COUNT_SATURATION = 20

# Component weights — sum to 1.0. Coverage carries the largest single share
# specifically because breadth-across-regimes is what the old formula lacked;
# confidence carries the smallest share so a bootstrap-adjacent value can
# never dominate (and is fully excluded for models with zero real updates).
_WEIGHT_DATA_VOLUME = 0.20
_WEIGHT_COVERAGE = 0.25
_WEIGHT_DIVERSITY = 0.15
_WEIGHT_CLEAN = 0.10
_WEIGHT_OUTCOME = 0.20
_WEIGHT_CONFIDENCE = 0.10

# Reused, unchanged from the previous quality-cap layer's own thresholds.
_QUALITY_GATE_MIN_OUTCOME_SAMPLES = 8
_QUALITY_UNSTABLE_TIMEOUT = 0.40
_QUALITY_UNSTABLE_OVERSHOOT = 0.35
_QUALITY_UNSTABLE_DATA_QUALITY = 0.30
_QUALITY_DEGRADED_TIMEOUT = 0.25
_QUALITY_DEGRADED_OVERSHOOT = 0.25
_QUALITY_DEGRADED_DATA_QUALITY = 0.45
_QUALITY_DEGRADED_PARTIAL_RATIO = 0.60

# Confounder penalty cap — heavy contamination can suppress up to half of the
# weighted score, never all of it (some trustworthy signal may still exist).
_MAX_CONFOUNDER_PENALTY = 0.5

# --- Regime-breadth ceiling constants -------------------------------------
# Below this many total accepted updates, missing outcome validation is
# treated as "not reached that stage yet" rather than "a real gap" — the
# ceiling stays generous so early learning can still grow moderately.
_EARLY_PHASE_TOTAL_THRESHOLD = 10

# Fixed percentage-point width of the soft-cap's transition zone. Scores at
# or below (ceiling - width) are returned unchanged; only scores above that
# are smoothly bent toward the ceiling. Kept as an absolute width (not
# proportional to the ceiling) so a high ceiling (e.g. 100, fully-covered
# zones) stays a no-op across the entire realistic range, while a low
# ceiling (e.g. 25, single-regime zones) still leaves enough room below the
# knee to keep quality/confidence differences visible instead of saturating
# everything against the ceiling.
_CAP_SOFTEN_WIDTH = 10.0

# (lower_bound_pct, stage_name) — human-readable learning_stage labels
# matching the requested calibration bands.
_STAGE_BANDS: tuple[tuple[float, str], ...] = (
    (0.0, "cold_start"),
    (5.0, "early_signal"),
    (15.0, "first_clean_episodes"),
    (30.0, "building_coverage"),
    (50.0, "broadening_coverage"),
    (70.0, "robust_repetition"),
    (90.0, "mature"),
)

_LEVEL_BANDS: tuple[tuple[float, str], ...] = (
    (0.0, "very_low"),
    (20.0, "low"),
    (45.0, "medium"),
    (70.0, "high"),
    (90.0, "very_high"),
)


def _band_label(pct: float, bands: tuple[tuple[float, str], ...]) -> str:
    label = bands[0][1]
    for lower, name in bands:
        if pct >= lower:
            label = name
        else:
            break
    return label


@dataclass(frozen=True)
class ModelSignal:
    """The subset of one model's real, already-tracked data this calculation
    uses. Every field is optional/defaulted so a missing or unavailable model
    contributes nothing rather than raising.
    """
    accepted_updates: int = 0
    sample_counts: Mapping[str, int] = None  # type: ignore[assignment]
    rejection_counts: Mapping[str, int] = None  # type: ignore[assignment]
    outlier_counts: Mapping[str, int] = None  # type: ignore[assignment]
    confidence_value: Optional[float] = None
    # Outcome-only quality signals.
    general_data_quality: Optional[float] = None
    timeout_rate: Optional[float] = None
    overshoot_rate: Optional[float] = None
    partial_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sample_counts is None:
            object.__setattr__(self, "sample_counts", {})
        if self.rejection_counts is None:
            object.__setattr__(self, "rejection_counts", {})
        if self.outlier_counts is None:
            object.__setattr__(self, "outlier_counts", {})


def _populated_bucket_count(signal: ModelSignal) -> int:
    try:
        return sum(1 for c in signal.sample_counts.values() if isinstance(c, (int, float)) and c > 0)
    except Exception:
        return 0


def _rejected_and_outliers(signal: ModelSignal) -> int:
    try:
        return sum(signal.rejection_counts.values()) + sum(signal.outlier_counts.values())
    except Exception:
        return 0


def cold_learning_progress_result(reason: str) -> dict:
    """Public, neutral 0%-progress attribute dict for a given blocker reason.

    Used both internally (missing data, calculation error) and by callers
    (e.g. LearningShadowController) that need the same neutral shape for a
    disabled/unavailable shadow, without duplicating the attribute set.
    """
    return {
        "data_source": "current_learning_engine",
        "progress_status": "cold_start",
        "learning_stage": "cold_start",
        "confidence_level": "very_low",
        "data_volume_score": 0.0,
        "clean_episode_score": 0.0,
        "thermal_regime_coverage_score": 0.0,
        "situation_diversity_score": 0.0,
        "outcome_validation_score": 0.0,
        "model_confidence_score": 0.0,
        "confounder_penalty": 0.0,
        "regime_cap_pct": 100.0,
        "recency_score": None,
        "recency_score_gap": "requires historical progress snapshots (not yet persisted)",
        "sample_count": 0,
        "clean_episode_count": 0,
        "heating_episode_count": 0,
        "afterheat_episode_count": 0,
        "cooling_episode_count": 0,
        "outcome_episode_count": 0,
        "outcome_quality": None,
        "quality_reason": None,
        "main_blocker": reason,
        "next_needed": (
            "at least one completed heating/afterheat/cooling/outcome episode"
            if reason == "no_completed_episodes_yet" else "diagnostics unavailable this cycle"
        ),
        "bootstrap_excluded": True,
        "reason": reason,
    }


def _regime_breadth_cap_pct(core_covered: int, diversity_ratio: float) -> float:
    """Ceiling from "how many of the 4 core regime areas are covered".

    A single covered area caps in [25, 30] — the exact fachlich ask ("nicht
    schon über ca. 30%"). Each additional covered area raises both the floor
    and the diversity-driven range, but full 4-area coverage removes this
    particular ceiling entirely (other ceilings below may still apply).
    """
    if core_covered <= 1:
        return 25.0 + 5.0 * diversity_ratio
    if core_covered == 2:
        return 50.0 + 20.0 * diversity_ratio
    if core_covered == 3:
        return 75.0 + 15.0 * diversity_ratio
    return 100.0


def _diversity_cap_pct(diversity_ratio: float) -> float:
    """Ceiling from situation diversity alone, independent of regime count.

    Even with all 4 core areas covered, many *identical* repeated situations
    (diversity_ratio near 0) must not read as fully understood — this caps
    that case regardless of how much raw volume/confidence accumulated.
    """
    return 30.0 + 70.0 * diversity_ratio


def _outcome_availability_cap_pct(outcome_updates: int, total_accepted: int) -> float:
    """Ceiling from whether real-world outcome validation exists at all.

    No outcome episodes yet: generous ceiling while total activity is still
    low (early learning phase, per the fachlich exception), stricter once a
    zone has accumulated real activity elsewhere without ever validating an
    outcome. Partial outcome evidence (some updates, below the quality-gate
    sample threshold) sits between "none" and "fully available".
    """
    if outcome_updates == 0:
        return 55.0 if total_accepted < _EARLY_PHASE_TOTAL_THRESHOLD else 35.0
    if outcome_updates < _QUALITY_GATE_MIN_OUTCOME_SAMPLES:
        return 70.0
    return 100.0


def _apply_regime_caps(raw_pct: float, cap_pct: float) -> float:
    """Smoothly bend ``raw_pct`` toward ``cap_pct`` without ever reaching it.

    Deliberately NOT a min()-clamp: a hard clamp would push every
    sufficiently-saturated scenario under a given ceiling onto the exact
    same output value — reproducing the original "several zones stuck at
    exactly X%" symptom under a new ceiling instead of removing it. The
    asymptotic form keeps differently-saturated zones distinguishable (e.g.
    good vs. poor outcome quality both under the same ceiling still differ)
    while still bounding runaway values.
    """
    threshold = max(0.0, cap_pct - _CAP_SOFTEN_WIDTH)
    if raw_pct <= threshold:
        return raw_pct
    excess = raw_pct - threshold
    return threshold + _CAP_SOFTEN_WIDTH * (1.0 - math.exp(-excess / _CAP_SOFTEN_WIDTH))


def compute_learning_progress(
    signals: Mapping[str, ModelSignal],
) -> tuple[float, dict]:
    """Compute (progress_pct, attributes) from real, per-model signals.

    ``signals`` maps model_name -> ModelSignal for whichever models are
    available (missing entries are treated as having zero data). Pure — never
    raises; malformed individual values are skipped defensively.
    """
    try:
        total_accepted = 0
        core_covered = 0
        diversity_buckets = 0
        clean_ratios: list[float] = []
        confidence_values: list[float] = []
        total_rejected_all = 0
        total_activity_all = 0

        for name in _CORE_MODELS:
            sig = signals.get(name) or ModelSignal()
            n = max(0, int(sig.accepted_updates or 0))
            total_accepted += n
            if n >= _MIN_UPDATES_FOR_COVERAGE:
                core_covered += 1

        if total_accepted == 0:
            # No accepted episode-driven update anywhere yet — there is
            # nothing to genuinely score. clean_episode_score's "no evidence
            # of contamination" default must not be mistaken for "clean data
            # exists"; force a true cold start instead of a false floor.
            return 0.0, cold_learning_progress_result("no_completed_episodes_yet")

        for name, sig in signals.items():
            n = max(0, int((sig.accepted_updates if sig else 0) or 0))
            if sig is None:
                continue
            rejected = _rejected_and_outliers(sig)
            activity = n + rejected
            total_rejected_all += rejected
            total_activity_all += activity
            if activity > 0:
                clean_ratios.append(n / activity)
            if n > 0 and sig.confidence_value is not None:
                try:
                    confidence_values.append(max(0.0, min(1.0, float(sig.confidence_value))))
                except Exception:
                    pass

        for name in _DIVERSITY_MODELS:
            sig = signals.get(name)
            if sig is not None:
                diversity_buckets += _populated_bucket_count(sig)

        outcome_sig = signals.get("outcome") or ModelSignal()
        outcome_updates = max(0, int(outcome_sig.accepted_updates or 0))

        data_volume_score = (
            min(1.0, math.log1p(total_accepted) / math.log1p(_DATA_VOLUME_SATURATION_TOTAL))
            if total_accepted > 0 else 0.0
        )
        thermal_regime_coverage_score = core_covered / len(_CORE_MODELS)
        situation_diversity_score = min(1.0, diversity_buckets / _DIVERSITY_BUCKET_SATURATION)
        clean_episode_score = (
            sum(clean_ratios) / len(clean_ratios) if clean_ratios else 1.0
        )

        # Outcome validation: count portion (saturating) modulated by quality.
        outcome_count_score = min(1.0, outcome_updates / _OUTCOME_COUNT_SATURATION)
        quality_factor = 1.0
        quality_reason: Optional[str] = None
        outcome_quality_val = (
            round(outcome_sig.general_data_quality, 3)
            if outcome_sig.general_data_quality is not None else None
        )
        if outcome_updates >= _QUALITY_GATE_MIN_OUTCOME_SAMPLES:
            t_rate = outcome_sig.timeout_rate or 0.0
            o_rate = outcome_sig.overshoot_rate or 0.0
            dq = outcome_sig.general_data_quality if outcome_sig.general_data_quality is not None else 1.0
            p_ratio = outcome_sig.partial_ratio or 0.0
            if t_rate > _QUALITY_UNSTABLE_TIMEOUT or o_rate > _QUALITY_UNSTABLE_OVERSHOOT \
                    or dq < _QUALITY_UNSTABLE_DATA_QUALITY:
                quality_factor = 0.5
                quality_reason = "unstable_outcomes"
            elif t_rate > _QUALITY_DEGRADED_TIMEOUT or o_rate > _QUALITY_DEGRADED_OVERSHOOT \
                    or dq < _QUALITY_DEGRADED_DATA_QUALITY or p_ratio > _QUALITY_DEGRADED_PARTIAL_RATIO:
                quality_factor = 0.75
                quality_reason = "degraded_outcome_quality"
        outcome_validation_score = outcome_count_score * quality_factor

        model_confidence_score = (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        )

        confounder_penalty = (
            min(_MAX_CONFOUNDER_PENALTY, total_rejected_all / total_activity_all)
            if total_activity_all > 0 else 0.0
        )

        raw_score = (
            _WEIGHT_DATA_VOLUME * data_volume_score
            + _WEIGHT_COVERAGE * thermal_regime_coverage_score
            + _WEIGHT_DIVERSITY * situation_diversity_score
            + _WEIGHT_CLEAN * clean_episode_score
            + _WEIGHT_OUTCOME * outcome_validation_score
            + _WEIGHT_CONFIDENCE * model_confidence_score
        )
        final_score = raw_score * (1.0 - confounder_penalty)
        pre_cap_pct = max(0.0, min(100.0, final_score * 100.0))

        regime_cap_pct = min(
            _regime_breadth_cap_pct(core_covered, situation_diversity_score),
            _diversity_cap_pct(situation_diversity_score),
            _outcome_availability_cap_pct(outcome_updates, total_accepted),
        )
        progress_pct = round(_apply_regime_caps(pre_cap_pct, regime_cap_pct), 1)

        # Determine the single most limiting factor among weighted components,
        # by how much of that component's own weight is still unrealised.
        component_gaps = {
            "insufficient_data_volume": _WEIGHT_DATA_VOLUME * (1.0 - data_volume_score),
            "insufficient_regime_coverage": _WEIGHT_COVERAGE * (1.0 - thermal_regime_coverage_score),
            "insufficient_situation_diversity": _WEIGHT_DIVERSITY * (1.0 - situation_diversity_score),
            "insufficient_clean_episodes": _WEIGHT_CLEAN * (1.0 - clean_episode_score),
            "insufficient_outcome_validation": _WEIGHT_OUTCOME * (1.0 - outcome_validation_score),
            "insufficient_model_confidence": _WEIGHT_CONFIDENCE * (1.0 - model_confidence_score),
        }
        if confounder_penalty > 0.05:
            main_blocker = "confounder_contamination"
        else:
            main_blocker = max(component_gaps, key=component_gaps.get)

        _next_needed = {
            "confounder_contamination": "reduce confounder exposure (window/manual overrides during heating)",
            "no_completed_episodes_yet": "at least one completed heating/afterheat/cooling/outcome episode",
            "insufficient_data_volume": "more accepted episode-driven model updates overall",
            "insufficient_regime_coverage": "episodes across more regimes (heating/afterheat/cooling/outcome)",
            "insufficient_situation_diversity": "episodes across a wider range of outdoor temperatures/situations",
            "insufficient_clean_episodes": "fewer confounded/rejected/outlier episodes relative to accepted ones",
            "insufficient_outcome_validation": "more accepted outcome episodes with stable (low timeout/overshoot) quality",
            "insufficient_model_confidence": "more repeated evidence for the models that already have some data",
        }[main_blocker]

        if progress_pct == 0.0:
            status = "cold_start"
        elif thermal_regime_coverage_score >= 0.99 and outcome_validation_score >= 0.99 \
                and confounder_penalty < 0.05:
            status = "learned"
        else:
            status = "learning"

        return progress_pct, {
            "data_source": "current_learning_engine",
            "progress_status": status,
            "learning_stage": _band_label(progress_pct, _STAGE_BANDS),
            "confidence_level": _band_label(progress_pct, _LEVEL_BANDS),
            "data_volume_score": round(data_volume_score, 3),
            "clean_episode_score": round(clean_episode_score, 3),
            "thermal_regime_coverage_score": round(thermal_regime_coverage_score, 3),
            "situation_diversity_score": round(situation_diversity_score, 3),
            "outcome_validation_score": round(outcome_validation_score, 3),
            "model_confidence_score": round(model_confidence_score, 3),
            "confounder_penalty": round(confounder_penalty, 3),
            # Transparency for the regime-breadth ceiling: the tightest of the
            # three real-signal-derived caps that bounded this cycle's score.
            # A value of 100.0 means no ceiling was binding this cycle.
            "regime_cap_pct": round(regime_cap_pct, 1),
            # recency_score is not genuinely computable yet: it would require
            # historical progress snapshots (stability over time), which are
            # not currently persisted anywhere. Exposed as None rather than
            # faked from last_update_ts recency (a different concept).
            "recency_score": None,
            "recency_score_gap": "requires historical progress snapshots (not yet persisted)",
            "sample_count": total_accepted,
            # clean_episode_count currently equals sample_count: every counted
            # update already passed its model's own acceptance gate, and a
            # granular "accepted-but-confounded" breakdown would require
            # per-episode persisted history (a separate, not-yet-wired step).
            "clean_episode_count": total_accepted,
            "heating_episode_count": signals.get("heat_rate", ModelSignal()).accepted_updates,
            "afterheat_episode_count": signals.get("afterheat", ModelSignal()).accepted_updates,
            "cooling_episode_count": signals.get("heat_loss", ModelSignal()).accepted_updates,
            "outcome_episode_count": outcome_updates,
            "outcome_quality": outcome_quality_val,
            "quality_reason": quality_reason,
            "main_blocker": main_blocker,
            "next_needed": _next_needed,
            "bootstrap_excluded": True,
            "reason": main_blocker,
        }
    except Exception:
        return 0.0, cold_learning_progress_result("calculation_error")
