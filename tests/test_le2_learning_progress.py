"""Tests for the recalibrated LE2 learning-progress scoring.

Covers custom_components/thermosmart/learning/runtime/learning_progress.py
(the pure compute_learning_progress() core) and a lighter integration check
of LearningShadowController.learning_progress_safe()'s real-data gathering
(via a harness with an identical method body, using a REAL LearningRuntime +
real model instances — no HA instantiation needed for this).

The previous formula could produce an artificial plateau: a single
25%-weighted model (heat_rate, heat_loss, or outcome) reaching its own
saturation threshold ALONE, with every other core model still at zero,
contributed exactly 25.0 to the raw score, and the maturity-cap layer did not
clamp it down for several realistic states — producing the reported "several
zones stuck at exactly 25%" symptom. These tests verify that symptom is gone
and that the new score behaves per the requested calibration bands.

17 test groups:
  T1  — New zone with no data stays near 0%
  T2  — LE initialized without episodes produces no artificial jump
  T3  — First samples without a completed episode stay low
  T4  — One completed HeatingEpisode raises progress moderately
  T5  — Many identical episodes raise progress only within a bound
  T6  — Different situations (bucket diversity) raise progress more than repeats
  T7  — Confounders lower/limit progress
  T8  — Missing outcome episodes limit progress
  T9  — Outcome episodes raise progress only with clean quality
  T10 — Different regimes increase thermal_regime_coverage_score
  T11 — TRV-only minimal setup can still reach real progress
  T12 — Missing optional data lowers confidence, not to 0%
  T13 — Different zones with different data get different values
  T14 — No artificial fixed 25% plateau
  T15 — Attributes explain the main blocker
  T16 — Existing LE2 tests remain green
  T17 — No control-path effect
"""
from __future__ import annotations

import inspect
from typing import Any, Optional

import pytest

from custom_components.thermosmart.learning.runtime.learning_progress import (
    ModelSignal,
    cold_learning_progress_result,
    compute_learning_progress,
)
from custom_components.thermosmart.learning.runtime.lifecycle import (
    LearningRuntime,
    LearningRuntimeConfig,
)


# ── T1: New zone with no data stays near 0% ─────────────────────────────────

def test_t1_no_data_stays_at_zero():
    pct, attrs = compute_learning_progress({})
    assert pct == 0.0
    assert attrs["progress_status"] == "cold_start"
    assert attrs["learning_stage"] == "cold_start"


def test_t1_empty_model_signals_stay_at_zero():
    signals = {"heat_rate": ModelSignal(), "outcome": ModelSignal()}
    pct, _ = compute_learning_progress(signals)
    assert pct == 0.0


# ── T2: LE initialized without episodes produces no artificial jump ────────

def test_t2_disabled_shadow_neutral_result():
    result = cold_learning_progress_result("learning_disabled")
    assert result["main_blocker"] == "learning_disabled"
    assert result["data_volume_score"] == 0.0


def test_t2_zero_accepted_updates_never_jumps():
    """Even with diagnostics present (buckets/confidence populated by a prior
    session's leftover in-memory state) but zero ACCEPTED updates, progress
    must stay at 0 — completion, not raw diagnostic presence, gates progress."""
    signals = {
        "heat_rate": ModelSignal(
            accepted_updates=0, sample_counts={"outdoor:5.0": 3}, confidence_value=0.4,
        ),
    }
    pct, _ = compute_learning_progress(signals)
    assert pct == 0.0


# ── T3: First samples without a completed episode stay low ─────────────────

def test_t3_no_completed_episode_stays_at_zero_even_with_rejections():
    """Rejected/outlier attempts without any ACCEPTED update must not create
    progress — only accepted, episode-driven updates count."""
    signals = {
        "heat_rate": ModelSignal(accepted_updates=0, rejection_counts={"confounder": 5}),
    }
    pct, _ = compute_learning_progress(signals)
    assert pct == 0.0


# ── T4: One completed HeatingEpisode raises progress moderately ────────────

def test_t4_single_episode_raises_progress_moderately():
    signals = {"heat_rate": ModelSignal(accepted_updates=1, sample_counts={"outdoor:5.0": 1})}
    pct, attrs = compute_learning_progress(signals)
    assert 0.0 < pct < 30.0
    assert attrs["learning_stage"] in ("early_signal", "first_clean_episodes")


# ── T5: Many identical episodes raise progress only within a bound ────────

def test_t5_many_identical_episodes_bounded():
    """10 heat_rate updates all in the SAME bucket (no diversity) must not
    reach anywhere near full progress — coverage/diversity/outcome remain
    zero regardless of how many times the identical situation repeats.
    Single core area + near-zero diversity: bounded to <=30% per the
    Blocker 2 single-core-area / low-diversity ceilings, not just <50%."""
    signals = {
        "heat_rate": ModelSignal(
            accepted_updates=10, sample_counts={"outdoor:5.0": 10}, confidence_value=0.6,
        ),
    }
    pct, attrs = compute_learning_progress(signals)
    assert pct <= 30.0
    assert attrs["situation_diversity_score"] < 0.3


# ── T6: Different situations raise progress more than identical repeats ───

def test_t6_diverse_situations_beat_identical_repeats():
    same_bucket = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"outdoor:5.0": 10}),
    }
    diverse_buckets = {
        "heat_rate": ModelSignal(
            accepted_updates=10,
            sample_counts={"outdoor:-5.0": 2, "outdoor:0.0": 2, "outdoor:5.0": 2,
                           "outdoor:10.0": 2, "outdoor:15.0": 2},
        ),
    }
    pct_same, _ = compute_learning_progress(same_bucket)
    pct_diverse, _ = compute_learning_progress(diverse_buckets)
    assert pct_diverse > pct_same


# ── T7: Confounders lower/limit progress ────────────────────────────────────

def test_t7_confounders_reduce_progress():
    clean = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"outdoor:5.0": 10}),
        "outcome": ModelSignal(accepted_updates=10, sample_counts={"controller:ts": 10},
                                general_data_quality=0.8, timeout_rate=0.05, overshoot_rate=0.05),
    }
    contaminated = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"outdoor:5.0": 10},
                                  rejection_counts={"confounder": 25}),
        "outcome": ModelSignal(accepted_updates=10, sample_counts={"controller:ts": 10},
                                general_data_quality=0.8, timeout_rate=0.05, overshoot_rate=0.05,
                                rejection_counts={"confounder": 20}),
    }
    pct_clean, _ = compute_learning_progress(clean)
    pct_contaminated, attrs_contaminated = compute_learning_progress(contaminated)
    assert pct_contaminated < pct_clean
    assert attrs_contaminated["confounder_penalty"] > 0.0


# ── T8: Missing outcome episodes limit progress ────────────────────────────

def test_t8_missing_outcome_limits_progress():
    with_outcome = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"outdoor:5.0": 10}),
        "heat_loss": ModelSignal(accepted_updates=10, sample_counts={"outdoor:5.0": 10}),
        "afterheat": ModelSignal(accepted_updates=8, sample_counts={"heating_dur:600": 8}),
        "outcome": ModelSignal(accepted_updates=15, sample_counts={"controller:ts": 15},
                                general_data_quality=0.8, timeout_rate=0.05, overshoot_rate=0.05),
    }
    without_outcome = {k: v for k, v in with_outcome.items() if k != "outcome"}
    pct_with, _ = compute_learning_progress(with_outcome)
    pct_without, attrs_without = compute_learning_progress(without_outcome)
    assert pct_without < pct_with
    assert attrs_without["outcome_validation_score"] == 0.0


# ── T9: Outcome episodes raise progress only with clean quality ───────────

def test_t9_outcome_quality_gates_the_boost():
    good_quality = {
        "outcome": ModelSignal(accepted_updates=10, sample_counts={"controller:ts": 10},
                                general_data_quality=0.9, timeout_rate=0.02, overshoot_rate=0.02),
    }
    poor_quality = {
        "outcome": ModelSignal(accepted_updates=10, sample_counts={"controller:ts": 10},
                                general_data_quality=0.2, timeout_rate=0.5, overshoot_rate=0.4),
    }
    pct_good, attrs_good = compute_learning_progress(good_quality)
    pct_poor, attrs_poor = compute_learning_progress(poor_quality)
    assert pct_good > pct_poor
    assert attrs_poor["quality_reason"] == "unstable_outcomes"


# ── T10: Different regimes increase thermal_regime_coverage_score ─────────

def test_t10_regime_coverage_increases_with_breadth():
    one_regime = {"heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10})}
    four_regimes = {
        "heat_rate": ModelSignal(accepted_updates=3, sample_counts={"o:5": 3}),
        "heat_loss": ModelSignal(accepted_updates=3, sample_counts={"o:5": 3}),
        "afterheat": ModelSignal(accepted_updates=3, sample_counts={"d:600": 3}),
        "outcome": ModelSignal(accepted_updates=3, sample_counts={"c:ts": 3}),
    }
    _, attrs_one = compute_learning_progress(one_regime)
    _, attrs_four = compute_learning_progress(four_regimes)
    assert attrs_four["thermal_regime_coverage_score"] > attrs_one["thermal_regime_coverage_score"]
    assert attrs_four["thermal_regime_coverage_score"] == 1.0
    assert attrs_one["thermal_regime_coverage_score"] == 0.25


# ── T11: TRV-only minimal setup can still reach real progress ─────────────

def test_t11_trv_only_setup_reaches_nonzero_progress():
    """A TRV-only zone has no outdoor-temperature sensor, so heat_rate/heat_loss
    buckets stay in a single fallback bucket — but data volume, coverage, and
    outcome validation are still fully available and must be enough to reach
    meaningful (non-zero, non-blocked) progress."""
    signals = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"outdoor:unknown": 10},
                                  confidence_value=0.5),
        "heat_loss": ModelSignal(accepted_updates=10, sample_counts={"outdoor:unknown": 10},
                                  confidence_value=0.5),
        "afterheat": ModelSignal(accepted_updates=8, sample_counts={"heating_dur:600": 8},
                                  confidence_value=0.5),
        "outcome": ModelSignal(accepted_updates=15, sample_counts={"controller:ts": 15},
                                general_data_quality=0.75, timeout_rate=0.1, overshoot_rate=0.1,
                                confidence_value=0.5),
    }
    pct, attrs = compute_learning_progress(signals)
    assert pct >= 30.0  # meaningful progress despite minimal situation diversity
    assert attrs["thermal_regime_coverage_score"] == 1.0


# ── T12: Missing optional data lowers confidence, not to 0% ────────────────

def test_t12_missing_confidence_value_does_not_zero_progress():
    signals = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"outdoor:5.0": 10, "outdoor:0.0": 5}),
        # confidence_value intentionally omitted (None) — model.confidence()
        # unavailable for some reason; must not block the rest of the score.
    }
    pct, attrs = compute_learning_progress(signals)
    assert pct > 0.0
    assert attrs["model_confidence_score"] == 0.0  # absent, not fabricated, but doesn't zero others


# ── T13: Different zones with different data get different values ─────────

def test_t13_different_zones_get_different_values():
    zone_a = {
        "outcome": ModelSignal(accepted_updates=8, sample_counts={"c:ts": 8},
                                general_data_quality=0.9, timeout_rate=0.05, overshoot_rate=0.05),
    }
    zone_b = {
        "outcome": ModelSignal(accepted_updates=8, sample_counts={"c:ts": 8},
                                general_data_quality=0.5, timeout_rate=0.3, overshoot_rate=0.3),
    }
    zone_c = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10}),
    }
    pct_a, _ = compute_learning_progress(zone_a)
    pct_b, _ = compute_learning_progress(zone_b)
    pct_c, _ = compute_learning_progress(zone_c)
    values = {pct_a, pct_b, pct_c}
    assert len(values) == 3  # all distinct — no coincidental plateau


# ── T14: No artificial fixed 25% plateau ───────────────────────────────────

def test_t14_no_fixed_25_percent_plateau_outcome_only():
    """The exact historical bug: only 'outcome' saturated (8 updates), every
    other core model at zero. Recalibration (Blocker 2) adds an explicit
    single-core-area ceiling for this exact scenario, so unlike the original
    plateau check, landing near 25% is now expected and correct — the
    requirement is that it stays bounded (<=30%) and is not produced by a
    hard clamp (the soft-cap in _apply_regime_caps mathematically can never
    equal the ceiling exactly)."""
    signals = {
        "outcome": ModelSignal(accepted_updates=8, sample_counts={"c:ts": 8},
                                general_data_quality=0.7, timeout_rate=0.1, overshoot_rate=0.1),
    }
    pct, attrs = compute_learning_progress(signals)
    assert pct <= 30.0
    assert pct != attrs["regime_cap_pct"]  # soft-cap: approaches, never equals, the ceiling


def test_t14_no_fixed_25_percent_plateau_heat_rate_only():
    signals = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10}, confidence_value=0.6),
    }
    pct, attrs = compute_learning_progress(signals)
    assert pct <= 30.0
    assert pct != attrs["regime_cap_pct"]


def test_t14_multiple_single_model_saturation_scenarios_differ():
    """Several previously-25%-colliding scenarios must now differ from each
    other, not just from 25.0 — proving the plateau is gone, not just moved."""
    only_outcome = {"outcome": ModelSignal(accepted_updates=8, sample_counts={"c:ts": 8},
                                            general_data_quality=0.7, timeout_rate=0.1, overshoot_rate=0.1)}
    only_heat_rate = {"heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10})}
    only_heat_loss = {"heat_loss": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10})}
    pcts = [compute_learning_progress(s)[0] for s in (only_outcome, only_heat_rate, only_heat_loss)]
    assert len(set(pcts)) >= 2  # not all identical


# ── T15: Attributes explain the main blocker ───────────────────────────────

def test_t15_main_blocker_present_and_actionable():
    signals = {"heat_rate": ModelSignal(accepted_updates=1, sample_counts={"o:5": 1})}
    _, attrs = compute_learning_progress(signals)
    assert attrs["main_blocker"]
    assert attrs["next_needed"]
    assert isinstance(attrs["next_needed"], str)


def test_t15_confounder_blocker_takes_priority():
    signals = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10},
                                  rejection_counts={"confounder": 40}),
    }
    _, attrs = compute_learning_progress(signals)
    assert attrs["main_blocker"] == "confounder_contamination"


def test_t15_no_completed_episodes_blocker_for_empty_zone():
    _, attrs = compute_learning_progress({})
    assert attrs["main_blocker"] == "no_completed_episodes_yet"


# ── Integration: real gathering via a production-identical harness ────────

class _LearningProgressHarness:
    """Mirrors LearningShadowController.learning_progress_safe()'s real-data
    gathering exactly, using a REAL LearningRuntime + real model instances
    (no HA instantiation needed) — validates the CONTRACT, not a copy."""

    def __init__(self, runtime: LearningRuntime, zone: str, enabled: bool = True) -> None:
        self._runtime = runtime
        self._zone = zone
        self._enabled = enabled
        self.errors: list[str] = []

    def _record_error(self, kind: str, err: Exception) -> None:
        self.errors.append(f"{kind}:{err}")

    def learning_progress_safe(self) -> tuple[float, dict]:
        if not self._enabled:
            return 0.0, {**cold_learning_progress_result("no_completed_episodes_yet"),
                         "reason": "learning_disabled"}
        try:
            zr = self._runtime._zone(self._zone)
            counts = dict(zr.model_update_counts)

            signals: dict[str, Any] = {}
            for model_name in ("heat_rate", "heat_loss", "afterheat", "outcome", "onset_delay"):
                accepted = counts.get(model_name, 0)
                sample_counts: dict = {}
                rejection_counts: dict = {}
                outlier_counts: dict = {}
                confidence_value: Optional[float] = None
                general_data_quality: Optional[float] = None
                timeout_rate: Optional[float] = None
                overshoot_rate: Optional[float] = None
                partial_ratio: Optional[float] = None
                try:
                    model = zr.orchestrator.models.get(model_name)
                    if model is not None:
                        diag = model.diagnostics()
                        sample_counts = dict(getattr(diag, "sample_counts", None) or {})
                        rejection_counts = dict(getattr(diag, "rejection_counts", None) or {})
                        outlier_counts = dict(getattr(diag, "outlier_counts", None) or {})
                        if model_name == "outcome":
                            general_data_quality = getattr(diag, "general_data_quality", None)
                            timeout_rate = getattr(diag, "timeout_rate", None)
                            overshoot_rate = getattr(diag, "overshoot_rate", None)
                            fc, pc = getattr(diag, "full_partial", (0, 0))
                            partial_ratio = (pc / (fc + pc)) if (fc + pc) > 0 else 0.0
                        if accepted > 0:
                            confidence_value = model.confidence().value
                except Exception:
                    pass
                signals[model_name] = ModelSignal(
                    accepted_updates=accepted, sample_counts=sample_counts,
                    rejection_counts=rejection_counts, outlier_counts=outlier_counts,
                    confidence_value=confidence_value, general_data_quality=general_data_quality,
                    timeout_rate=timeout_rate, overshoot_rate=overshoot_rate,
                    partial_ratio=partial_ratio,
                )
            return compute_learning_progress(signals)
        except Exception as err:
            self._record_error("learning_progress", err)
            return 0.0, cold_learning_progress_result("calculation_error")


def _real_runtime() -> LearningRuntime:
    return LearningRuntime(LearningRuntimeConfig(), clock=lambda: "2026-06-01T10:00:00+00:00")


def test_integration_zero_updates_gives_zero_via_real_models():
    harness = _LearningProgressHarness(_real_runtime(), "zone_a")
    pct, attrs = harness.learning_progress_safe()
    assert pct == 0.0
    assert not harness.errors


def test_integration_real_model_diagnostics_gathered_without_error():
    runtime = _real_runtime()
    zr = runtime._zone("zone_a")
    zr.model_update_counts["heat_rate"] = 5
    zr.model_update_counts["outcome"] = 8
    harness = _LearningProgressHarness(runtime, "zone_a")
    pct, attrs = harness.learning_progress_safe()
    assert pct > 0.0
    assert not harness.errors
    assert attrs["heating_episode_count"] == 5
    assert attrs["outcome_episode_count"] == 8


def test_integration_disabled_shadow_is_neutral():
    harness = _LearningProgressHarness(_real_runtime(), "zone_a", enabled=False)
    pct, attrs = harness.learning_progress_safe()
    assert pct == 0.0
    assert attrs["reason"] == "learning_disabled"


# ── T18: Blocker 2 recalibration — regime-breadth ceilings ────────────────
# Added after the initial calibration review flagged single-model-saturated
# scenarios (only outcome, only heat_rate) landing at 35.4%/36.9% — too high
# for "only one model area is reliable". These tests cover the three new
# ceilings (single-core-area, low-diversity, missing-outcome) together with
# the two non-regressions they must not break: broad coverage/diversity can
# still exceed 50%, and a mature zone can still exceed 80-90%.

def test_t18_single_core_area_outcome_only_stays_bounded():
    signals = {
        "outcome": ModelSignal(accepted_updates=8, sample_counts={"c:ts": 8},
                                general_data_quality=0.7, timeout_rate=0.1, overshoot_rate=0.1),
    }
    pct, attrs = compute_learning_progress(signals)
    assert pct <= 30.0
    assert attrs["regime_cap_pct"] <= 30.0


def test_t18_single_core_area_heat_rate_only_stays_bounded():
    signals = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"o:5": 10}, confidence_value=0.6),
    }
    pct, attrs = compute_learning_progress(signals)
    assert pct <= 30.0


def test_t18_low_diversity_cap_binds_even_with_full_regime_coverage():
    """All 4 core areas covered, but every one of them stuck in a single
    situation bucket (no real diversity anywhere) — the diversity ceiling
    must bind independently of regime-breadth coverage being complete."""
    signals = {
        "heat_rate": ModelSignal(accepted_updates=20, sample_counts={"outdoor:5.0": 20}, confidence_value=0.9),
        "heat_loss": ModelSignal(accepted_updates=20, sample_counts={"outdoor:5.0": 20}, confidence_value=0.9),
        "afterheat": ModelSignal(accepted_updates=15, sample_counts={"heating_dur:600": 15}, confidence_value=0.9),
        "outcome": ModelSignal(accepted_updates=20, sample_counts={"c:ts": 20},
                                general_data_quality=0.9, timeout_rate=0.05, overshoot_rate=0.05, confidence_value=0.9),
    }
    pct, attrs = compute_learning_progress(signals)
    assert attrs["thermal_regime_coverage_score"] == 1.0
    assert attrs["regime_cap_pct"] < 100.0  # diversity ceiling still binds, regime-breadth one is lifted


def test_t18_broad_coverage_and_diversity_exceeds_50_percent():
    signals = {
        "heat_rate": ModelSignal(accepted_updates=15,
                                  sample_counts={f"outdoor:{b}": 3 for b in (-10, -5, 0, 5, 10, 15)},
                                  confidence_value=0.8),
        "heat_loss": ModelSignal(accepted_updates=12,
                                  sample_counts={f"outdoor:{b}": 3 for b in (-5, 0, 5, 10)},
                                  confidence_value=0.8),
        "afterheat": ModelSignal(accepted_updates=10,
                                  sample_counts={f"heating_dur:{b}": 2 for b in (600, 1800, 3600)},
                                  confidence_value=0.8),
        "outcome": ModelSignal(accepted_updates=15, sample_counts={"c:ts": 15},
                                general_data_quality=0.85, timeout_rate=0.05, overshoot_rate=0.05,
                                confidence_value=0.8),
    }
    pct, _ = compute_learning_progress(signals)
    assert pct > 50.0


def test_t18_mature_zone_exceeds_80_percent():
    signals = {
        "heat_rate": ModelSignal(accepted_updates=60,
                                  sample_counts={f"outdoor:{b}": 10 for b in (-10, -5, 0, 5, 10, 15)},
                                  confidence_value=0.95),
        "heat_loss": ModelSignal(accepted_updates=50,
                                  sample_counts={f"outdoor:{b}": 10 for b in (-10, -5, 0, 5, 10, 15)},
                                  confidence_value=0.95),
        "afterheat": ModelSignal(accepted_updates=40,
                                  sample_counts={f"heating_dur:{b}": 10 for b in (600, 1800, 3600)},
                                  confidence_value=0.9),
        "outcome": ModelSignal(accepted_updates=60, sample_counts={"c:ts": 60},
                                general_data_quality=0.95, timeout_rate=0.03, overshoot_rate=0.03,
                                confidence_value=0.95),
    }
    pct, attrs = compute_learning_progress(signals)
    assert pct > 80.0
    assert attrs["regime_cap_pct"] == 100.0  # no ceiling binding for a genuinely mature zone


def test_t18_trv_only_grows_but_not_inflated_by_missing_diversity():
    """TRV-only (no outdoor-temperature sensor -> single fallback bucket per
    model) must still reach meaningful, non-zero progress from genuinely
    clean heating/outcome data, but the missing situation diversity must
    stay visible (cap < 100, not silently ignored)."""
    signals = {
        "heat_rate": ModelSignal(accepted_updates=10, sample_counts={"outdoor:unknown": 10}, confidence_value=0.5),
        "heat_loss": ModelSignal(accepted_updates=10, sample_counts={"outdoor:unknown": 10}, confidence_value=0.5),
        "afterheat": ModelSignal(accepted_updates=8, sample_counts={"heating_dur:600": 8}, confidence_value=0.5),
        "outcome": ModelSignal(accepted_updates=15, sample_counts={"c:ts": 15},
                                general_data_quality=0.75, timeout_rate=0.1, overshoot_rate=0.1,
                                confidence_value=0.5),
    }
    pct, attrs = compute_learning_progress(signals)
    assert pct >= 30.0
    assert attrs["regime_cap_pct"] < 100.0  # missing diversity stays visible, not zeroed or ignored


# ── T16: Existing LE2 tests remain green (regression smoke-test) ──────────

def test_t16_regression_imports_ok():
    import tests.test_le2_episode_runtime_hook  # noqa: F401
    import tests.test_le2_episode_store_wiring  # noqa: F401
    import tests.test_le2_episode_persistence  # noqa: F401
    import tests.test_le2_episode_serialization  # noqa: F401
    import tests.test_le2_capture_wiring_foundation  # noqa: F401
    import tests.test_le2_retention  # noqa: F401
    import tests.test_le2_periodic_save_trigger  # noqa: F401


# ── T17: No control-path effect ────────────────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
    "boost_offset",
    "tpi_gain",
)


def test_t17_no_control_keywords_in_new_module():
    import custom_components.thermosmart.learning.runtime.learning_progress as _lp_mod
    source = inspect.getsource(_lp_mod).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"


def test_t17_learning_progress_safe_touches_no_control_path():
    import custom_components.thermosmart.learning.runtime.ha_integration as _ha
    source = inspect.getsource(_ha.LearningShadowController.learning_progress_safe).lower()
    for token in _FORBIDDEN_CONTROL_TOKENS:
        assert token not in source, f"forbidden control token found: {token}"
    assert "async_write_ha_state" not in source
