"""SimulationResult and ABRunResult — structured artifacts for validation.

SimulationResult: one simulation run (Item 6).
ABRunResult: side-by-side comparison of baseline and adaptive runs (Item 8).

Each long-run simulation produces one SimulationResult.  The result captures
all quantitative acceptance criteria from the Item 6 specification in a single
dataclass so tests can assert on the result without re-implementing calculations.

Usage::

    result = SimulationResult.from_summary(summary, cfg)
    result.check_acceptance()   # raises AssertionError on violations
    print(result.report())      # human-readable text summary
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from tests.simulation.metrics import MetricsSummary, ModelLearningSnapshot


# ── Acceptance limits ────────────────────────────────────────────────────────

ACCEPTANCE_COLD_FRACTION_MAX = 0.08        # ≤ 8% of steps below safety floor
ACCEPTANCE_OVERSHOOT_FRACTION_MAX = 0.12   # ≤ 12% of steps above ceiling
ACCEPTANCE_P95_MS_MAX = 50.0               # p95 per-step time ≤ 50 ms
ACCEPTANCE_P99_MS_MAX = 100.0              # p99 per-step time ≤ 100 ms
ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY = 50.0  # blob growth ≤ 50 KB/day


@dataclass
class SimulationResult:
    """Structured acceptance artifact for one simulation run."""

    scenario_id: str
    profile_name: str
    seed: int
    duration_days: float
    step_s: float
    total_steps: int
    n_restarts: int

    # ── Comfort metrics ───────────────────────────────────────────────────
    comfort_fraction: float = 0.0
    cold_fraction: float = 0.0
    overshoot_fraction: float = 0.0
    overshoot_area_k_h: float = 0.0      # ∫(T_room − T_comfort)+ dt [K·h]
    undershoot_area_k_h: float = 0.0     # ∫(T_comfort_floor − T_room)+ dt [K·h]
    service_call_count: int = 0
    window_open_steps: int = 0
    heating_on_steps: int = 0
    total_heating_wh: float = 0.0

    # ── Learning quality (from model snapshots) ───────────────────────────
    model_update_total_final: int = 0
    final_control_applied: int = 0
    final_control_fallback: int = 0
    final_fallback_rate: float = 0.0
    final_blob_bytes: int = 0
    final_save_count: int = 0
    # Per-model update counts at end of run
    heat_rate_update_count_final: int = 0
    heat_loss_update_count_final: int = 0
    onset_delay_update_count_final: int = 0
    afterheat_update_count_final: int = 0
    outcome_update_count_final: int = 0

    # Learned vs ground-truth at end of run (None = model never converged)
    final_heat_rate_error: Optional[float] = None
    final_heat_rate_relative_error: Optional[float] = None
    final_learned_heat_rate: Optional[float] = None
    gt_heat_rate: Optional[float] = None
    # Supply-delay learning quality (from final model snapshot)
    final_learned_supply_delay_s: Optional[float] = None
    gt_supply_delay_s: Optional[float] = None
    supply_delay_error_s: Optional[float] = None          # |learned − gt|
    supply_delay_relative_error: Optional[float] = None   # |learned − gt| / gt

    # ── Performance ───────────────────────────────────────────────────────
    mean_step_ms: float = 0.0
    p95_step_ms: float = 0.0
    p99_step_ms: float = 0.0
    max_step_ms: float = 0.0

    # ── Storage growth ────────────────────────────────────────────────────
    storage_growth_kb_per_day: float = 0.0
    initial_blob_bytes: int = 0

    # ── Acceptance ────────────────────────────────────────────────────────
    acceptance_passed: bool = False
    failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @classmethod
    def from_summary(
        cls,
        summary: MetricsSummary,
        *,
        scenario_id: str,
        profile_name: str,
        seed: int,
        duration_days: float,
        step_s: float,
    ) -> "SimulationResult":
        """Build a SimulationResult from a MetricsSummary."""
        total_steps = summary.total_steps

        final_snap: Optional[ModelLearningSnapshot] = summary.final_model_snapshot()
        first_snap: Optional[ModelLearningSnapshot] = (
            summary.model_snapshots[0] if summary.model_snapshots else None
        )

        # Performance
        perf = summary.perf
        mean_ms = perf.mean_ms
        p95_ms = perf.p95_ms
        p99_ms = perf.p99_ms
        max_ms = perf.max_ms

        # Learning quality from final snapshot
        model_update_total = 0
        final_control_applied = 0
        final_control_fallback = 0
        final_blob_bytes = 0
        final_save_count = 0
        heat_rate_error = None
        heat_rate_rel_error = None
        learned_heat_rate = None
        gt_heat_rate = None
        learned_supply_delay_s = None
        gt_supply_delay_s = None
        supply_delay_error_s = None
        supply_delay_rel_err = None

        heat_rate_upd = 0
        heat_loss_upd = 0
        onset_delay_upd = 0
        afterheat_upd = 0
        outcome_upd = 0

        if final_snap is not None:
            model_update_total = final_snap.model_update_total
            final_control_applied = final_snap.control_applied
            final_control_fallback = final_snap.control_fallback
            final_blob_bytes = final_snap.blob_bytes
            final_save_count = final_snap.save_count
            heat_rate_error = final_snap.heat_rate_error()
            heat_rate_rel_err = final_snap.heat_rate_relative_error()
            heat_rate_rel_error = heat_rate_rel_err
            learned_heat_rate = final_snap.learned_heat_rate_c_per_h
            gt_heat_rate = final_snap.gt_heat_rate_c_per_h
            heat_rate_upd = final_snap.heat_rate_update_count
            heat_loss_upd = final_snap.heat_loss_update_count
            onset_delay_upd = final_snap.onset_delay_update_count
            afterheat_upd = final_snap.afterheat_update_count
            outcome_upd = final_snap.outcome_update_count
            # Supply delay
            learned_supply_delay_s = final_snap.learned_supply_delay_s
            gt_supply_delay_s = final_snap.gt_supply_delay_s
            if learned_supply_delay_s is not None and gt_supply_delay_s is not None:
                supply_delay_error_s = abs(learned_supply_delay_s - gt_supply_delay_s)
                if gt_supply_delay_s > 0:
                    supply_delay_rel_err = supply_delay_error_s / gt_supply_delay_s

        initial_blob = first_snap.blob_bytes if first_snap is not None else 0
        blob_growth_bytes = max(0, final_blob_bytes - initial_blob)
        storage_growth_kb_per_day = (
            blob_growth_bytes / 1024.0 / max(1.0, duration_days)
        )

        # Fallback rate
        total_control = final_control_applied + final_control_fallback
        fallback_rate = (
            final_control_fallback / total_control if total_control > 0 else 0.0
        )

        result = cls(
            scenario_id=scenario_id,
            profile_name=profile_name,
            seed=seed,
            duration_days=duration_days,
            step_s=step_s,
            total_steps=total_steps,
            n_restarts=summary.restart_count,
            comfort_fraction=summary.comfort_fraction,
            cold_fraction=summary.cold_fraction,
            overshoot_fraction=summary.overshoot_fraction,
            overshoot_area_k_h=getattr(summary, "overshoot_area_k_h", 0.0),
            undershoot_area_k_h=getattr(summary, "undershoot_area_k_h", 0.0),
            service_call_count=summary.service_call_count,
            window_open_steps=getattr(summary, "window_open_steps", 0),
            heating_on_steps=getattr(summary, "heating_on_steps", 0),
            total_heating_wh=summary.total_heating_wh,
            model_update_total_final=model_update_total,
            final_control_applied=final_control_applied,
            final_control_fallback=final_control_fallback,
            final_fallback_rate=fallback_rate,
            final_blob_bytes=final_blob_bytes,
            final_save_count=final_save_count,
            heat_rate_update_count_final=heat_rate_upd,
            heat_loss_update_count_final=heat_loss_upd,
            onset_delay_update_count_final=onset_delay_upd,
            afterheat_update_count_final=afterheat_upd,
            outcome_update_count_final=outcome_upd,
            final_heat_rate_error=heat_rate_error,
            final_heat_rate_relative_error=heat_rate_rel_error,
            final_learned_heat_rate=learned_heat_rate,
            gt_heat_rate=gt_heat_rate,
            final_learned_supply_delay_s=learned_supply_delay_s,
            gt_supply_delay_s=gt_supply_delay_s,
            supply_delay_error_s=supply_delay_error_s,
            supply_delay_relative_error=supply_delay_rel_err,
            mean_step_ms=mean_ms,
            p95_step_ms=p95_ms,
            p99_step_ms=p99_ms,
            max_step_ms=max_ms,
            storage_growth_kb_per_day=storage_growth_kb_per_day,
            initial_blob_bytes=initial_blob,
        )
        result._run_acceptance_checks(summary)
        return result

    def _run_acceptance_checks(self, summary: MetricsSummary) -> None:
        """Populate failures/warnings and set acceptance_passed."""
        failures = []
        warnings = []

        # NaN/Inf in model values
        if not summary.has_no_nan_inf():
            failures.append("NaN or Inf found in learned model values")

        # Cold-fraction bound
        if self.cold_fraction > ACCEPTANCE_COLD_FRACTION_MAX:
            failures.append(
                f"cold_fraction {self.cold_fraction:.3f} > {ACCEPTANCE_COLD_FRACTION_MAX}"
            )

        # Overshoot bound
        if self.overshoot_fraction > ACCEPTANCE_OVERSHOOT_FRACTION_MAX:
            warnings.append(
                f"overshoot_fraction {self.overshoot_fraction:.3f} > {ACCEPTANCE_OVERSHOOT_FRACTION_MAX}"
            )

        # Performance
        if self.p95_step_ms > ACCEPTANCE_P95_MS_MAX:
            warnings.append(
                f"p95 step time {self.p95_step_ms:.1f} ms > {ACCEPTANCE_P95_MS_MAX} ms"
            )
        if self.p99_step_ms > ACCEPTANCE_P99_MS_MAX:
            failures.append(
                f"p99 step time {self.p99_step_ms:.1f} ms > {ACCEPTANCE_P99_MS_MAX} ms"
            )

        # Storage growth
        if self.storage_growth_kb_per_day > ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY:
            failures.append(
                f"storage growth {self.storage_growth_kb_per_day:.1f} KB/day "
                f"> {ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY} KB/day"
            )

        # Heat rate learning: must have > 0 updates in adaptive runs with 30+ days
        if (self.model_update_total_final > 0  # adaptive run (baseline has 0)
                and self.duration_days >= 30
                and self.heat_rate_update_count_final == 0):
            warnings.append(
                f"heat_rate model received 0 updates after {self.duration_days:.0f} days "
                f"(total updates={self.model_update_total_final} — may be all non-heat_rate)"
            )

        # Heat rate relative error (warn if > 40% after 90+ days, > 60% after 30+ days)
        if self.final_heat_rate_relative_error is not None:
            if self.duration_days >= 90 and self.final_heat_rate_relative_error > 0.40:
                warnings.append(
                    f"learned heat rate relative error {self.final_heat_rate_relative_error:.2%} > 40% "
                    f"after {self.duration_days:.0f} days (net GT={self.gt_heat_rate:.3f} C/h, "
                    f"learned={self.final_learned_heat_rate})"
                )
            elif self.duration_days >= 30 and self.final_heat_rate_relative_error > 0.60:
                warnings.append(
                    f"learned heat rate relative error {self.final_heat_rate_relative_error:.2%} > 60% "
                    f"after {self.duration_days:.0f} days"
                )

        self.failures = failures
        self.warnings = warnings
        self.acceptance_passed = len(failures) == 0

    def check_acceptance(self) -> None:
        """Raise AssertionError listing all failures if acceptance_passed is False."""
        if not self.acceptance_passed:
            msg = f"SimulationResult [{self.scenario_id}] FAILED:\n" + "\n".join(
                f"  - {f}" for f in self.failures
            )
            if self.warnings:
                msg += "\nWarnings:\n" + "\n".join(f"  WARN: {w}" for w in self.warnings)
            raise AssertionError(msg)

    def report(self) -> str:
        """Human-readable text summary for test output."""
        lines = [
            f"SimulationResult: {self.scenario_id}",
            f"  Profile:           {self.profile_name}",
            f"  Seed:              {self.seed}",
            f"  Duration:          {self.duration_days:.0f} days "
            f"({self.total_steps} steps x {self.step_s:.0f}s)",
            f"  Restarts:          {self.n_restarts}",
            f"  Comfort fraction:  {self.comfort_fraction:.3f}",
            f"  Cold fraction:     {self.cold_fraction:.3f} "
            f"(limit <= {ACCEPTANCE_COLD_FRACTION_MAX})",
            f"  Overshoot:         {self.overshoot_fraction:.3f} "
            f"(limit <= {ACCEPTANCE_OVERSHOOT_FRACTION_MAX})",
            f"  Service calls:     {self.service_call_count}",
            f"  Window-open steps: {self.window_open_steps}",
            f"  Control applied:   {self.final_control_applied}  "
            f"fallback: {self.final_control_fallback} "
            f"({self.final_fallback_rate:.1%})",
            f"  Final blob:        {self.final_blob_bytes / 1024:.1f} KB  "
            f"(saves: {self.final_save_count})",
            f"  Storage growth:    {self.storage_growth_kb_per_day:.2f} KB/day",
            f"  Step timing:       mean={self.mean_step_ms:.1f} ms  "
            f"p95={self.p95_step_ms:.1f}  p99={self.p99_step_ms:.1f}  "
            f"max={self.max_step_ms:.1f}",
        ]
        lines.append(
            f"  Model updates:     total={self.model_update_total_final}"
            f"  heat_rate={self.heat_rate_update_count_final}"
            f"  heat_loss={self.heat_loss_update_count_final}"
            f"  onset={self.onset_delay_update_count_final}"
            f"  afterheat={self.afterheat_update_count_final}"
            f"  outcome={self.outcome_update_count_final}"
        )
        if self.gt_heat_rate is not None:
            rel_err_str = (
                f"{self.final_heat_rate_relative_error:.2%}"
                if self.final_heat_rate_relative_error is not None else "N/A"
            )
            lines.append(
                f"  Heat rate (net GT):{self.gt_heat_rate:.3f} C/h  "
                f"learned:{self.final_learned_heat_rate}  "
                f"abs_err:{self.final_heat_rate_error}  "
                f"rel_err:{rel_err_str}  "
                f"updates:{self.heat_rate_update_count_final}"
            )
        if self.acceptance_passed:
            lines.append("  ACCEPTANCE: PASSED")
        else:
            lines.append("  ACCEPTANCE: FAILED")
            for f in self.failures:
                lines.append(f"    FAIL: {f}")
        for w in self.warnings:
            lines.append(f"    WARN: {w}")
        return "\n".join(lines)


# ── A/B run result ────────────────────────────────────────────────────────────

# Maximum cold_fraction regression allowed: adaptive may be at most this much
# worse than baseline before it constitutes a safety violation.
AB_COLD_REGRESSION_MAX = 0.05      # +5 percentage points

# Maximum dispatch regression: adaptive service calls ≤ baseline × this factor
AB_SERVICE_CALL_RATIO_MAX = 2.0

# Maximum heating energy regression (absolute Wh)
AB_HEATING_WH_RATIO_MAX = 1.50     # adaptive may use at most 50% more energy

# Maximum overshoot area regression
AB_OVERSHOOT_AREA_RATIO_MAX = 1.30  # 30% more overshoot area triggers a warning


@dataclass
class ABRunResult:
    """Side-by-side comparison of a baseline and an adaptive simulation run.

    Deltas are defined as (adaptive − baseline); negative values mean adaptive
    improved over baseline (e.g. delta_cold_fraction < 0 → less cold time).

    Acceptance rules:
    - HARD FAILURE: adaptive cold_fraction > baseline + AB_COLD_REGRESSION_MAX
    - HARD FAILURE: adaptive p99 > 100 ms (same as individual run limit)
    - HARD FAILURE: adaptive storage_growth > 50 KB/day
    - WARNING: adaptive service_calls > baseline × AB_SERVICE_CALL_RATIO_MAX
    - WARNING: adaptive overshoot_area > baseline × AB_OVERSHOOT_AREA_RATIO_MAX
    - WARNING: NaN or Inf in adaptive model values
    """

    scenario_id: str
    duration_days: float
    seed: int
    baseline: SimulationResult
    adaptive: SimulationResult

    # ── Comfort deltas (adaptive − baseline; negative = adaptive improved) ──
    delta_cold_fraction: float = 0.0
    delta_overshoot_fraction: float = 0.0
    delta_overshoot_area_k_h: float = 0.0
    delta_undershoot_area_k_h: float = 0.0
    delta_comfort_fraction: float = 0.0

    # ── Effort deltas (adaptive − baseline; negative = adaptive more efficient)
    delta_service_calls: int = 0
    delta_heating_on_steps: int = 0
    delta_heating_wh: float = 0.0

    # ── Learning quality (adaptive only; baseline has none) ─────────────────
    supply_delay_error_s: Optional[float] = None
    supply_delay_relative_error: Optional[float] = None
    heat_rate_relative_error: Optional[float] = None
    onset_delay_update_count: int = 0
    afterheat_update_count: int = 0
    final_control_applied: int = 0
    final_control_fallback: int = 0
    final_fallback_rate: float = 0.0
    model_update_total: int = 0

    # ── A/B acceptance ────────────────────────────────────────────────────────
    ab_acceptance_passed: bool = False
    ab_failures: list = field(default_factory=list)
    ab_warnings: list = field(default_factory=list)

    @classmethod
    def from_pair(
        cls,
        baseline: SimulationResult,
        adaptive: SimulationResult,
        *,
        scenario_id: str = "",
    ) -> "ABRunResult":
        """Build an ABRunResult from a (baseline, adaptive) pair of SimulationResults."""
        sid = scenario_id or f"{adaptive.scenario_id}_vs_{baseline.scenario_id}"
        result = cls(
            scenario_id=sid,
            duration_days=adaptive.duration_days,
            seed=adaptive.seed,
            baseline=baseline,
            adaptive=adaptive,
            delta_cold_fraction=adaptive.cold_fraction - baseline.cold_fraction,
            delta_overshoot_fraction=adaptive.overshoot_fraction - baseline.overshoot_fraction,
            delta_overshoot_area_k_h=adaptive.overshoot_area_k_h - baseline.overshoot_area_k_h,
            delta_undershoot_area_k_h=adaptive.undershoot_area_k_h - baseline.undershoot_area_k_h,
            delta_comfort_fraction=adaptive.comfort_fraction - baseline.comfort_fraction,
            delta_service_calls=adaptive.service_call_count - baseline.service_call_count,
            delta_heating_on_steps=adaptive.heating_on_steps - baseline.heating_on_steps,
            delta_heating_wh=adaptive.total_heating_wh - baseline.total_heating_wh,
            supply_delay_error_s=adaptive.supply_delay_error_s,
            supply_delay_relative_error=adaptive.supply_delay_relative_error,
            heat_rate_relative_error=adaptive.final_heat_rate_relative_error,
            onset_delay_update_count=adaptive.onset_delay_update_count_final,
            afterheat_update_count=adaptive.afterheat_update_count_final,
            final_control_applied=adaptive.final_control_applied,
            final_control_fallback=adaptive.final_control_fallback,
            final_fallback_rate=adaptive.final_fallback_rate,
            model_update_total=adaptive.model_update_total_final,
        )
        result._run_ab_checks()
        return result

    def _run_ab_checks(self) -> None:
        """Populate ab_failures / ab_warnings and set ab_acceptance_passed."""
        failures = []
        warnings = []

        # Cold-fraction regression (hard)
        if self.delta_cold_fraction > AB_COLD_REGRESSION_MAX:
            failures.append(
                f"cold_fraction regression: adaptive={self.adaptive.cold_fraction:.3f} "
                f"baseline={self.baseline.cold_fraction:.3f} "
                f"delta={self.delta_cold_fraction:+.3f} > {AB_COLD_REGRESSION_MAX}"
            )

        # Performance (hard — same limits as individual run)
        if self.adaptive.p99_step_ms > ACCEPTANCE_P99_MS_MAX:
            failures.append(
                f"adaptive p99 step time {self.adaptive.p99_step_ms:.1f} ms "
                f"> {ACCEPTANCE_P99_MS_MAX} ms"
            )

        # Storage growth (hard)
        if self.adaptive.storage_growth_kb_per_day > ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY:
            failures.append(
                f"adaptive storage growth {self.adaptive.storage_growth_kb_per_day:.1f} KB/day "
                f"> {ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY} KB/day"
            )

        # NaN/Inf in adaptive model values
        if not self.adaptive.acceptance_passed and any(
            "NaN" in f or "Inf" in f for f in self.adaptive.failures
        ):
            failures.append("NaN or Inf found in adaptive learned model values")

        # Service call ratio (warning)
        b_calls = self.baseline.service_call_count
        if b_calls > 0 and self.adaptive.service_call_count > b_calls * AB_SERVICE_CALL_RATIO_MAX:
            warnings.append(
                f"adaptive service calls {self.adaptive.service_call_count} "
                f"> baseline {b_calls} x {AB_SERVICE_CALL_RATIO_MAX:.0f} "
                f"(ratio={self.adaptive.service_call_count / b_calls:.2f})"
            )

        # Overshoot area ratio (warning)
        b_area = self.baseline.overshoot_area_k_h
        if b_area > 0 and self.adaptive.overshoot_area_k_h > b_area * AB_OVERSHOOT_AREA_RATIO_MAX:
            warnings.append(
                f"adaptive overshoot area {self.adaptive.overshoot_area_k_h:.2f} K*h "
                f"> baseline {b_area:.2f} K*h x {AB_OVERSHOOT_AREA_RATIO_MAX:.2f} "
                f"(ratio={self.adaptive.overshoot_area_k_h / b_area:.2f})"
            )

        # Heating energy ratio (warning — not hard, since adaptive may heat more
        # to achieve better comfort)
        b_wh = self.baseline.total_heating_wh
        if b_wh > 0 and self.adaptive.total_heating_wh > b_wh * AB_HEATING_WH_RATIO_MAX:
            warnings.append(
                f"adaptive heating energy {self.adaptive.total_heating_wh:.0f} Wh "
                f"> baseline {b_wh:.0f} Wh x {AB_HEATING_WH_RATIO_MAX:.2f} "
                f"(ratio={self.adaptive.total_heating_wh / b_wh:.2f})"
            )

        self.ab_failures = failures
        self.ab_warnings = warnings
        self.ab_acceptance_passed = len(failures) == 0

    def check_ab_acceptance(self) -> None:
        """Raise AssertionError if A/B acceptance failed."""
        if not self.ab_acceptance_passed:
            msg = f"ABRunResult [{self.scenario_id}] FAILED:\n" + "\n".join(
                f"  - {f}" for f in self.ab_failures
            )
            if self.ab_warnings:
                msg += "\nWarnings:\n" + "\n".join(f"  WARN: {w}" for w in self.ab_warnings)
            raise AssertionError(msg)

    def report(self) -> str:
        """Human-readable A/B comparison report."""
        b, a = self.baseline, self.adaptive
        lines = [
            f"ABRunResult: {self.scenario_id}",
            f"  Duration:          {self.duration_days:.0f} days | Seed: {self.seed}",
            "",
            "  Comfort (baseline -> adaptive, delta):",
            f"    cold_fraction:     {b.cold_fraction:.3f} -> {a.cold_fraction:.3f}"
            f"  d={self.delta_cold_fraction:+.3f}",
            f"    overshoot_frac:    {b.overshoot_fraction:.3f} -> {a.overshoot_fraction:.3f}"
            f"  d={self.delta_overshoot_fraction:+.3f}",
            f"    overshoot_area:    {b.overshoot_area_k_h:.2f} -> {a.overshoot_area_k_h:.2f} K*h"
            f"  d={self.delta_overshoot_area_k_h:+.2f}",
            f"    undershoot_area:   {b.undershoot_area_k_h:.2f} -> {a.undershoot_area_k_h:.2f} K*h"
            f"  d={self.delta_undershoot_area_k_h:+.2f}",
            f"    comfort_fraction:  {b.comfort_fraction:.3f} -> {a.comfort_fraction:.3f}"
            f"  d={self.delta_comfort_fraction:+.3f}",
            "",
            "  Effort (baseline -> adaptive, delta):",
            f"    service_calls:     {b.service_call_count} -> {a.service_call_count}"
            f"  d={self.delta_service_calls:+d}",
            f"    heating_on_steps:  {b.heating_on_steps} -> {a.heating_on_steps}"
            f"  d={self.delta_heating_on_steps:+d}",
            f"    heating_wh:        {b.total_heating_wh:.0f} -> {a.total_heating_wh:.0f} Wh"
            f"  d={self.delta_heating_wh:+.0f}",
            "",
            "  Learning quality (adaptive):",
            f"    model_updates:     {self.model_update_total}",
            f"    control_applied:   {self.final_control_applied}"
            f"  fallback: {self.final_control_fallback} ({self.final_fallback_rate:.1%})",
            f"    onset_delay_upd:   {self.onset_delay_update_count}",
            f"    afterheat_upd:     {self.afterheat_update_count}",
        ]
        if self.supply_delay_error_s is not None:
            rel = (
                f"{self.supply_delay_relative_error:.1%}"
                if self.supply_delay_relative_error is not None else "N/A"
            )
            lines.append(
                f"    supply_delay_err:  {self.supply_delay_error_s:.0f}s  rel={rel}"
            )
        if self.heat_rate_relative_error is not None:
            lines.append(
                f"    heat_rate_rel_err: {self.heat_rate_relative_error:.2%}"
            )
        lines.append("")
        if self.ab_acceptance_passed:
            lines.append("  A/B ACCEPTANCE: PASSED")
        else:
            lines.append("  A/B ACCEPTANCE: FAILED")
            for f in self.ab_failures:
                lines.append(f"    FAIL: {f}")
        for w in self.ab_warnings:
            lines.append(f"    WARN: {w}")
        return "\n".join(lines)
