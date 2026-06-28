"""SimulationResult — structured artifact for Item 6 validation reports.

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
    service_call_count: int = 0
    window_open_steps: int = 0

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
            heat_rate_rel_error = final_snap.heat_rate_relative_error()
            learned_heat_rate = final_snap.learned_heat_rate_c_per_h
            gt_heat_rate = final_snap.gt_heat_rate_c_per_h
            heat_rate_upd = final_snap.heat_rate_update_count
            heat_loss_upd = final_snap.heat_loss_update_count
            onset_delay_upd = final_snap.onset_delay_update_count
            afterheat_upd = final_snap.afterheat_update_count
            outcome_upd = final_snap.outcome_update_count

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
            service_call_count=summary.service_call_count,
            window_open_steps=getattr(summary, "window_open_steps", 0),
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
        """Human-readable text summary for test output / Abschlussbericht."""
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
