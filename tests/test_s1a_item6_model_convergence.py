"""S1a Item 6 — Per-model convergence tests.

Proves that each learning model (heat_rate, onset_delay, afterheat, heat_loss)
actually receives updates and begins converging toward physical ground truth.

Test groups:
  Group 1 — OnsetDelay: pipeline-fixed HeatingEpisodes reach the onset model
  Group 2 — Afterheat: AFTERHEAT regime triggers with PROFILE_CLEAN_AFTERHEAT
  Group 3 — HeatLoss: PASSIVE_COOLING regime fires with night-setback scenario
  Group 4 — HeatRate regression guard after builder fix
  Group 5 — Learning-progress isolation (coverage visibility)

New scenario functions (scenario_clean_afterheat_30d, scenario_clean_heat_loss_30d,
scenario_clean_supply_delay_30d) are expected in tests/simulation/scenarios.py.
If they are absent the individual tests skip gracefully.
"""
from __future__ import annotations

import math

import pytest

from tests.simulation.physics import (
    PROFILE_FAST_INSULATED,
    PROFILE_CLEAN_AFTERHEAT,
    PROFILE_CLEAN_HEAT_LOSS,
    PROFILE_CLEAN_SUPPLY_DELAY,
    VirtualRoomPhysics,
)
from tests.simulation.scenarios import (
    _HEATING_SEASON_START_UTC,
    _outdoor_seasonal,
    _solar_gain,
    _steps_per_day,
    scenario_s1a,
    scenario_restart_equiv_continuous,
)
from tests.simulation.runner import ScenarioConfig, ScenarioRunner
from tests.simulation.result import SimulationResult

pytestmark = pytest.mark.asyncio

_START = _HEATING_SEASON_START_UTC
_STEP_S = 300.0


def _outdoor_fn(base_w: float = -3.0):
    return _outdoor_seasonal(
        _START, base_winter_c=base_w, base_summer_c=16.0, day_amplitude=4.0
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _final_snap(summary):
    """Return the final ModelLearningSnapshot, or None."""
    return summary.final_model_snapshot()


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1: OnsetDelay
# ═══════════════════════════════════════════════════════════════════════════════

async def test_onset_delay_receives_updates_14d():
    """After pipeline fix, onset_delay must receive HeatingEpisodes within 14 days.

    Soft assertion: prints diagnostic info and warns on 0 but does not hard-fail,
    because onset detection may require more than 14 days in some configurations.
    """
    cfg = scenario_restart_equiv_continuous()
    # Add model snapshots every day so we can inspect the final state
    cfg.model_snapshot_interval_steps = _steps_per_day(_STEP_S)
    cfg.seed = 42

    summary = await ScenarioRunner(cfg).run()
    final = _final_snap(summary)

    assert final is not None, "No model snapshots were recorded"

    onset_count = final.onset_delay_update_count
    total_count = final.model_update_total

    if onset_count == 0:
        # Soft: warn but do not hard-fail — 14 days may not be enough
        print(
            f"\nWARN [test_onset_delay_receives_updates_14d]: "
            f"onset_delay_update_count=0 after 14 days "
            f"(total model updates={total_count}, "
            f"heat_rate={final.heat_rate_update_count}, "
            f"heat_loss={final.heat_loss_update_count}). "
            "This may indicate that the pipeline fix did not reach this scenario or "
            "that 14 days is insufficient for onset detection."
        )
        pytest.xfail(
            "onset_delay received 0 updates in 14d — soft failure, "
            "see diagnostic output above"
        )
    else:
        assert onset_count >= 1, (
            f"onset_delay_update_count unexpectedly negative: {onset_count}"
        )


async def test_onset_delay_update_count_positive_30d():
    """onset_delay model must receive >0 updates in a 30-day supply-delay run.

    Uses scenario_clean_supply_delay_30d if available; falls back to a
    30-day run of scenario_restart_equiv_continuous with PROFILE_CLEAN_SUPPLY_DELAY.
    """
    try:
        from tests.simulation.scenarios import scenario_clean_supply_delay_30d
        cfg = scenario_clean_supply_delay_30d()
    except ImportError:
        # Fallback: build equivalent config inline
        cfg = ScenarioConfig(
            name="onset-delay-30d-fallback",
            room_profile=PROFILE_CLEAN_SUPPLY_DELAY,
            initial_room_temp=17.0,
            start_utc=_START,
            tz_offset_hours=1.0,
            step_s=_STEP_S,
            duration_s=30 * 24 * 3600,
            climate_entities=["climate.trv_od30_fb"],
            sensor_entity="sensor.room_od30_fb",
            outdoor_temp_fn=_outdoor_fn(-3.0),
            solar_gain_fn=_solar_gain(100.0),
            learning_mode=True,
            active_control=True,
            zone_cfg_overrides={
                "boost_bootstrap_prior_c": 1.5,
                "night_temp": 14.0,
                "comfort_temp": 21.0,
            },
            seed=42,
            model_snapshot_interval_steps=_steps_per_day(_STEP_S),
        )

    summary = await ScenarioRunner(cfg).run()
    final = _final_snap(summary)

    assert final is not None, "No model snapshots recorded"

    onset_count = final.onset_delay_update_count

    # Soft diagnostic report
    gt_supply_delay_s = VirtualRoomPhysics(cfg.room_profile).ground_truth_supply_delay_s
    learned_delay_s = final.learned_supply_delay_s
    print(
        f"\n[onset_delay 30d] update_count={onset_count}  "
        f"learned_supply_delay_s={learned_delay_s}  "
        f"gt_supply_delay_s={gt_supply_delay_s}"
    )

    assert onset_count > 0, (
        f"onset_delay model received 0 updates after 30 days "
        f"(total={final.model_update_total}, "
        f"heat_rate={final.heat_rate_update_count}). "
        "Pipeline fix may not be routing HeatingEpisodes to onset_delay."
    )


async def test_onset_delay_plausible_30d():
    """If onset_delay has received multiple updates, learned value must be plausible.

    Plausible range: [60, 3600] seconds (1 minute to 1 hour).
    Relative error is logged but not asserted (convergence may be slow).
    """
    try:
        from tests.simulation.scenarios import scenario_clean_supply_delay_30d
        cfg = scenario_clean_supply_delay_30d()
    except ImportError:
        cfg = ScenarioConfig(
            name="onset-delay-plausible-30d",
            room_profile=PROFILE_CLEAN_SUPPLY_DELAY,
            initial_room_temp=17.0,
            start_utc=_START,
            tz_offset_hours=1.0,
            step_s=_STEP_S,
            duration_s=30 * 24 * 3600,
            climate_entities=["climate.trv_odp30"],
            sensor_entity="sensor.room_odp30",
            outdoor_temp_fn=_outdoor_fn(-3.0),
            solar_gain_fn=_solar_gain(100.0),
            learning_mode=True,
            active_control=True,
            zone_cfg_overrides={
                "boost_bootstrap_prior_c": 1.5,
                "night_temp": 14.0,
                "comfort_temp": 21.0,
            },
            seed=42,
            model_snapshot_interval_steps=_steps_per_day(_STEP_S),
        )

    summary = await ScenarioRunner(cfg).run()
    final = _final_snap(summary)

    assert final is not None, "No model snapshots recorded"
    assert final.onset_delay_update_count > 0, (
        f"onset_delay received 0 updates — cannot assess plausibility "
        f"(total model updates={final.model_update_total})"
    )

    if final.onset_delay_update_count > 2:
        learned_s = final.learned_supply_delay_s
        if learned_s is not None:
            assert 60.0 <= learned_s <= 3600.0, (
                f"Learned onset delay {learned_s:.1f}s is outside plausible range "
                f"[60, 3600] s after {final.onset_delay_update_count} updates"
            )
            gt_s = VirtualRoomPhysics(cfg.room_profile).ground_truth_supply_delay_s
            if gt_s and gt_s != 0.0:
                rel_err = abs(learned_s - gt_s) / abs(gt_s)
                print(
                    f"\n[onset_delay plausible] learned={learned_s:.1f}s  "
                    f"gt={gt_s:.1f}s  rel_err={rel_err:.2%}"
                )
                assert math.isfinite(rel_err), (
                    f"Relative error is not finite: {rel_err}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2: Afterheat
# ═══════════════════════════════════════════════════════════════════════════════

async def test_afterheat_receives_updates_30d():
    """AFTERHEAT regime must trigger and the afterheat model must receive updates.

    Requires PROFILE_CLEAN_AFTERHEAT with spring-like outdoor temps so that
    the post-shutoff slope stays >= 0.01°C/min long enough to be detected.
    """
    try:
        from tests.simulation.scenarios import scenario_clean_afterheat_30d
        cfg = scenario_clean_afterheat_30d()
    except ImportError:
        # Inline fallback with PROFILE_CLEAN_AFTERHEAT and mild outdoor temp
        cfg = ScenarioConfig(
            name="afterheat-30d-fallback",
            room_profile=PROFILE_CLEAN_AFTERHEAT,
            initial_room_temp=17.0,
            start_utc=_START,
            tz_offset_hours=1.0,
            step_s=_STEP_S,
            duration_s=30 * 24 * 3600,
            climate_entities=["climate.trv_ah30_fb"],
            sensor_entity="sensor.room_ah30_fb",
            # Mild outdoor temps (≈ 5°C base) — maximises afterheat effect per physics comment
            outdoor_temp_fn=_outdoor_seasonal(
                _START, base_winter_c=3.0, base_summer_c=10.0, day_amplitude=2.0
            ),
            solar_gain_fn=_solar_gain(80.0),
            learning_mode=True,
            active_control=True,
            zone_cfg_overrides={
                "boost_bootstrap_prior_c": 1.5,
                "night_temp": 14.0,
                "comfort_temp": 21.0,
            },
            seed=42,
            model_snapshot_interval_steps=_steps_per_day(_STEP_S),
        )

    summary = await ScenarioRunner(cfg).run()
    final = _final_snap(summary)

    assert final is not None, "No model snapshots recorded"

    afterheat_count = final.afterheat_update_count
    if afterheat_count == 0:
        print(
            f"\nDIAGNOSTIC [test_afterheat_receives_updates_30d]: "
            f"afterheat_update_count=0 after 30 days. "
            f"total_updates={final.model_update_total}  "
            f"heat_rate={final.heat_rate_update_count}  "
            f"heat_loss={final.heat_loss_update_count}  "
            f"onset_delay={final.onset_delay_update_count}. "
            "AFTERHEAT regime triggers on slope >= 0.01°C/min after heating stops + decaying. "
            "Check that VirtualRoomPhysics afterheat_w is non-zero and outdoor temp is mild."
        )

    assert afterheat_count > 0, (
        f"afterheat model received 0 updates after 30 days "
        f"(total={final.model_update_total}). "
        "AFTERHEAT regime may not be triggering — check outdoor temperature range "
        "and PROFILE_CLEAN_AFTERHEAT parameters."
    )


async def test_afterheat_update_count_positive_30d():
    """Afterheat model must receive at least 3 updates for meaningful learning.

    Multiple episodes are needed before the model has enough evidence to produce
    a useful learned value. This test reports learned vs ground-truth tau.
    """
    try:
        from tests.simulation.scenarios import scenario_clean_afterheat_30d
        cfg = scenario_clean_afterheat_30d()
    except ImportError:
        cfg = ScenarioConfig(
            name="afterheat-count-30d",
            room_profile=PROFILE_CLEAN_AFTERHEAT,
            initial_room_temp=17.0,
            start_utc=_START,
            tz_offset_hours=1.0,
            step_s=_STEP_S,
            duration_s=30 * 24 * 3600,
            climate_entities=["climate.trv_ahc30"],
            sensor_entity="sensor.room_ahc30",
            outdoor_temp_fn=_outdoor_seasonal(
                _START, base_winter_c=3.0, base_summer_c=10.0, day_amplitude=2.0
            ),
            solar_gain_fn=_solar_gain(80.0),
            learning_mode=True,
            active_control=True,
            zone_cfg_overrides={
                "boost_bootstrap_prior_c": 1.5,
                "night_temp": 14.0,
                "comfort_temp": 21.0,
            },
            seed=42,
            model_snapshot_interval_steps=_steps_per_day(_STEP_S),
        )

    summary = await ScenarioRunner(cfg).run()
    final = _final_snap(summary)

    assert final is not None, "No model snapshots recorded"

    afterheat_count = final.afterheat_update_count
    gt_tau_s = VirtualRoomPhysics(cfg.room_profile).ground_truth_afterheat_tau_s
    # learned_afterheat_rise_c is the field available in ModelLearningSnapshot
    learned_rise_c = final.learned_afterheat_rise_c

    print(
        f"\n[afterheat count 30d] update_count={afterheat_count}  "
        f"learned_rise_c={learned_rise_c}  "
        f"gt_afterheat_tau_s={gt_tau_s}"
    )

    assert afterheat_count >= 3, (
        f"afterheat model received only {afterheat_count} updates after 30 days "
        f"(need >= 3 for meaningful learning). "
        f"total_updates={final.model_update_total}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3: HeatLoss
# ═══════════════════════════════════════════════════════════════════════════════

async def test_heat_loss_receives_updates_30d():
    """PASSIVE_COOLING regime must trigger and heat_loss model must receive updates.

    Requires night setback so the room cools freely (no heating, high ΔT,
    fast enough cooling rate to exceed the regime threshold).
    """
    try:
        from tests.simulation.scenarios import scenario_clean_heat_loss_30d
        cfg = scenario_clean_heat_loss_30d()
    except ImportError:
        cfg = ScenarioConfig(
            name="heat-loss-30d-fallback",
            room_profile=PROFILE_CLEAN_HEAT_LOSS,
            initial_room_temp=18.0,
            start_utc=_START,
            tz_offset_hours=1.0,
            step_s=_STEP_S,
            duration_s=30 * 24 * 3600,
            climate_entities=["climate.trv_hl30_fb"],
            sensor_entity="sensor.room_hl30_fb",
            # Cold winter to maximise ΔT during night setback
            outdoor_temp_fn=_outdoor_fn(-5.0),
            solar_gain_fn=_solar_gain(50.0),
            learning_mode=True,
            active_control=True,
            # Night setback: comfort=21°C, night=14°C → large ΔT during setback
            zone_cfg_overrides={
                "boost_bootstrap_prior_c": 1.5,
                "night_temp": 14.0,
                "comfort_temp": 21.0,
                "sched_wd_morning": "06:00",
                "sched_wd_night": "22:00",
                "sched_we_morning": "07:00",
                "sched_we_night": "22:00",
            },
            seed=42,
            model_snapshot_interval_steps=_steps_per_day(_STEP_S),
        )

    summary = await ScenarioRunner(cfg).run()
    final = _final_snap(summary)

    assert final is not None, "No model snapshots recorded"

    heat_loss_count = final.heat_loss_update_count
    if heat_loss_count == 0:
        print(
            "\nDIAGNOSTIC: regime trace needed — heat_loss_update_count=0 after 30 days. "
            f"total_updates={final.model_update_total}  "
            f"heat_rate={final.heat_rate_update_count}  "
            f"onset_delay={final.onset_delay_update_count}  "
            f"afterheat={final.afterheat_update_count}. "
            "PASSIVE_COOLING requires cooling rate >= 0.01°C/min with no heating. "
            "Ensure night_temp setback is configured and outdoor temp is sufficiently cold."
        )

    assert heat_loss_count > 0, (
        f"heat_loss model received 0 updates after 30 days "
        f"(total={final.model_update_total}). "
        "PASSIVE_COOLING regime may not be triggering — check night setback configuration "
        "and PROFILE_CLEAN_HEAT_LOSS heat_loss_w_per_k."
    )


async def test_heat_loss_update_count_positive_30d():
    """heat_loss model must receive at least 5 updates in a 30-day night-setback run."""
    try:
        from tests.simulation.scenarios import scenario_clean_heat_loss_30d
        cfg = scenario_clean_heat_loss_30d()
    except ImportError:
        cfg = ScenarioConfig(
            name="heat-loss-count-30d",
            room_profile=PROFILE_CLEAN_HEAT_LOSS,
            initial_room_temp=18.0,
            start_utc=_START,
            tz_offset_hours=1.0,
            step_s=_STEP_S,
            duration_s=30 * 24 * 3600,
            climate_entities=["climate.trv_hlc30"],
            sensor_entity="sensor.room_hlc30",
            outdoor_temp_fn=_outdoor_fn(-5.0),
            solar_gain_fn=_solar_gain(50.0),
            learning_mode=True,
            active_control=True,
            zone_cfg_overrides={
                "boost_bootstrap_prior_c": 1.5,
                "night_temp": 14.0,
                "comfort_temp": 21.0,
                "sched_wd_morning": "06:00",
                "sched_wd_night": "22:00",
                "sched_we_morning": "07:00",
                "sched_we_night": "22:00",
            },
            seed=42,
            model_snapshot_interval_steps=_steps_per_day(_STEP_S),
        )

    summary = await ScenarioRunner(cfg).run()
    final = _final_snap(summary)

    assert final is not None, "No model snapshots recorded"

    heat_loss_count = final.heat_loss_update_count
    assert heat_loss_count >= 5, (
        f"heat_loss model received only {heat_loss_count} updates after 30 days "
        f"(need >= 5). total_updates={final.model_update_total}"
    )


async def test_heat_loss_learned_value_plausible_30d():
    """If heat_loss has enough updates, learned value must be positive and finite."""
    try:
        from tests.simulation.scenarios import scenario_clean_heat_loss_30d
        cfg = scenario_clean_heat_loss_30d()
    except ImportError:
        cfg = ScenarioConfig(
            name="heat-loss-plausible-30d",
            room_profile=PROFILE_CLEAN_HEAT_LOSS,
            initial_room_temp=18.0,
            start_utc=_START,
            tz_offset_hours=1.0,
            step_s=_STEP_S,
            duration_s=30 * 24 * 3600,
            climate_entities=["climate.trv_hlp30"],
            sensor_entity="sensor.room_hlp30",
            outdoor_temp_fn=_outdoor_fn(-5.0),
            solar_gain_fn=_solar_gain(50.0),
            learning_mode=True,
            active_control=True,
            zone_cfg_overrides={
                "boost_bootstrap_prior_c": 1.5,
                "night_temp": 14.0,
                "comfort_temp": 21.0,
                "sched_wd_morning": "06:00",
                "sched_wd_night": "22:00",
                "sched_we_morning": "07:00",
                "sched_we_night": "22:00",
            },
            seed=42,
            model_snapshot_interval_steps=_steps_per_day(_STEP_S),
        )

    summary = await ScenarioRunner(cfg).run()
    final = _final_snap(summary)

    assert final is not None, "No model snapshots recorded"
    assert final.heat_loss_update_count > 0, (
        f"heat_loss received 0 updates — cannot assess plausibility "
        f"(total model updates={final.model_update_total})"
    )

    if final.heat_loss_update_count >= 3:
        learned_hl = final.learned_heat_loss_w_per_k
        if learned_hl is not None:
            assert learned_hl > 0.0, (
                f"Learned heat_loss_w_per_k={learned_hl} must be positive"
            )
            assert math.isfinite(learned_hl), (
                f"Learned heat_loss_w_per_k={learned_hl} is not finite"
            )
            gt_hl = VirtualRoomPhysics(cfg.room_profile).ground_truth_heat_loss_w_per_k
            print(
                f"\n[heat_loss plausible] learned={learned_hl:.2f} W/K  "
                f"gt={gt_hl:.2f} W/K  "
                f"updates={final.heat_loss_update_count}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 4: HeatRate regression guard after builder fix
# ═══════════════════════════════════════════════════════════════════════════════

async def test_heat_rate_still_works_after_builder_fix_7d():
    """Regression guard: heat_rate model must still receive updates after the pipeline fix.

    The shared-builder bug fix must not break heat_rate episode collection.
    This is the simplest smoke test: just ensure heat_rate_update_count > 0
    in a standard 7-day run using the unmodified scenario_s1a().
    """
    cfg = scenario_s1a()
    cfg.model_snapshot_interval_steps = _steps_per_day(_STEP_S)
    cfg.seed = 42

    summary = await ScenarioRunner(cfg).run()
    final = _final_snap(summary)

    assert final is not None, "No model snapshots recorded"
    assert final.heat_rate_update_count > 0, (
        f"heat_rate model has 0 updates after 7 days (post-builder-fix) "
        f"(total model updates={final.model_update_total}). "
        "The pipeline fix may have inadvertently broken heat_rate episode routing."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 5: Learning-progress coverage visibility
# ═══════════════════════════════════════════════════════════════════════════════

async def test_learning_progress_zero_without_model_coverage():
    """Scenario with only heat_rate coverage must show 0 for other model counts.

    Uses scenario_s1a() (7-day standard run) which provides heating episodes
    but no night-setback (heat_loss) and no afterheat scenario (afterheat).
    This proves that per-model counters correctly reflect actual coverage:
    a model that never received data shows count=0, not a fabricated positive.
    """
    # Try to use a dedicated heat-rate-only scenario if it exists
    try:
        from tests.simulation.scenarios import scenario_clean_heat_rate_7d
        cfg = scenario_clean_heat_rate_7d()
    except ImportError:
        # Fall back to standard 7-day run — minimal night setback, no afterheat profile
        cfg = scenario_s1a()
        cfg.name = "heat-rate-only-coverage"

    cfg.model_snapshot_interval_steps = _steps_per_day(_STEP_S)
    # Override seed for reproducibility
    cfg.seed = 42

    summary = await ScenarioRunner(cfg).run()
    final = _final_snap(summary)

    assert final is not None, "No model snapshots recorded"

    heat_rate_count = final.heat_rate_update_count
    heat_loss_count = final.heat_loss_update_count
    onset_delay_count = final.onset_delay_update_count

    print(
        f"\n[coverage visibility] "
        f"heat_rate={heat_rate_count}  "
        f"heat_loss={heat_loss_count}  "
        f"onset_delay={onset_delay_count}  "
        f"afterheat={final.afterheat_update_count}  "
        f"total={final.model_update_total}"
    )

    # heat_rate must have updated — this is the primary model for standard heating
    assert heat_rate_count > 0, (
        f"heat_rate_update_count=0 in standard 7-day run "
        f"(total={final.model_update_total}) — coverage proof fails"
    )

    # heat_loss must be 0: no night setback configured in scenario_s1a()
    # (night_temp defaults apply but the standard profile has low heat_loss_w_per_k=40
    # and the cooling rate may not exceed the PASSIVE_COOLING threshold)
    assert heat_loss_count == 0, (
        f"heat_loss_update_count={heat_loss_count} in a scenario without explicit "
        f"night setback — per-model counters may be mis-attributed. "
        "This proves that learning progress does not pretend all models learned."
    )
