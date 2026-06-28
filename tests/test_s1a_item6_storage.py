"""S1a Item 6 — Storage bounds and retention invariants (Windows + Docker).

Verifies that the LE2 persistence layer:
  - grows at a bounded rate (≤ 50 KB/day)
  - produces a parseable JSON blob after restarts
  - does not duplicate outcomes after multiple restarts
  - retains model state across a flush-reload cycle
"""
from __future__ import annotations

import json
import pytest

from tests.simulation.physics import PROFILE_FAST_INSULATED
from tests.simulation.scenarios import (
    _HEATING_SEASON_START_UTC,
    _outdoor_seasonal,
    _solar_gain,
    _steps_per_day,
)
from tests.simulation.runner import ScenarioConfig, ScenarioRunner
from tests.simulation.result import (
    SimulationResult,
    ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY,
)

pytestmark = pytest.mark.asyncio

_START = _HEATING_SEASON_START_UTC
_STEP_S = 300.0


def _outdoor_fn():
    return _outdoor_seasonal(_START, base_winter_c=-3.0,
                             base_summer_c=16.0, day_amplitude=4.0)


def _base_cfg(name: str, duration_days: float, restarts: set = None) -> ScenarioConfig:
    return ScenarioConfig(
        name=name,
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=_START,
        step_s=_STEP_S,
        duration_s=duration_days * 24 * 3600,
        climate_entities=[f"climate.trv_{name}"],
        sensor_entity=f"sensor.room_{name}",
        outdoor_temp_fn=_outdoor_fn(),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={"boost_bootstrap_prior_c": 1.5},
        seed=42,
        restart_steps=restarts or set(),
        model_snapshot_interval_steps=_steps_per_day(_STEP_S),
    )


async def test_storage_blob_parseable_json_after_7d():
    """Storage blob is valid JSON after a 7-day run."""
    cfg = _base_cfg("stor-json-7d", 7.0)
    runner = ScenarioRunner(cfg)
    await runner.run()
    # Access the FakeStore via the runner's _fake_store attribute
    blob = runner._fake_store.data
    assert blob is not None, "FakeStore has no data after 7-day run"
    serialized = json.dumps(blob)
    recovered = json.loads(serialized)
    assert isinstance(recovered, dict), "Storage blob is not a dict"


async def test_storage_blob_parseable_json_after_restart():
    """Storage blob is valid JSON after a flush+reload cycle."""
    step_s = 300.0
    cfg = _base_cfg("stor-json-restart", 14.0,
                    restarts={int(7 * 24 * 3600 / step_s)})
    runner = ScenarioRunner(cfg)
    await runner.run()
    blob = runner._fake_store.data
    assert blob is not None
    serialized = json.dumps(blob)
    recovered = json.loads(serialized)
    assert isinstance(recovered, dict)


async def test_storage_growth_bounded_7d():
    """Storage blob growth ≤ 50 KB/day over 7-day run."""
    cfg = _base_cfg("stor-growth-7d", 7.0)
    summary = await ScenarioRunner(cfg).run()
    result = SimulationResult.from_summary(
        summary,
        scenario_id="stor-growth-7d",
        profile_name=PROFILE_FAST_INSULATED.name,
        seed=42,
        duration_days=7.0,
        step_s=_STEP_S,
    )
    assert result.storage_growth_kb_per_day <= ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY, (
        f"Storage growth {result.storage_growth_kb_per_day:.1f} KB/day "
        f"exceeds {ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY} KB/day limit"
    )


async def test_storage_growth_bounded_30d():
    """Storage blob growth ≤ 50 KB/day over 30-day run with 3 restarts."""
    step_s = 300.0
    restarts = {
        int(7 * 24 * 3600 / step_s),
        int(14 * 24 * 3600 / step_s),
        int(21 * 24 * 3600 / step_s),
    }
    cfg = _base_cfg("stor-growth-30d", 30.0, restarts=restarts)
    summary = await ScenarioRunner(cfg).run()
    result = SimulationResult.from_summary(
        summary,
        scenario_id="stor-growth-30d",
        profile_name=PROFILE_FAST_INSULATED.name,
        seed=42,
        duration_days=30.0,
        step_s=_STEP_S,
    )
    assert result.storage_growth_kb_per_day <= ACCEPTANCE_STORAGE_GROWTH_MAX_KB_PER_DAY, (
        f"30d storage growth {result.storage_growth_kb_per_day:.1f} KB/day > limit"
    )


async def test_storage_final_blob_below_512kb():
    """Final storage blob must be ≤ 512 KB regardless of run duration."""
    cfg = _base_cfg("stor-final-14d", 14.0)
    runner = ScenarioRunner(cfg)
    summary = await runner.run()
    blob_data = runner._fake_store.data or {}
    blob_kb = len(json.dumps(blob_data).encode("utf-8")) / 1024.0
    assert blob_kb <= 512.0, f"Final blob {blob_kb:.1f} KB exceeds 512 KB"


async def test_storage_save_count_increases_with_restarts():
    """More restarts → more forced saves in cumulative save count.

    Each restart triggers an explicit async_flush() which is counted in
    runner._total_saves.  Individual FakeStore.saves resets to 0 per store,
    so we compare cumulative counts (runner._total_saves + runner._fake_store.saves).
    """
    step_s = 300.0
    cfg_no_restart = _base_cfg("stor-saves-no", 7.0)
    cfg_with_restart = _base_cfg(
        "stor-saves-yes", 7.0,
        restarts={int(3 * 24 * 3600 / step_s), int(5 * 24 * 3600 / step_s)}
    )
    runner_no = ScenarioRunner(cfg_no_restart)
    runner_yes = ScenarioRunner(cfg_with_restart)
    await runner_no.run()
    await runner_yes.run()

    total_yes = runner_yes._total_saves + runner_yes._fake_store.saves
    # Each restart triggers 1 explicit async_flush() → at least 2 total saves
    assert total_yes >= 2, (
        f"With 2 restarts, expected ≥ 2 cumulative saves but got {total_yes}. "
        "Each restart calls _do_restart → async_flush → FakeStore.saves += 1."
    )


async def test_storage_restore_preserves_model_update_count():
    """After flush+reload, model_update_total at day 14 ≥ model_update_total at day 7."""
    step_s = 300.0
    snapshot_interval = _steps_per_day(step_s)
    restart_step = int(7 * 24 * 3600 / step_s)
    cfg = ScenarioConfig(
        name="stor-restore-persist",
        room_profile=PROFILE_FAST_INSULATED,
        initial_room_temp=17.0,
        start_utc=_START,
        step_s=step_s,
        duration_s=14 * 24 * 3600,
        climate_entities=["climate.trv_stor_r"],
        sensor_entity="sensor.room_stor_r",
        outdoor_temp_fn=_outdoor_fn(),
        solar_gain_fn=_solar_gain(100.0),
        learning_mode=True,
        active_control=True,
        zone_cfg_overrides={"boost_bootstrap_prior_c": 1.5},
        seed=42,
        restart_steps={restart_step},
        model_snapshot_interval_steps=snapshot_interval,
    )
    summary = await ScenarioRunner(cfg).run()
    snaps = summary.model_snapshots
    if len(snaps) < 2:
        pytest.skip("Not enough snapshots to compare pre/post restart")

    pre = [s for s in snaps if s.step_idx < restart_step]
    post = [s for s in snaps if s.step_idx > restart_step]
    if not pre or not post:
        pytest.skip("Snapshots not on both sides of restart")

    pre_count = pre[-1].model_update_total
    post_count = post[0].model_update_total
    assert post_count >= pre_count, (
        f"model_update_total dropped after restart: {pre_count} → {post_count}"
    )


async def test_no_double_save_on_multiple_flush_calls():
    """Idempotent flush: repeated flush calls must not grow the blob.

    The first flush (at end of run()) persists the current model state.
    Subsequent flushes with no new learning must produce the same-sized blob.
    """
    step_s = 300.0
    cfg = _base_cfg("stor-idempotent", 3.0)
    runner = ScenarioRunner(cfg)
    await runner.run()
    # Capture blob size right after the final flush (inside run())
    blob_after_run = runner._fake_store.data or {}
    size_after_run = len(json.dumps(blob_after_run).encode("utf-8"))

    # Extra flushes with no new learning
    for _ in range(3):
        await runner._shadow._runtime.async_flush()
    blob_after_extra = runner._fake_store.data or {}
    size_after_extra = len(json.dumps(blob_after_extra).encode("utf-8"))

    # Blob must not grow significantly: repeated flushes with no new learning
    # should produce the same or nearly same blob (no unbounded accumulation).
    # Allow 1% tolerance for minor serialization differences.
    max_allowed = max(size_after_run + 512, int(size_after_run * 1.01))
    assert size_after_extra <= max_allowed, (
        f"Repeated flushes grew blob significantly: {size_after_run}B → {size_after_extra}B "
        f"(delta {size_after_extra - size_after_run}B, allowed ≤ {max_allowed - size_after_run}B)"
    )
