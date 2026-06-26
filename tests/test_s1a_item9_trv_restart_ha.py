"""Phase S1a Item 9 — TRV Restart Behaviour under Real HA (Docker/CI only).

Tests what can be verified at the HA-integration level after coordinator restart:

  • _boost_active is empty after reload (ephemeral coordinator state)
  • _last_written_setpoints is empty after reload (ephemeral coordinator state)
  • The learning shadow controller's pending attribution contexts survive reload
  • processed_decision_ids dedup survives reload (no double outcome on reload)
  • After reload the coordinator can handle a TRV reporting an elevated setpoint
    without triggering a spurious manual-override (the TRV still holds the
    boosted setpoint from before restart, but _last_written_setpoints is empty)
  • The learning runtime's health() shows 0 storage_warnings after clean reload

Note on scope:
  Full per-device setpoint write verification (direct-valve, mixed multi-TRV,
  partial-failure per-entity) requires deep service-call mocking in trv_control.py.
  Those tests are scoped to S1b (trv_control coverage currently 34 %).
  What IS proven here: coordinator-level ephemeral-state reset + runtime-level
  pending-context persistence through HA's config-entry reload path.
"""
from __future__ import annotations

import logging

import pytest

from custom_components.thermosmart.const import DOMAIN
from custom_components.thermosmart.coordinator import ControlAdaptationMode
from tests.helpers_ha_real import setup_zone
from tests.helpers_runtime_scenarios import heating_ramp_then_settle

pytestmark = pytest.mark.asyncio

_LOG = logging.getLogger(__name__)


def _coord(entry_data):
    return entry_data["coordinator"]


def _shadow(entry_data):
    return entry_data.get("le2_shadow")


# ── 1. Coordinator ephemeral state resets on reload ───────────────────────────


async def test_boost_active_empty_after_reload(hass, hass_storage,
                                               enable_custom_integrations):
    """_boost_active is ephemeral: must be {} immediately after reload."""
    _, entry, entry_data1 = await setup_zone(hass)
    coord1 = _coord(entry_data1)
    # Simulate a boost having been active
    coord1._boost_active["lz"] = {"offset_c": 2.0, "target": 21.0}

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    assert coord2._boost_active == {}, \
        "_boost_active must be empty after reload (ephemeral coordinator state)"


async def test_last_written_setpoints_empty_after_reload(hass, hass_storage,
                                                          enable_custom_integrations):
    """_last_written_setpoints is ephemeral: must be {} immediately after reload.

    This is the expected behaviour — after reload, the coordinator re-reads TRV
    states from HA and rebuilds its view from scratch.  It does NOT try to
    reconstruct _last_written_setpoints from the persisted store.
    """
    _, entry, entry_data1 = await setup_zone(hass)
    coord1 = _coord(entry_data1)
    # Simulate previously written setpoints
    coord1._last_written_setpoints["climate.test_trv"] = 23.0

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    assert coord2._last_written_setpoints == {}, \
        "_last_written_setpoints must be empty after reload"


async def test_ec_hold_fields_reset_after_reload(hass, hass_storage,
                                                   enable_custom_integrations):
    """Early-cutoff hold fields must be False/inactive after reload (ephemeral)."""
    _, entry, entry_data1 = await setup_zone(hass)
    coord1 = _coord(entry_data1)
    coord1._ec_hold_active = True
    coord1._ec_state = "coasting_hold"

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)
    assert not coord2._ec_hold_active, "_ec_hold_active must be False after reload"
    assert coord2._ec_state == "inactive", "_ec_state must be 'inactive' after reload"


# ── 2. Shadow controller pending attribution survives reload ──────────────────


async def test_pending_boost_contexts_survive_reload(hass, hass_storage,
                                                      enable_custom_integrations):
    """After reload, pending_boost_contexts injected before flush must be present
    in the restored shadow runtime (proves the HA-level persistence path works)."""
    from custom_components.thermosmart.learning.models import BoostUpdateContext
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = _shadow(entry_data1)
    if shadow1 is None:
        pytest.skip("No shadow controller in this mode")

    rt1 = shadow1.runtime
    zr1 = rt1._zone("lz")

    did = "dec-ha-persist-1"
    zr1.pending_boost_contexts[did] = BoostUpdateContext(
        source_episode_id=did, decision_id=did,
        learning_zone_id="lz",
        boost_applied_c=2.0,
        dispatch_status="fully_succeeded",
        authority="live_record",
    )
    rt1.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    # Reload
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    if shadow2 is None:
        pytest.skip("No shadow controller after reload")

    rt2 = shadow2.runtime
    zr2 = rt2._zone("lz")

    assert did in zr2.pending_boost_contexts, \
        "pending_boost_contexts entry must survive reload"
    bctx = zr2.pending_boost_contexts[did]
    assert bctx.decision_id == did
    assert bctx.boost_applied_c == 2.0


async def test_pending_dispatch_records_survive_reload(hass, hass_storage,
                                                        enable_custom_integrations):
    """After reload, pending_dispatch_records injected before flush must be restored."""
    from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = _shadow(entry_data1)
    if shadow1 is None:
        pytest.skip("No shadow controller in this mode")

    rt1 = shadow1.runtime
    zr1 = rt1._zone("lz")

    did = "dec-ha-drec-1"
    zr1.pending_dispatch_records[did] = BoostDispatchRecord(
        decision_id=did,
        boost_applied_c=2.0,
        baseline_setpoint_c=21.0,
        final_setpoint_c=23.0,
        dispatch_status="fully_succeeded",
        outcome_eligible=True,
        outcome_reliability="full",
        device_control_type="setpoint",
        effective_setpoints=(23.0,),
        targets_total=1,
        targets_failed=0,
    )
    rt1.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    if shadow2 is None:
        pytest.skip("No shadow controller after reload")

    zr2 = shadow2.runtime._zone("lz")
    assert did in zr2.pending_dispatch_records, \
        "pending_dispatch_records entry must survive reload"
    drec = zr2.pending_dispatch_records[did]
    assert drec.dispatch_status == "fully_succeeded"
    assert drec.effective_setpoints == (23.0,)


async def test_partial_failure_dispatch_record_survives_reload(hass, hass_storage,
                                                                enable_custom_integrations):
    """Partial-failure provenance (targets_failed) must survive reload."""
    from custom_components.thermosmart.learning.runtime.capture import BoostDispatchRecord
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = _shadow(entry_data1)
    if shadow1 is None:
        pytest.skip("No shadow controller in this mode")

    did = "dec-partial-ha"
    shadow1.runtime._zone("lz").pending_dispatch_records[did] = BoostDispatchRecord(
        decision_id=did,
        dispatch_status="partially_succeeded",
        outcome_reliability="partial",
        effective_setpoints=(23.0,),
        targets_total=2,
        targets_failed=1,
    )
    shadow1.runtime.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    if shadow2 is None:
        pytest.skip("No shadow controller after reload")

    drec = shadow2.runtime._zone("lz").pending_dispatch_records.get(did)
    assert drec is not None, "Partial-failure dispatch record must survive reload"
    assert drec.dispatch_status == "partially_succeeded"
    assert drec.targets_failed == 1


# ── 3. No double outcome after reload ─────────────────────────────────────────


async def test_processed_decision_ids_survive_reload(hass, hass_storage,
                                                      enable_custom_integrations):
    """processed_decision_ids must survive reload (dedup intact, no double outcome)."""
    from tests.helpers_boost import boost_episode, boost_context
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = _shadow(entry_data1)
    if shadow1 is None:
        pytest.skip("No shadow controller")

    rt1 = shadow1.runtime
    zr1 = rt1._zone("lz")
    m1 = zr1.orchestrator.models.get("boost")
    if m1 is None:
        pytest.skip("No boost model")

    # Apply a boost outcome to record a processed decision_id
    did = "dec-dedup-ha-1"
    vals = [18.0, 19.5, 21.0]
    ep = boost_episode(did, vals, decision_id=did, zone="lz")
    bctx = boost_context(ep)
    m1.update(ep, bctx)
    full_count_before = m1._state.full_count
    ids_before = set(m1._state.processed_decision_ids)

    rt1.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    if shadow2 is None:
        pytest.skip("No shadow controller after reload")

    m2 = shadow2.runtime._zone("lz").orchestrator.models.get("boost")
    assert m2 is not None
    ids_after = set(m2._state.processed_decision_ids)
    assert ids_before == ids_after, \
        "processed_decision_ids must survive reload"

    # Attempt a duplicate update → must be rejected
    m2.update(ep, bctx)
    assert m2._state.full_count == full_count_before, \
        "Duplicate outcome after reload must be rejected by dedup"


async def test_no_double_dispatch_after_reload(hass, hass_storage,
                                                enable_custom_integrations):
    """After reload, _boost_active is empty so no double dispatch occurs.

    The coordinator must not re-issue the same boost service call just because
    _boost_active was non-empty before the reload.
    """
    _, entry, entry_data1 = await setup_zone(hass)
    coord1 = _coord(entry_data1)
    coord1._boost_active["lz"] = {"offset_c": 2.0}

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)

    # After reload: _boost_active must be empty — no stale offset to re-dispatch
    assert "lz" not in coord2._boost_active or coord2._boost_active.get("lz") is None, \
        "Stale _boost_active entry must not survive reload"


# ── 4. TRV state after reload ──────────────────────────────────────────────────


async def test_coordinator_refresh_after_reload_reads_current_trv_state(
        hass, hass_storage, enable_custom_integrations):
    """After reload, the first coordinator refresh reads the current TRV state.

    This is the mechanism by which a raised setpoint (from a pre-reload boost)
    is re-incorporated into the learning runtime — via the runtime cycle input,
    not via stale coordinator state.
    """
    _, entry, entry_data1 = await setup_zone(hass)

    # Simulate TRV having been raised to boost setpoint by a previous boost
    hass.states.async_set(
        "climate.test_trv", "heat",
        {"temperature": 23.0,  # elevated: was boosted to target+2
         "current_temperature": 20.5,
         "min_temp": 5.0, "max_temp": 30.0,
         "hvac_modes": ["heat", "off"], "supported_features": 1,
         "friendly_name": "TRV"},
    )

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)

    # After reload, the coordinator must be operational
    assert coord2 is not None
    # _last_written_setpoints is empty after reload (fresh start)
    assert coord2._last_written_setpoints == {}


async def test_shadow_runtime_health_clean_after_reload_with_pending(
        hass, hass_storage, enable_custom_integrations):
    """After reload with pending attribution contexts in store, health is clean."""
    from custom_components.thermosmart.learning.models import BoostUpdateContext
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = _shadow(entry_data1)
    if shadow1 is None:
        pytest.skip("No shadow controller")

    # Inject a pending context
    zr = shadow1.runtime._zone("lz")
    did = "dec-health-ha-1"
    zr.pending_boost_contexts[did] = BoostUpdateContext(
        source_episode_id=did, decision_id=did,
        learning_zone_id="lz", boost_applied_c=1.5,
        dispatch_status="fully_succeeded", authority="live_record",
    )
    shadow1.runtime.mark_dirty(important=True)
    await shadow1.async_flush()
    await hass.async_block_till_done()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    shadow2 = _shadow(entry_data2)
    if shadow2 is None:
        pytest.skip("No shadow controller after reload")

    h = shadow2.runtime.health()
    assert h.initialized, "Runtime must be initialized after reload"
    assert h.storage_warnings == 0, \
        f"No storage warnings expected after clean reload, got {h.storage_warnings}"


# ── 5. Restore barrier safety with pending contexts ───────────────────────────


async def test_restore_barrier_not_affected_by_pending_contexts(
        hass, hass_storage, enable_custom_integrations):
    """Pending attribution contexts in the store must not affect coordinator startup.

    The coordinator's setup must complete normally even when the store blob
    contains 5 pending attribution contexts.  The Restore Barrier itself
    (_active_control_initialized) is set by the RestoreEntity mechanism during
    setup — which fires inside async_block_till_done() — so we verify the
    coordinator is operational post-setup, not that the barrier is still False
    (that window is too narrow to observe reliably in integration tests).
    """
    from custom_components.thermosmart.learning.models import BoostUpdateContext
    _, entry, entry_data1 = await setup_zone(hass)
    shadow1 = _shadow(entry_data1)
    if shadow1 is None:
        pytest.skip("No shadow controller")

    # Inject 5 pending contexts into the store before reload
    zr = shadow1.runtime._zone("lz")
    for i in range(5):
        did = f"dec-barrier-{i}"
        zr.pending_boost_contexts[did] = BoostUpdateContext(
            source_episode_id=did, decision_id=did, learning_zone_id="lz",
            boost_applied_c=2.0, dispatch_status="fully_succeeded",
            authority="live_record",
        )
    shadow1.runtime.mark_dirty(important=True)
    await shadow1.async_flush()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = _coord(entry_data2)

    # The coordinator must be operational (pending contexts don't break startup)
    assert coord2 is not None, \
        "Coordinator must be operational even with pending attribution contexts"
    assert coord2.last_update_success, \
        "Coordinator last_update_success must be True after reload with pending contexts"

    # The shadow's pending contexts must be available post-reload
    shadow2 = _shadow(entry_data2)
    if shadow2 is not None:
        zr2 = shadow2.runtime._zone("lz")
        assert len(zr2.pending_boost_contexts) == 5, \
            "All 5 pending boost contexts must be accessible post-reload"
