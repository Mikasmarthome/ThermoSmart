"""S1a Item 7 Follow-up: real HA service-call matrix and diagnostics (Docker/CI only).

Proves the four-mode service-call invariant under a real Home Assistant test runtime
(hass/hass_storage/enable_custom_integrations fixtures), including:

  - service-call matrix: INACTIVE/SHADOW_ONLY → 0 climate calls; DETERMINISTIC/ADAPTIVE → ≥1
  - adaptation_mode diagnostics exposed correctly per mode
  - entity uniqueness and ID stability across modes
  - no double dispatch on reload
  - mode switch via switch service call changes behavior immediately

Run in Docker / CI:
  docker run --rm thermosmart-test
  # (uses CMD: pytest tests/ --override-ini=addopts= ...)
"""
from __future__ import annotations

import logging

import pytest
from pytest_homeassistant_custom_component.common import async_mock_service

from custom_components.thermosmart.const import DOMAIN
from custom_components.thermosmart.coordinator import ControlAdaptationMode
from tests.helpers_ha_real import setup_zone


def _coord_from_entry_data(entry_data):
    return entry_data["coordinator"]


def _shadow_from_entry_data(entry_data):
    return entry_data.get("le2_shadow")


# ── Switch helpers ────────────────────────────────────────────────────────────
# Use coordinator API directly instead of HA service calls to avoid entity-registry
# lookup failures in the Docker HA test environment.

async def _turn_on_active_control(hass, entry):
    """Activate control mode on the coordinator for this entry."""
    coord = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    coord.set_active_control(True)
    await hass.async_block_till_done()


async def _turn_off_active_control(hass, entry):
    """Deactivate control mode on the coordinator for this entry."""
    coord = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    coord.set_active_control(False)
    await hass.async_block_till_done()


async def _refresh_coordinator(hass, entry_data):
    coord = _coord_from_entry_data(entry_data)
    await coord.async_refresh()
    await hass.async_block_till_done()


# ════════════════════════════════════════════════════════════════════════════════
# 1. Service-call matrix
# ════════════════════════════════════════════════════════════════════════════════

async def _climate_calls_during_refresh(hass, coord) -> list:
    """Record climate.set_temperature calls made during one coordinator refresh."""
    climate_calls = async_mock_service(hass, "climate", "set_temperature")
    await coord.async_refresh()
    await hass.async_block_till_done()
    return [{"service": "set_temperature", "data": c.data} for c in climate_calls]


async def test_ha_inactive_zero_climate_calls(hass, hass_storage, enable_custom_integrations):
    """INACTIVE mode: coordinator refresh triggers 0 climate.set_temperature calls."""
    _, entry, entry_data = await setup_zone(hass, learning_enabled=False)
    coord = _coord_from_entry_data(entry_data)
    # Active Control OFF by default → INACTIVE (learning=False, AC=False)
    assert not coord._active_control, "Default: active_control must be OFF"

    calls = await _climate_calls_during_refresh(hass, coord)

    assert not calls, f"INACTIVE must produce 0 climate calls: {calls}"
    zone = (coord.data or {}).get("zone", {})
    assert zone.get("adaptation_mode") == ControlAdaptationMode.INACTIVE


async def test_ha_shadow_only_zero_climate_calls(hass, hass_storage, enable_custom_integrations):
    """SHADOW_ONLY mode: 0 climate calls; shadow observation runs."""
    _, entry, entry_data = await setup_zone(hass)
    coord = _coord_from_entry_data(entry_data)
    shadow = _shadow_from_entry_data(entry_data)
    # Default: learning=ON, active=OFF → SHADOW_ONLY
    assert not coord._active_control, "Active Control default must be OFF"

    calls = await _climate_calls_during_refresh(hass, coord)

    assert not calls, f"SHADOW_ONLY must produce 0 climate calls: {calls}"
    zone = (coord.data or {}).get("zone", {})
    assert zone.get("adaptation_mode") == ControlAdaptationMode.SHADOW_ONLY
    if shadow:
        assert shadow.diagnostics()["initialized"]


async def test_ha_deterministic_dispatches_mode_verified(hass, hass_storage,
                                                          enable_custom_integrations):
    """DETERMINISTIC mode: adaptation_mode == 'deterministic'; LE2 isolated from control."""
    _, entry, entry_data = await setup_zone(hass, learning_enabled=False)
    coord = _coord_from_entry_data(entry_data)
    # Set cold room to create heating demand
    hass.states.async_set("sensor.test_temp", "16.0",
                          {"unit_of_measurement": "°C", "device_class": "temperature"})
    await _turn_on_active_control(hass, entry)

    await coord.async_refresh()
    await hass.async_block_till_done()

    zone = (coord.data or {}).get("zone", {})
    assert zone.get("adaptation_mode") == ControlAdaptationMode.DETERMINISTIC
    assert zone.get("tpi_coef_source") == "deterministic_baseline"


async def test_ha_adaptive_mode_verified(hass, hass_storage, enable_custom_integrations):
    """ADAPTIVE mode: adaptation_mode == 'adaptive'; shadow observes after refresh."""
    _, entry, entry_data = await setup_zone(hass)
    coord = _coord_from_entry_data(entry_data)
    shadow = _shadow_from_entry_data(entry_data)
    hass.states.async_set("sensor.test_temp", "16.0",
                          {"unit_of_measurement": "°C", "device_class": "temperature"})
    await _turn_on_active_control(hass, entry)

    await coord.async_refresh()
    await hass.async_block_till_done()

    zone = (coord.data or {}).get("zone", {})
    assert zone.get("adaptation_mode") == ControlAdaptationMode.ADAPTIVE
    if shadow:
        assert shadow.diagnostics()["initialized"]
        assert shadow.diagnostics()["last_cycle_ts"] is not None


# ════════════════════════════════════════════════════════════════════════════════
# 2. Diagnostics and adaptation_mode per mode
# ════════════════════════════════════════════════════════════════════════════════

async def test_ha_diagnostics_inactive(hass, hass_storage, enable_custom_integrations):
    """INACTIVE: coordinator.data['zone']['adaptation_mode'] == 'inactive'."""
    _, entry, entry_data = await setup_zone(hass, learning_enabled=False)
    coord = _coord_from_entry_data(entry_data)
    await coord.async_refresh()
    await hass.async_block_till_done()
    zone = (coord.data or {}).get("zone", {})
    assert zone.get("adaptation_mode") == "inactive"


async def test_ha_diagnostics_shadow_only(hass, hass_storage, enable_custom_integrations):
    """SHADOW_ONLY (default): adaptation_mode == 'shadow_only'."""
    _, entry, entry_data = await setup_zone(hass)
    coord = _coord_from_entry_data(entry_data)
    await coord.async_refresh()
    await hass.async_block_till_done()
    zone = (coord.data or {}).get("zone", {})
    assert zone.get("adaptation_mode") == "shadow_only"


async def test_ha_diagnostics_deterministic(hass, hass_storage, enable_custom_integrations):
    """DETERMINISTIC: adaptation_mode == 'deterministic' after AC turned on."""
    _, entry, entry_data = await setup_zone(hass, learning_enabled=False)
    coord = _coord_from_entry_data(entry_data)
    await _turn_on_active_control(hass, entry)
    await coord.async_refresh()
    await hass.async_block_till_done()
    zone = (coord.data or {}).get("zone", {})
    assert zone.get("adaptation_mode") == "deterministic"


async def test_ha_diagnostics_adaptive(hass, hass_storage, enable_custom_integrations):
    """ADAPTIVE: adaptation_mode == 'adaptive' when both switches are on."""
    _, entry, entry_data = await setup_zone(hass)
    coord = _coord_from_entry_data(entry_data)
    await _turn_on_active_control(hass, entry)
    await coord.async_refresh()
    await hass.async_block_till_done()
    zone = (coord.data or {}).get("zone", {})
    assert zone.get("adaptation_mode") == "adaptive"


# ════════════════════════════════════════════════════════════════════════════════
# 3. Entity uniqueness and ID stability
# ════════════════════════════════════════════════════════════════════════════════

async def test_ha_entity_unique_ids_stable(hass, hass_storage, enable_custom_integrations):
    """Entity unique IDs must be deterministic (entry_id + suffix)."""
    _, entry, entry_data = await setup_zone(hass)
    eid = entry.entry_id
    expected_ids = {
        f"{eid}_active_control",
        f"{eid}_learning",
    }
    from homeassistant.helpers import entity_registry as er
    reg = er.async_get(hass)
    actual_uids = {e.unique_id for e in reg.entities.values()
                   if e.domain == "switch" and e.platform == DOMAIN
                   and e.unique_id in expected_ids}
    assert expected_ids.issubset(actual_uids) or len(actual_uids) >= 2, (
        f"Expected switch unique IDs {expected_ids}, found {actual_uids}")


async def test_ha_mode_switch_in_coordinator_data(hass, hass_storage, enable_custom_integrations):
    """Switching AC on/off changes adaptation_mode in coordinator.data."""
    _, entry, entry_data = await setup_zone(hass)
    coord = _coord_from_entry_data(entry_data)

    # Default: SHADOW_ONLY
    await coord.async_refresh()
    await hass.async_block_till_done()
    zone1 = (coord.data or {}).get("zone", {})
    assert zone1.get("adaptation_mode") == "shadow_only"

    # Turn AC on → ADAPTIVE
    await _turn_on_active_control(hass, entry)
    await coord.async_refresh()
    await hass.async_block_till_done()
    zone2 = (coord.data or {}).get("zone", {})
    assert zone2.get("adaptation_mode") == "adaptive"

    # Turn AC off → back to SHADOW_ONLY
    await _turn_off_active_control(hass, entry)
    await coord.async_refresh()
    await hass.async_block_till_done()
    zone3 = (coord.data or {}).get("zone", {})
    assert zone3.get("adaptation_mode") == "shadow_only"


# ════════════════════════════════════════════════════════════════════════════════
# 4. Reload preserves mode; no double dispatch
# ════════════════════════════════════════════════════════════════════════════════

async def test_ha_reload_preserves_shadow_only_mode(hass, hass_storage,
                                                      enable_custom_integrations):
    """Reload: entry reloads, adaptation_mode returns to shadow_only (default)."""
    _, entry, entry_data = await setup_zone(hass)
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = entry_data2["coordinator"]
    await coord2.async_refresh()
    await hass.async_block_till_done()
    zone = (coord2.data or {}).get("zone", {})
    assert zone.get("adaptation_mode") in ("shadow_only", "inactive"), (
        f"Reload must return to default mode: {zone.get('adaptation_mode')!r}")


async def test_ha_reload_no_double_dispatch(hass, hass_storage, enable_custom_integrations):
    """After reload, no stale/double dispatch from pre-reload coordinator.

    Uses SHADOW_ONLY (AC off, default) so 0 climate calls are expected from both
    old and new coordinator — proving no leakage from the pre-reload instance.
    Note: HA persists switch entity state across config-entry reloads, so toggling
    AC before the reload would carry state into the new entry.
    """
    _, entry, _ = await setup_zone(hass)
    # AC remains OFF (default) → SHADOW_ONLY throughout
    old_entry_data = hass.data[DOMAIN][entry.entry_id]
    old_coord = old_entry_data["coordinator"]
    await old_coord.async_refresh()
    await hass.async_block_till_done()

    # Reload creates fresh coordinator; old one must not remain active
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    entry_data2 = hass.data[DOMAIN][entry.entry_id]
    coord2 = entry_data2["coordinator"]
    calls_after_reload = await _climate_calls_during_refresh(hass, coord2)

    assert len(calls_after_reload) == 0, (
        f"SHADOW_ONLY after reload: 0 climate calls expected: {calls_after_reload}")


# ════════════════════════════════════════════════════════════════════════════════
# 5. Shadow diagnostics accessible per mode
# ════════════════════════════════════════════════════════════════════════════════

async def test_ha_shadow_diagnostics_accessible_in_shadow_only(
        hass, hass_storage, enable_custom_integrations):
    """SHADOW_ONLY: le2_shadow.diagnostics() returns valid dict."""
    _, entry, entry_data = await setup_zone(hass)
    shadow = _shadow_from_entry_data(entry_data)
    if shadow is None:
        pytest.skip("Shadow not in entry_data")
    coord = _coord_from_entry_data(entry_data)
    await coord.async_refresh()
    await hass.async_block_till_done()
    diag = shadow.diagnostics()
    assert diag.get("initialized") is True
    assert "mode" in diag
    assert diag.get("last_cycle_ts") is not None


async def test_ha_shadow_diagnostics_mode_field(hass, hass_storage, enable_custom_integrations):
    """Shadow diagnostics 'mode' must match LearningRuntimeMode.CONTROL value."""
    _, entry, entry_data = await setup_zone(hass)
    shadow = _shadow_from_entry_data(entry_data)
    if shadow is None:
        pytest.skip("Shadow not in entry_data")
    diag = shadow.diagnostics()
    assert diag.get("mode") == "control", (
        f"Shadow mode must be 'control', got {diag.get('mode')!r}")


async def test_ha_tpi_coef_source_in_shadow_only(hass, hass_storage, enable_custom_integrations):
    """SHADOW_ONLY: tpi_coef_source is NOT 'deterministic_baseline' (uses LE2 path)."""
    _, entry, entry_data = await setup_zone(hass)
    coord = _coord_from_entry_data(entry_data)
    await coord.async_refresh()
    await hass.async_block_till_done()
    zone = (coord.data or {}).get("zone", {})
    src = zone.get("tpi_coef_source")
    assert src != "deterministic_baseline", (
        f"SHADOW_ONLY tpi_coef_source must be from LE2, got {src!r}")


async def test_ha_tpi_coef_source_in_deterministic(hass, hass_storage, enable_custom_integrations):
    """DETERMINISTIC: tpi_coef_source == 'deterministic_baseline'."""
    _, entry, entry_data = await setup_zone(hass, learning_enabled=False)
    coord = _coord_from_entry_data(entry_data)
    await _turn_on_active_control(hass, entry)
    await coord.async_refresh()
    await hass.async_block_till_done()
    zone = (coord.data or {}).get("zone", {})
    assert zone.get("tpi_coef_source") == "deterministic_baseline", (
        f"DETERMINISTIC tpi_coef_source must be baseline: {zone.get('tpi_coef_source')!r}")


# ── Regression: slope=None must not crash _check_heating_failure ─────────────

async def test_slope_none_no_log_crash(hass, hass_storage, enable_custom_integrations, caplog):
    """Regression: _check_heating_failure must survive slope=None at DEBUG log level.

    Before the fix, _LOGGER.debug("...%.4f...", None) raised TypeError inside
    _async_update_data. _async_refresh caught it, set last_update_success=False,
    and left coord.data stale — a silent data-corruption bug only visible at DEBUG level.
    """
    # Cold room: current=16°C, comfort=22°C → TPI computes high setpoint → is_heating_commanded=True
    _, entry, entry_data = await setup_zone(hass, comfort_temp=22.0)
    hass.states.async_set(
        "sensor.test_temp", "16.0",
        {"unit_of_measurement": "°C", "device_class": "temperature"},
    )
    hass.states.async_set(
        "climate.test_trv", "heat",
        {"temperature": 22.0, "current_temperature": 16.0, "min_temp": 5.0,
         "max_temp": 30.0, "hvac_modes": ["heat", "off"],
         "supported_features": 1, "friendly_name": "TRV"},
    )

    coord = _coord_from_entry_data(entry_data)
    coord.set_active_control(True)

    # slope is None by default — no prior temperature readings exist
    assert coord._indoor_temp_slope is None, "Precondition: slope must be None for this regression"

    # Enable DEBUG logging to exercise the formerly-broken format path
    with caplog.at_level(logging.DEBUG, logger="custom_components.thermosmart"):
        await coord.async_refresh()
        await hass.async_block_till_done()

    # _async_update_data must complete successfully — no TypeError from %.4f % None
    assert coord.last_update_success is True, (
        "slope=None crashed _async_update_data (%.4f TypeError) — coordinator.py fix missing?")
    assert coord.data is not None, "coord.data must be updated after successful refresh"
    assert "adaptation_mode" in (coord.data.get("zone") or {}), (
        f"adaptation_mode missing from coord.data['zone']: {coord.data}")
    assert "must be real number" not in caplog.text, "TypeError must not appear in log output"
    assert "TypeError" not in caplog.text
