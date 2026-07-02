"""Tests for the P1 beta-prep cleanup: neutralize remaining public-visible
``"le2"`` values in sensor attributes and export "reason" fields.

Covers custom_components/thermosmart/sensor.py (Heat-Rate/Heat-Loss sensor
``extra_state_attributes()["source"]``) and custom_components/thermosmart/
export.py (the "reason": "le2_shadow_unavailable" value used by every
``_le2_*_export()`` missing-shadow branch).

This is a value-only neutralization — no export key/layout change, no
entity-id/unique-id/attribute-key change, no storage-key/prefix change, no
runtime/control logic change. Internal names (``_le2_shadow``, ``_le2_*``
helpers, ``thermosmart_le2__`` storage prefix, ``test_le2_*`` filenames) are
explicitly out of scope and untouched.

9 test groups:
  T1  — ThermoSmartHeatingPowerSensor attrs report source "learning_engine"
  T2  — ThermoSmartHeatLossRateSensor attrs report source "learning_engine"
  T3  — No sensor attribute value "le2" for these two sensors
  T4  — Export unavailable reason is "learning_engine_unavailable"
  T5  — Export no longer contains "le2_shadow_unavailable"
  T6  — Export block key set unchanged (no layout change)
  T7  — No storage-key/prefix change (thermosmart_le2__ untouched)
  T8  — No entity-id/unique-id change on the two sensors
  T9  — No control-path keywords touched
"""
from __future__ import annotations

import inspect

import pytest

from custom_components.thermosmart import export as _export_module
from custom_components.thermosmart import sensor as _sensor_module
from custom_components.thermosmart.sensor import (
    ThermoSmartHeatingPowerSensor,
    ThermoSmartHeatLossRateSensor,
)


class _FakeShadow:
    def __init__(self, *, rate: float = 1.2, status: str = "valid"):
        self._rate = rate
        self._status = status

    def read_heat_rate_safe(self):
        return self._rate, self._status

    def read_heat_loss_rate_safe(self):
        return self._rate, self._status


class _FakeCoordinator:
    def __init__(self, shadow=None):
        self._le2_shadow = shadow


def _make_sensor(cls, shadow):
    """Construct a real sensor entity via object.__new__(), bypassing
    CoordinatorEntity.__init__ (no HA instance needed) — only sets the one
    attribute (.coordinator) extra_state_attributes() actually reads."""
    sensor = object.__new__(cls)
    sensor.coordinator = _FakeCoordinator(shadow)
    return sensor


# ── T1: Heating power (Heat-Rate) sensor reports "learning_engine" ─────────

def test_t1_heating_power_sensor_source_is_learning_engine():
    sensor = _make_sensor(ThermoSmartHeatingPowerSensor, _FakeShadow())
    attrs = sensor.extra_state_attributes
    assert attrs["source"] == "learning_engine"


def test_t1_heating_power_sensor_source_unavailable_when_no_shadow():
    sensor = _make_sensor(ThermoSmartHeatingPowerSensor, None)
    attrs = sensor.extra_state_attributes
    assert attrs["source"] == "unavailable"


# ── T2: Heat-Loss Rate sensor reports "learning_engine" ─────────────────────

def test_t2_heat_loss_rate_sensor_source_is_learning_engine():
    sensor = _make_sensor(ThermoSmartHeatLossRateSensor, _FakeShadow())
    attrs = sensor.extra_state_attributes
    assert attrs["source"] == "learning_engine"


def test_t2_heat_loss_rate_sensor_source_unavailable_when_no_shadow():
    sensor = _make_sensor(ThermoSmartHeatLossRateSensor, None)
    attrs = sensor.extra_state_attributes
    assert attrs["source"] == "unavailable"


# ── T3: No sensor attribute value "le2" for these two sensors ──────────────

def test_t3_no_le2_value_in_heating_power_attrs():
    sensor = _make_sensor(ThermoSmartHeatingPowerSensor, _FakeShadow())
    attrs = sensor.extra_state_attributes
    assert "le2" not in str(attrs.values())


def test_t3_no_le2_value_in_heat_loss_attrs():
    sensor = _make_sensor(ThermoSmartHeatLossRateSensor, _FakeShadow())
    attrs = sensor.extra_state_attributes
    assert "le2" not in str(attrs.values())


def test_t3_no_le2_source_literal_remains_in_sensor_module_source():
    source = inspect.getsource(_sensor_module)
    assert '"source": "le2"' not in source


# ── T4: Export unavailable reason is "learning_engine_unavailable" ─────────

def test_t4_learning_progress_export_unavailable_reason():
    from custom_components.thermosmart.export import _le2_learning_progress_export

    class _Coord:
        _le2_shadow = None

    block = _le2_learning_progress_export(_Coord())
    assert block == {"available": False, "reason": "learning_engine_unavailable"}


def test_t4_episode_history_export_unavailable_reason():
    from custom_components.thermosmart.export import _le2_episode_history_export
    from datetime import datetime, timezone

    class _Coord:
        _le2_shadow = None

    block = _le2_episode_history_export(_Coord(), now=datetime(2026, 6, 15, tzinfo=timezone.utc))
    assert block == {"available": False, "reason": "learning_engine_unavailable"}


def test_t4_research_daily_export_unavailable_reason():
    from custom_components.thermosmart.export import _le2_research_daily_export
    from datetime import datetime, timezone

    class _Coord:
        _le2_shadow = None

    block = _le2_research_daily_export(_Coord(), now=datetime(2026, 6, 15, tzinfo=timezone.utc))
    assert block == {"available": False, "reason": "learning_engine_unavailable"}


def test_t4_critical_events_export_unavailable_reason():
    from custom_components.thermosmart.export import _le2_critical_events_export
    from datetime import datetime, timezone

    class _Coord:
        _le2_shadow = None

    block = _le2_critical_events_export(_Coord(), now=datetime(2026, 6, 15, tzinfo=timezone.utc))
    assert block == {"available": False, "reason": "learning_engine_unavailable"}


# ── T5: Export no longer contains "le2_shadow_unavailable" ─────────────────

def test_t5_no_le2_shadow_unavailable_literal_in_export_module():
    source = inspect.getsource(_export_module)
    assert "le2_shadow_unavailable" not in source


# ── T6: Export block key set unchanged (no layout change) ──────────────────

def test_t6_unavailable_block_key_set_unchanged():
    from custom_components.thermosmart.export import _le2_learning_progress_export

    class _Coord:
        _le2_shadow = None

    block = _le2_learning_progress_export(_Coord())
    assert set(block.keys()) == {"available", "reason"}


# ── T7: No storage-key/prefix change ────────────────────────────────────────

def test_t7_storage_prefix_untouched():
    from custom_components.thermosmart.learning.storage import naming
    assert naming.research_daily_key("zone_alpha_01").startswith("thermosmart_le2__")
    assert naming.support_critical_events_key("zone_alpha_01").startswith("thermosmart_le2__")


# ── T8: No entity-id/unique-id change on the two sensors ───────────────────

def test_t8_heating_power_unique_id_pattern_unchanged():
    source = inspect.getsource(ThermoSmartHeatingPowerSensor.__init__)
    assert '"heating_power"' in source
    assert "_heating_power" in source


def test_t8_heat_loss_unique_id_pattern_unchanged():
    source = inspect.getsource(ThermoSmartHeatLossRateSensor.__init__)
    assert '"heat_loss"' in source
    assert "_heat_loss_rate" in source


# ── T9: No control-path keywords touched ─────────────────────────────────────

_FORBIDDEN_CONTROL_TOKENS = (
    "set_temperature",
    "async_set_temperature",
    "async_write_ha_state",
    "dispatch",
    "service_call",
)


def test_t9_no_control_keywords_in_changed_sensor_properties():
    for source in (
        inspect.getsource(ThermoSmartHeatingPowerSensor.extra_state_attributes.fget),
        inspect.getsource(ThermoSmartHeatLossRateSensor.extra_state_attributes.fget),
    ):
        lowered = source.lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            assert token not in lowered, f"forbidden control token found: {token}"
