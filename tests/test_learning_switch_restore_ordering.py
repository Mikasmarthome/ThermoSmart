"""The Learning switch state must be in place before the first control cycle.

The switch entity is created only after the zone's first coordinator refresh,
so its own restore comes too late for that cycle. Setup therefore reads the
state from the same Home Assistant restore data before the first refresh.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components import thermosmart as ts_init
from custom_components.thermosmart.const import DOMAIN
from tests.helpers import make_mock_hass

_ENTRY_ID = "zone_1"


class _Engine:
    """The part of LearningEngine the switch state lives in."""

    def __init__(self):
        self.enabled: dict[str, bool] = {}

    def set_zone_enabled(self, zone_id, enabled):
        self.enabled[zone_id] = enabled

    def is_zone_enabled(self, zone_id):
        return self.enabled.get(zone_id, True)


class _FirstRefreshReached(Exception):
    pass


def _entry(**data):
    entry = MagicMock()
    entry.entry_id = _ENTRY_ID
    entry.data = {"entry_type": "zone", "name": "Zone", **data}
    entry.options = {}
    return entry


def _restore_data(*, stored_state):
    """Registry + restore data fakes: the switch is known and has this stored state."""
    registry = MagicMock()
    registry.async_get_entity_id.return_value = (
        "switch.zone_learning" if stored_state is not None else None
    )
    last_states = {}
    if stored_state is not None:
        last_states["switch.zone_learning"] = SimpleNamespace(
            state=SimpleNamespace(state=stored_state)
        )
    return registry, SimpleNamespace(last_states=last_states)


async def _state_seen_by_the_first_refresh(entry, *, stored_state):
    engine = _Engine()
    hass = make_mock_hass()
    hass.data = {DOMAIN: {"learning_engine": engine}}
    seen = {}

    class _Coordinator:
        def __init__(self, *a, **kw):
            pass

        async def async_config_entry_first_refresh(self):
            seen["enabled"] = engine.is_zone_enabled(_ENTRY_ID)
            raise _FirstRefreshReached

    registry, restore = _restore_data(stored_state=stored_state)
    with patch.object(ts_init, "_async_register_card", AsyncMock()), \
         patch.object(ts_init, "WeatherEngine", MagicMock()), \
         patch.object(ts_init, "ThermoSmartCoordinator", _Coordinator), \
         patch.object(ts_init.er, "async_get", return_value=registry), \
         patch.object(ts_init.restore_state, "async_get", return_value=restore):
        with pytest.raises(_FirstRefreshReached):
            await ts_init.async_setup_entry(hass, entry)
    return seen["enabled"]


class TestFirstCycleAfterRestart:
    async def test_switch_that_was_off_is_off_for_the_first_refresh(self):
        assert await _state_seen_by_the_first_refresh(_entry(), stored_state="off") is False

    async def test_switch_that_was_on_is_on_for_the_first_refresh(self):
        assert await _state_seen_by_the_first_refresh(_entry(), stored_state="on") is True

    async def test_fresh_install_without_stored_state_defaults_to_on(self):
        assert await _state_seen_by_the_first_refresh(_entry(), stored_state=None) is True

    async def test_config_flag_off_still_wins_over_a_stored_on(self):
        entry = _entry(learning_enabled=False)
        assert await _state_seen_by_the_first_refresh(entry, stored_state="on") is False


class TestRestoredStateHelper:
    def test_unknown_switch_has_no_restored_state(self):
        registry, restore = _restore_data(stored_state=None)
        hass = make_mock_hass()
        with patch.object(ts_init.er, "async_get", return_value=registry), \
             patch.object(ts_init.restore_state, "async_get", return_value=restore):
            assert ts_init._restored_learning_switch_state(hass, _entry()) is None

    def test_known_switch_without_stored_state_has_none(self):
        registry, _ = _restore_data(stored_state="on")
        hass = make_mock_hass()
        with patch.object(ts_init.er, "async_get", return_value=registry), \
             patch.object(ts_init.restore_state, "async_get",
                          return_value=SimpleNamespace(last_states={})):
            assert ts_init._restored_learning_switch_state(hass, _entry()) is None

    def test_lookup_uses_the_switch_unique_id(self):
        registry, restore = _restore_data(stored_state="on")
        hass = make_mock_hass()
        with patch.object(ts_init.er, "async_get", return_value=registry), \
             patch.object(ts_init.restore_state, "async_get", return_value=restore):
            ts_init._restored_learning_switch_state(hass, _entry())
        registry.async_get_entity_id.assert_called_once_with("switch", DOMAIN, f"{_ENTRY_ID}_learning")
