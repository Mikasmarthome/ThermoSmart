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
from custom_components.thermosmart.coordinator import ThermoSmartCoordinator
from custom_components.thermosmart.sensor import ThermoSmartStatusSensor
from custom_components.thermosmart.switch import ThermoSmartLearningSwitch
from tests.helpers import make_mock_hass

_ENTRY_ID = "zone_1"


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


async def _coordinator_at_first_refresh(entry, *, stored_state):
    """Run the real zone setup up to the coordinator's first refresh.

    The coordinator is the real class — real ``__init__``, real
    ``set_learning_switch`` and real ``_learning_mode_enabled`` — and only the
    refresh itself is replaced by a probe that records what the first control
    cycle would see and stops setup there.
    """
    hass = make_mock_hass()
    hass.data = {DOMAIN: {}}
    seen: dict = {}

    class _Probe(ThermoSmartCoordinator):
        async def async_config_entry_first_refresh(self):
            seen["effective"] = self._learning_mode_enabled(self.zone_cfg)
            seen["switch"] = self._learning_switch_on
            seen["active_control_initialized"] = self._active_control_initialized
            raise _FirstRefreshReached

    def _factory(*args, **kwargs):
        with patch("homeassistant.helpers.frame.report_usage"):
            return _Probe(*args, **kwargs)

    registry, restore = _restore_data(stored_state=stored_state)
    with patch.object(ts_init, "_async_register_card", AsyncMock()),          patch.object(ts_init, "WeatherEngine", MagicMock()),          patch.object(ts_init, "ThermoSmartCoordinator", _factory),          patch.object(ts_init.er, "async_get", return_value=registry),          patch.object(ts_init.restore_state, "async_get", return_value=restore):
        with pytest.raises(_FirstRefreshReached):
            await ts_init.async_setup_entry(hass, entry)
    return seen


class TestFirstCycleAfterRestart:
    @pytest.mark.parametrize(
        ("config_flag", "stored", "expected"),
        [
            (True, "on", True),
            (True, "off", False),
            (False, "on", False),
            (False, "off", False),
        ],
    )
    async def test_effective_learning_is_config_and_switch(self, config_flag, stored, expected):
        seen = await _coordinator_at_first_refresh(
            _entry(learning_enabled=config_flag), stored_state=stored)
        assert seen["effective"] is expected
        assert seen["switch"] is (stored == "on")

    async def test_fresh_install_without_stored_state_defaults_to_on(self):
        seen = await _coordinator_at_first_refresh(_entry(), stored_state=None)
        assert seen["effective"] is True and seen["switch"] is True

    async def test_active_control_is_not_restored_yet_when_learning_already_is(self):
        """Learning is restored before the first refresh; Active Control is restored
        later by its own switch entity, and until then its restore barrier holds."""
        seen = await _coordinator_at_first_refresh(_entry(), stored_state="off")
        assert seen["switch"] is False
        assert seen["active_control_initialized"] is False


def _real_coordinator(**cfg):
    from tests.helpers import make_coordinator
    return make_coordinator(cfg)


class TestEveryLearningStateTheCoordinatorCanBeIn:
    @pytest.mark.parametrize("config_flag", [True, False])
    @pytest.mark.parametrize("switch_on", [True, False])
    def test_matrix_on_the_real_coordinator(self, config_flag, switch_on):
        coord = _real_coordinator(learning_enabled=config_flag)
        coord.set_learning_switch(switch_on)
        assert coord._learning_mode_enabled(coord.zone_cfg) is (config_flag and switch_on)

    def test_restoring_the_switch_does_not_touch_active_control(self):
        coord = _real_coordinator()
        coord.set_learning_switch(False)
        assert coord._active_control_initialized is False
        coord.set_active_control(True)
        assert coord._learning_mode_enabled(coord.zone_cfg) is False
        assert coord._active_control_initialized is True


class TestSwitchEntityAndStatusSensorFollowTheCoordinator:
    async def test_toggling_the_entity_changes_the_coordinators_effective_state(self):
        coord = _real_coordinator()
        coord.async_request_refresh = AsyncMock()
        switch = ThermoSmartLearningSwitch(coord, _entry())
        switch.async_write_ha_state = MagicMock()

        await switch.async_turn_off()
        assert coord._learning_mode_enabled(coord.zone_cfg) is False
        await switch.async_turn_on()
        assert coord._learning_mode_enabled(coord.zone_cfg) is True

    @pytest.mark.parametrize(
        ("active", "learning", "expected"),
        [
            (False, False, "disabled"),
            (False, True, "observation_mode"),
            (True, False, "controlling_no_learning"),
        ],
    )
    def test_status_reports_the_effective_state(self, active, learning, expected):
        coord = _real_coordinator()
        coord._active_control = active
        coord.set_learning_switch(learning)
        coord.data = {"zone": {}}
        assert ThermoSmartStatusSensor(coord, _entry()).native_value == expected

    def test_status_with_config_flag_off_ignores_a_switch_that_is_on(self):
        coord = _real_coordinator(learning_enabled=False)
        coord._active_control = True
        coord.set_learning_switch(True)
        coord.data = {"zone": {}}
        assert ThermoSmartStatusSensor(coord, _entry()).native_value == "controlling_no_learning"

    def test_status_with_learning_and_active_control_on_is_none_of_the_learning_off_states(self):
        coord = _real_coordinator()
        coord._active_control = True
        coord.set_learning_switch(True)
        coord.data = {"zone": {}}
        assert ThermoSmartStatusSensor(coord, _entry()).native_value not in {
            "disabled", "observation_mode", "controlling_no_learning"}


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
