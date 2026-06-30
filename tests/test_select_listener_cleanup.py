"""Unit tests for ThermoSmartModeSelect listener lifecycle (Welle 1, no hass needed).

Verifies the fix for the async_remove_listener bug:
  Before: async_add_listener return value discarded; async_remove_listener (non-existent) called
  After:  async_on_remove(coordinator.async_add_listener(cb)) — HA-idiomatic cancel pattern
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


def _make_mode_select():
    """Return (entity, coord, unsub_called) for a ThermoSmartModeSelect with mock coordinator."""
    from custom_components.thermosmart.select import ThermoSmartModeSelect

    unsub_called: list[bool] = []

    def _tracked_unsub():
        unsub_called.append(True)

    coord = MagicMock()
    coord.async_add_listener.return_value = _tracked_unsub
    coord._mode = "auto"

    entry = MagicMock()
    entry.entry_id = "test_entry_id"
    entry.data = {"name": "TestZone"}

    entity = ThermoSmartModeSelect(coord, entry)
    return entity, coord, unsub_called


async def _call_added_to_hass(entity, last_state=None):
    """Invoke async_added_to_hass while mocking out HA restore-state I/O."""
    with (
        patch(
            "homeassistant.helpers.restore_state.RestoreEntity.async_added_to_hass",
            new=AsyncMock(),
        ),
        patch.object(entity, "async_get_last_state", new=AsyncMock(return_value=last_state)),
    ):
        await entity.async_added_to_hass()


# ── Test 1: cancel callable passed to async_on_remove ────────────────────────

@pytest.mark.asyncio
async def test_cancel_callable_passed_to_async_on_remove():
    """async_on_remove receives exactly the unsub callable from async_add_listener."""
    entity, coord, unsub_called = _make_mode_select()

    received: list = []
    entity.async_on_remove = lambda cb: received.append(cb)

    await _call_added_to_hass(entity)

    assert coord.async_add_listener.call_count == 1
    assert len(received) == 1
    # The exact unsub callable must have been passed in
    received[0]()
    assert unsub_called == [True]


# ── Test 2: calling the on_remove callback unsubscribes the listener ──────────

@pytest.mark.asyncio
async def test_on_remove_callback_unsubscribes_listener():
    """When HA calls the on_remove callbacks, the coordinator listener is removed."""
    entity, coord, unsub_called = _make_mode_select()

    on_remove_callbacks: list = []
    entity.async_on_remove = lambda cb: on_remove_callbacks.append(cb)

    await _call_added_to_hass(entity)

    assert unsub_called == []
    for cb in on_remove_callbacks:
        cb()
    assert unsub_called == [True]


# ── Test 3: no double listener on simulated reload ────────────────────────────

@pytest.mark.asyncio
async def test_single_listener_after_simulated_reload():
    """Reload = remove old entity + add new entity → 1 add per setup, each unsub once."""
    from custom_components.thermosmart.select import ThermoSmartModeSelect

    # Per-call tracking: each async_add_listener call gets its own unsub counter
    unsub_counts: list[int] = []

    def _add_listener_side_effect(cb, context=None):
        idx = len(unsub_counts)
        unsub_counts.append(0)
        def _unsub():
            unsub_counts[idx] += 1
        return _unsub

    coord = MagicMock()
    coord.async_add_listener.side_effect = _add_listener_side_effect
    coord._mode = "auto"

    entry = MagicMock()
    entry.entry_id = "reload_test"
    entry.data = {"name": "Z"}

    # First entity (pre-reload)
    entity1 = ThermoSmartModeSelect(coord, entry)
    on_remove1: list = []
    entity1.async_on_remove = lambda cb: on_remove1.append(cb)
    await _call_added_to_hass(entity1)
    assert coord.async_add_listener.call_count == 1

    # Unload entity1
    for cb in on_remove1:
        cb()
    assert unsub_counts[0] == 1  # entity1's listener was removed

    # Second entity (post-reload)
    entity2 = ThermoSmartModeSelect(coord, entry)
    on_remove2: list = []
    entity2.async_on_remove = lambda cb: on_remove2.append(cb)
    await _call_added_to_hass(entity2)
    assert coord.async_add_listener.call_count == 2  # 1 per setup cycle, never 2 at once

    # Unload entity2
    for cb in on_remove2:
        cb()
    assert unsub_counts[1] == 1  # entity2's listener was also removed


# ── Test 4: multiple cleanup is safe (idempotent) ────────────────────────────

@pytest.mark.asyncio
async def test_multiple_unsub_calls_are_safe():
    """Calling the unsub callable multiple times does not raise."""
    call_count = []

    def _multi_unsub():
        call_count.append(1)

    from custom_components.thermosmart.select import ThermoSmartModeSelect

    coord = MagicMock()
    coord.async_add_listener.return_value = _multi_unsub
    coord._mode = "auto"
    entry = MagicMock()
    entry.entry_id = "x"
    entry.data = {"name": "Z"}

    entity = ThermoSmartModeSelect(coord, entry)
    on_remove: list = []
    entity.async_on_remove = lambda cb: on_remove.append(cb)

    await _call_added_to_hass(entity)

    # Call multiple times — should not raise
    for cb in on_remove:
        cb()
    for cb in on_remove:
        cb()

    assert len(call_count) == 2  # called twice, no exception


# ── Test 5: async_remove_listener is never called ────────────────────────────

@pytest.mark.asyncio
async def test_async_remove_listener_never_called():
    """The non-existent async_remove_listener method is never invoked."""
    entity, coord, _ = _make_mode_select()
    entity.async_on_remove = lambda cb: None

    await _call_added_to_hass(entity)

    assert not coord.async_remove_listener.called


# ── Test 6: coordinator not called after entity removal ──────────────────────

@pytest.mark.asyncio
async def test_coordinator_callback_not_invoked_after_removal():
    """Once the listener is removed, coordinator updates don't fire the callback."""
    write_state_calls: list = []

    from custom_components.thermosmart.select import ThermoSmartModeSelect

    # Build a coordinator that actually fires listeners
    listeners: dict = {}

    def _add_listener(cb, context=None):
        listeners[id(cb)] = cb
        def _unsub():
            listeners.pop(id(cb), None)
        return _unsub

    coord = MagicMock()
    coord.async_add_listener.side_effect = _add_listener
    coord._mode = "auto"

    entry = MagicMock()
    entry.entry_id = "y"
    entry.data = {"name": "Z"}

    entity = ThermoSmartModeSelect(coord, entry)
    entity.async_write_ha_state = lambda: write_state_calls.append(1)

    on_remove: list = []
    entity.async_on_remove = lambda cb: on_remove.append(cb)

    await _call_added_to_hass(entity)

    # Fire all listeners (coordinator update)
    for cb in list(listeners.values()):
        cb()
    assert len(write_state_calls) == 1

    # Remove entity → unsub
    for cb in on_remove:
        cb()

    # Fire again — entity callback must NOT be invoked
    for cb in list(listeners.values()):
        cb()
    assert len(write_state_calls) == 1  # still 1, not 2
