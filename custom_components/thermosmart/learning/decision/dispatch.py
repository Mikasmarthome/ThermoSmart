"""Shadow-mode dispatch recorder (Phase 19A, pure Python).

Records what a ``DeviceControlCommand`` would send without calling Home Assistant.
Used by the trace pipeline (``compute_decision_trace_safe``) to validate the
resolved command in SHADOW mode.

NOTE (Phase B1): The live coordinator dispatches via
``trv_control._apply_temperature`` and ``_async_set_valve_percent`` directly —
not through this class. ``SingleDispatcher`` has no sink in the trace pipeline
and never performs a real HA service call. ``shadow_only`` commands are never sent.
"""
from __future__ import annotations

from typing import Callable, Optional

from .contracts import DeviceControlCommand, DispatchResult


class SingleDispatcher:
    """The one dispatch path. A sink (the existing controller apply) may be injected;
    in SHADOW or without a sink, nothing is sent and the command is only recorded."""

    def __init__(self, sink: Optional[Callable[[DeviceControlCommand], None]] = None) -> None:
        self._sink = sink
        self.last_command: Optional[DeviceControlCommand] = None
        self.dispatch_count = 0

    def dispatch(self, command: DeviceControlCommand) -> DispatchResult:
        self.last_command = command
        if command.shadow_only:
            return DispatchResult(command.zone_id, False, command, True, "shadow_only")
        if self._sink is None:
            return DispatchResult(command.zone_id, False, command, True, "no_sink")
        self._sink(command)
        self.dispatch_count += 1
        return DispatchResult(command.zone_id, True, command, False, command.source_reason)
