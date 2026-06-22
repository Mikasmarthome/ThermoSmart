"""Single dispatch boundary (Phase 19A, pure Python).

Exactly one place may turn a ``DeviceControlCommand`` into a real device write.
This module does NOT call Home Assistant; it enforces the single-path invariant and
records what would be sent. The live coordinator remains the only actual dispatcher
and is wired to go through this boundary. ``shadow_only`` commands are never sent.
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
