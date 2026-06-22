"""Dirty tracking and declarative flush policy (pure Python).

No Home Assistant timer or coordinator tick is wired here. This module only
models *when* a save is warranted and guarantees correct dirty-state semantics
around a save: dirty survives failure, is cleared only for the exact saved
generation, and concurrent saves of the same store are prevented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from ..clock import Clock


class FlushReason(Enum):
    INTERVAL = "interval"
    SEGMENT_FULL = "segment_full"
    IMPORTANT_EVENT = "important_event"
    UNLOAD = "unload"
    SHUTDOWN = "shutdown"
    MANUAL = "manual"
    RECOVERY = "recovery"


class PersistenceError(Exception):
    """Raised on an invalid persistence operation (e.g. concurrent save)."""


@dataclass(frozen=True)
class BackoffPolicy:
    """Declarative retry/backoff configuration (no scheduling here)."""

    base_seconds: float = 5.0
    factor: float = 2.0
    max_seconds: float = 300.0

    def delay_for(self, retry_count: int) -> float:
        if retry_count <= 0:
            return 0.0
        raw = self.base_seconds * (self.factor ** (retry_count - 1))
        return min(raw, self.max_seconds)


@dataclass
class DirtyState:
    change_seq: int = 0          # increments on every change
    saved_seq: int = 0           # the change_seq durably saved
    first_dirty_utc: Optional[datetime] = None
    last_dirty_utc: Optional[datetime] = None
    last_saved_utc: Optional[datetime] = None
    save_in_progress: bool = False
    retry_count: int = 0
    last_error: Optional[str] = None

    @property
    def is_dirty(self) -> bool:
        return self.change_seq != self.saved_seq


@dataclass(frozen=True)
class SaveToken:
    """Captures the change generation a save attempt is responsible for."""

    key: str
    target_seq: int


class DirtyTracker:
    """Tracks dirty state per store key and coordinates safe saves."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._states: dict[str, DirtyState] = {}

    def state(self, key: str) -> DirtyState:
        return self._states.setdefault(key, DirtyState())

    def is_dirty(self, key: str) -> bool:
        return key in self._states and self._states[key].is_dirty

    def dirty_keys(self) -> tuple[str, ...]:
        return tuple(k for k, s in self._states.items() if s.is_dirty)

    # -- change ----------------------------------------------------------

    def mark_dirty(self, key: str) -> None:
        state = self.state(key)
        now = self._clock.now_utc()
        state.change_seq += 1
        if state.first_dirty_utc is None or not state.is_dirty:
            # starting a fresh dirty span
            state.first_dirty_utc = state.first_dirty_utc or now
        state.last_dirty_utc = now

    # -- save lifecycle --------------------------------------------------

    def begin_save(self, key: str) -> SaveToken:
        state = self.state(key)
        if state.save_in_progress:
            raise PersistenceError(f"a save for '{key}' is already in progress")
        if not state.is_dirty:
            raise PersistenceError(f"'{key}' is not dirty; nothing to save")
        state.save_in_progress = True
        return SaveToken(key=key, target_seq=state.change_seq)

    def complete_save(self, token: SaveToken) -> None:
        state = self.state(token.key)
        state.save_in_progress = False
        now = self._clock.now_utc()
        state.last_saved_utc = now
        state.saved_seq = token.target_seq
        state.retry_count = 0
        state.last_error = None
        if not state.is_dirty:
            state.first_dirty_utc = None  # fully clean
        # If new changes arrived during the save (change_seq advanced past the
        # token), the store remains dirty automatically (saved_seq < change_seq).

    def fail_save(self, token: SaveToken, error: str) -> None:
        state = self.state(token.key)
        state.save_in_progress = False
        state.retry_count += 1
        state.last_error = error
        # saved_seq is untouched -> the store stays dirty.


@dataclass(frozen=True)
class FlushDecision:
    should_flush: bool
    reason: Optional[FlushReason] = None


class FlushPolicy:
    """Declarative flush decisioning. No timer; callers ask for a decision."""

    def __init__(self, *, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise PersistenceError("interval_seconds must be > 0")
        self._interval = interval_seconds

    def decide(
        self,
        state: DirtyState,
        *,
        now: datetime,
        segment_full: bool = False,
        important_event: bool = False,
        forced_reason: Optional[FlushReason] = None,
    ) -> FlushDecision:
        if forced_reason in (FlushReason.UNLOAD, FlushReason.SHUTDOWN,
                             FlushReason.MANUAL, FlushReason.RECOVERY):
            return FlushDecision(state.is_dirty, forced_reason if state.is_dirty else None)
        if not state.is_dirty:
            return FlushDecision(False, None)
        if segment_full:
            return FlushDecision(True, FlushReason.SEGMENT_FULL)
        if important_event:
            return FlushDecision(True, FlushReason.IMPORTANT_EVENT)
        anchor = state.last_saved_utc or state.first_dirty_utc
        if anchor is not None and (now - anchor).total_seconds() >= self._interval:
            return FlushDecision(True, FlushReason.INTERVAL)
        return FlushDecision(False, None)
