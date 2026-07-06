"""Persistence orchestration for LE 2.0 shadow runtime (pure Python core).

Dirty tracking + an explicit, versioned save policy over an INJECTED async store
(duck-typed: ``async_load()`` / ``async_save(dict)``). Storage failures are
isolated and surfaced as warnings — they must never disable heating control.
Store keys stay under the ``thermosmart_le2__`` namespace (the key is chosen by
the injected store; this layer only decides WHEN and WHAT to persist).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Mapping, Optional, Protocol


class SaveTrigger(Enum):
    AUTHORITATIVE_CHANGE = "authoritative_change"
    IMPORTANT_OUTCOME = "important_outcome"
    PERIODIC = "periodic"
    SHUTDOWN = "shutdown"
    NONE = "none"


@dataclass(frozen=True)
class SavePolicy:
    debounce_s: float = 30.0
    max_pending_s: float = 3600.0          # at least hourly while dirty
    important_debounce_s: float = 2.0
    policy_version: int = 1


class AsyncStore(Protocol):
    async def async_load(self) -> Optional[Mapping[str, Any]]: ...
    async def async_save(self, data: Mapping[str, Any]) -> None: ...


@dataclass
class _DirtyState:
    dirty: bool = False
    important: bool = False
    last_save_iso: Optional[str] = None
    first_dirty_iso: Optional[str] = None
    last_dirty_iso: Optional[str] = None
    save_count: int = 0
    failed_saves: int = 0


class PersistenceOrchestrator:
    """Decides when to persist; performs isolated saves via the injected store."""

    # Bounded warnings buffer: keeps the most recent N save/load failure
    # markers. Diagnostics/export only ever need "what's failing now", not
    # an unbounded history since process start — a long-running session
    # with a persistent storage fault must not grow this without limit.
    _MAX_WARNINGS = 100

    def __init__(self, store: AsyncStore, *, policy: Optional[SavePolicy] = None) -> None:
        self._store = store
        self._policy = policy or SavePolicy()
        self._s = _DirtyState()
        self._warnings: list[str] = []

    @property
    def dirty(self) -> bool:
        return self._s.dirty

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def save_count(self) -> int:
        return self._s.save_count

    def mark_dirty(self, now_iso: str, *, important: bool = False) -> None:
        if not self._s.dirty:
            self._s.first_dirty_iso = now_iso
        self._s.dirty = True
        self._s.last_dirty_iso = now_iso
        if important:
            self._s.important = True

    def decide(self, now_iso: str) -> SaveTrigger:
        if not self._s.dirty:
            return SaveTrigger.NONE
        now = _parse(now_iso)
        last = _parse(self._s.last_save_iso)
        first_dirty = _parse(self._s.first_dirty_iso)
        last_dirty = _parse(self._s.last_dirty_iso)
        p = self._policy
        # hourly safety net while pending
        if first_dirty is not None and now is not None and \
                (now - first_dirty).total_seconds() >= p.max_pending_s:
            return SaveTrigger.PERIODIC
        if self._s.important and last_dirty is not None and now is not None and \
                (now - last_dirty).total_seconds() >= p.important_debounce_s:
            return SaveTrigger.IMPORTANT_OUTCOME
        if last_dirty is not None and now is not None and \
                (now - last_dirty).total_seconds() >= p.debounce_s:
            return SaveTrigger.AUTHORITATIVE_CHANGE
        if last is None and last_dirty is not None and now is not None and \
                (now - last_dirty).total_seconds() >= p.debounce_s:
            return SaveTrigger.AUTHORITATIVE_CHANGE
        return SaveTrigger.NONE

    async def maybe_save(self, now_iso: str, payload_builder) -> SaveTrigger:
        trigger = self.decide(now_iso)
        if trigger is SaveTrigger.NONE:
            return trigger
        await self._do_save(now_iso, payload_builder)
        return trigger

    async def flush(self, now_iso: str, payload_builder, *, force: bool = False) -> bool:
        """Flush pending state (unload/reload/shutdown); isolated on failure.

        With ``force`` the current state is persisted even when not dirty, so open
        builder snapshots survive a graceful reload/restart.
        """
        if not self._s.dirty and not force:
            return False
        return await self._do_save(now_iso, payload_builder)

    async def _do_save(self, now_iso: str, payload_builder) -> bool:
        try:
            payload = payload_builder()
            await self._store.async_save(payload)
        except Exception as err:  # storage failure must not break heating
            self._s.failed_saves += 1
            self._warnings.append(f"save_failed:{type(err).__name__}")
            if len(self._warnings) > self._MAX_WARNINGS:
                self._warnings = self._warnings[-self._MAX_WARNINGS:]
            return False
        self._s.dirty = False
        self._s.important = False
        self._s.first_dirty_iso = None
        self._s.last_save_iso = now_iso
        self._s.save_count += 1
        return True

    async def load(self) -> Optional[Mapping[str, Any]]:
        try:
            return await self._store.async_load()
        except Exception as err:
            self._warnings.append(f"load_failed:{type(err).__name__}")
            if len(self._warnings) > self._MAX_WARNINGS:
                self._warnings = self._warnings[-self._MAX_WARNINGS:]
            return None

    def health(self) -> dict:
        return {"dirty": self._s.dirty, "important": self._s.important,
                "last_save": self._s.last_save_iso, "save_count": self._s.save_count,
                "failed_saves": self._s.failed_saves, "policy_version": self._policy.policy_version}


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
