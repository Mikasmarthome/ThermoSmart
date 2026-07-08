"""Typed initialization / reset state for Learning (pure Python, no Home Assistant).

These types describe the outcome of zone initialization and reset. They are pure
so they can be imported and asserted without Home Assistant. The actual store
work lives in ``reset.py`` (HA-dependent).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

INIT_VERSION = 1
CONTAINER_VERSION = 1


class InitializationStatus(Enum):
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    RECOVERY_REQUIRED = "recovery_required"
    FAILED_RECOVERABLE = "failed_recoverable"
    FAILED_TERMINAL = "failed_terminal"


class InitializationAction(Enum):
    INITIALIZED_NOW = "initialized_now"
    ALREADY_INITIALIZED = "already_initialized"
    RECOVERED_PARTIAL = "recovered_partial"
    FAILED = "failed"


class ResetReason(Enum):
    MANUAL = "manual"
    CORRUPTION = "corruption"
    SCHEMA_INCOMPATIBLE = "schema_incompatible"
    TEST = "test"


class InitializationError(Exception):
    """Raised on an unrecoverable initialization fault (terminal)."""


@dataclass(frozen=True)
class InitializationParameters:
    create_global_index: bool = False
    parameter_version: int = 1


@dataclass(frozen=True)
class InitializationState:
    """A read-only view of a zone's current initialization."""

    status: InitializationStatus
    learning_zone_id: Optional[str] = None
    container_version: Optional[int] = None
    present_components: tuple[str, ...] = ()
    missing_components: tuple[str, ...] = ()
    corrupted_components: tuple[str, ...] = ()
    v1_detected: bool = False
    init_version: int = INIT_VERSION


@dataclass(frozen=True)
class InitializationResult:
    """The structured outcome of an ``ensure_v2_initialized`` call."""

    status: InitializationStatus
    action: InitializationAction
    learning_zone_id: Optional[str] = None
    created_components: tuple[str, ...] = ()
    reused_components: tuple[str, ...] = ()
    failed_component: Optional[str] = None
    v1_detected: bool = False
    v1_import_skipped: bool = True
    info_log_emitted: bool = False
    init_version: int = INIT_VERSION
    timestamp_utc: Optional[str] = None
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is InitializationStatus.INITIALIZED
