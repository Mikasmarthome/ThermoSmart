"""Store-access facade for Learning episode capture.

Constructing ``LearningCaptureStores`` performs NO storage I/O — it only holds
a ``StoreFactory`` reference plus the canonical registries (built in
``capture_registry.py``) and hands back correctly-keyed store wrappers on
demand. The returned handles ARE actively saved/loaded through the runtime
shadow controller (``learning/runtime/ha_integration.py``'s
``record_completed_episode_safe()``, and its episode/support-event/
research-daily load/save methods) — see ``capture_registry.py`` for the
retention values applied before each save.

No HA imports beyond the existing ``StoreFactory`` abstraction. No control
modification of any kind.
"""
from __future__ import annotations

from typing import Optional

from ..registry import EpisodeRegistry
from .capture_registry import build_episode_registry
from .stores import (
    EpisodesStore,
    ResearchDailyStore,
    StoreFactory,
    SupportCriticalEventStore,
)


class LearningCaptureStores:
    """Per-zone, lazy accessor for the Learning episode / support / research stores.

    Holds only a ``StoreFactory`` + registries — no cached store instances,
    no open file handles, no background tasks, no writes. Every accessor call
    creates a fresh, correctly-keyed store wrapper; the caller decides if and
    when to actually read/write through it.
    """

    def __init__(
        self,
        factory: StoreFactory,
        learning_zone_id: str,
        *,
        episode_registry: Optional[EpisodeRegistry] = None,
    ) -> None:
        self._factory = factory
        self._zone = learning_zone_id
        self._episode_registry = (
            episode_registry if episode_registry is not None else build_episode_registry()
        )

    @property
    def learning_zone_id(self) -> str:
        return self._zone

    @property
    def episode_registry(self) -> EpisodeRegistry:
        return self._episode_registry

    def episodes_store(self) -> EpisodesStore:
        """Return the single per-zone EpisodesStore (all episode types share it)."""
        return EpisodesStore(self._factory, self._zone)

    def support_critical_events_store(self) -> SupportCriticalEventStore:
        """Return the single per-zone SupportCriticalEventStore.

        Foundation only — nothing calls ``.load()``/``.save()`` on the
        returned handle from anywhere in the live runtime yet (see
        ``support_event_persistence.py``'s module docstring).
        """
        return SupportCriticalEventStore(self._factory, self._zone)

    def research_daily_store(self) -> ResearchDailyStore:
        """Return the single per-zone ResearchDailyStore.

        Reached from ``LearningShadowController``'s
        ``_async_load_research_daily_safe()``/
        ``_async_save_research_daily_safe()`` (see
        ``research_daily_persistence.py``'s module docstring for the pure
        merge/persist pipeline the controller calls into before saving
        through the returned handle).
        """
        return ResearchDailyStore(self._factory, self._zone)
