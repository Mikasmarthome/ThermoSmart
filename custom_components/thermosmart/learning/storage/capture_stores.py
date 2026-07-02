"""Live-reachable, write-nothing store-access facade for LE2 raw/episode capture.

Constructing ``LearningCaptureStores`` performs NO storage I/O — it only holds
a ``StoreFactory`` reference plus the canonical registries (built in
``capture_registry.py``) and hands back correctly-keyed store wrappers on
demand. This is intentionally a pure wiring foundation: nothing calls
``.save()``/``.load()`` on the returned handles from this module or from its
construction. That is a deliberate scope boundary for this step — see the
module-level docstring in ``capture_registry.py`` for the retention values
that will apply once an actual capture/persist call site is built on top of
this facade in a later step.

No HA imports beyond the existing ``StoreFactory`` abstraction. No control
modification of any kind.
"""
from __future__ import annotations

from typing import Optional

from ..raw_schemas import RawTrackName
from ..registry import EpisodeRegistry, RawTrackRegistry
from .capture_registry import build_episode_registry, build_raw_track_registry
from .stores import (
    EpisodesStore,
    RawSegmentIndexStore,
    RawSegmentStore,
    ResearchDailyStore,
    StoreFactory,
    SupportCriticalEventStore,
)


class LearningCaptureStores:
    """Per-zone, lazy accessor for LE2 raw segment / episode stores.

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
        raw_registry: Optional[RawTrackRegistry] = None,
        episode_registry: Optional[EpisodeRegistry] = None,
    ) -> None:
        self._factory = factory
        self._zone = learning_zone_id
        self._raw_registry = raw_registry if raw_registry is not None else build_raw_track_registry()
        self._episode_registry = (
            episode_registry if episode_registry is not None else build_episode_registry()
        )

    @property
    def learning_zone_id(self) -> str:
        return self._zone

    @property
    def raw_registry(self) -> RawTrackRegistry:
        return self._raw_registry

    @property
    def episode_registry(self) -> EpisodeRegistry:
        return self._episode_registry

    def raw_segment_store(self, track_name: RawTrackName) -> RawSegmentStore:
        """Return a RawSegmentStore for this zone's given track.

        Raises UnknownDefinitionError if the track is not a declared raw
        track — this is a programmer error to surface immediately, not a
        runtime condition to swallow.
        """
        self._raw_registry.get(track_name)  # validates the track is declared
        return RawSegmentStore(self._factory, self._zone, track_name)

    def raw_segment_index_store(self, track_name: RawTrackName) -> RawSegmentIndexStore:
        """Return a RawSegmentIndexStore for this zone's given track."""
        self._raw_registry.get(track_name)  # validates the track is declared
        return RawSegmentIndexStore(self._factory, self._zone, track_name)

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
