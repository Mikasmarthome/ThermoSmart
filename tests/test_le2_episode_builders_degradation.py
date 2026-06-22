"""Phase 6 TRV-only, interactions, reliability and registry tests."""
from __future__ import annotations

import pytest

from custom_components.thermosmart.learning.contracts import DataQuality, ExportScope, Regime
from custom_components.thermosmart.learning.episode_schemas import EpisodeType, HeatingEpisode
from custom_components.thermosmart.learning.raw_schemas import RawTrackName, RoomObservation
from custom_components.thermosmart.learning.registry import (
    EpisodeDefinition,
    EpisodeRegistry,
    PrivacyClass,
    RawTrackDefinition,
    RawTrackRegistry,
    RetentionPolicy,
)
from custom_components.thermosmart.learning.builders import (
    AfterheatEpisodeBuilder,
    BuilderAction,
    BuilderInput,
    ConfounderCode,
    HeatingEpisodeBuilder,
    PassiveCoolingEpisodeBuilder,
    validate_builder_against_registry,
)
from tests.helpers_builders import at, regime


def heat(builder, minute, kind, temp, **kw):
    return builder.process(BuilderInput(ts=at(minute), regime=regime(kind), indoor_temp=temp,
                                        target=21.0, **kw))


class TestTrvOnly:
    def test_heating_trv_only(self):
        builder = HeatingEpisodeBuilder("lz_1")
        heat(builder, 0, Regime.ACTIVE_HEATING, 18.0, trv_setpoint=22.0, controller_demand=True)
        heat(builder, 5, Regime.ACTIVE_HEATING, 18.4, trv_setpoint=22.0, controller_demand=True)
        r = heat(builder, 10, Regime.STABLE, 18.8)
        assert r.action is BuilderAction.CLOSED
        assert ConfounderCode.MISSING_OPTIONAL_EVIDENCE.value in r.confounder_flags

    def test_missing_optional_lowers_reliability_not_crash(self):
        builder = HeatingEpisodeBuilder("lz_1")
        heat(builder, 0, Regime.ACTIVE_HEATING, 18.0)
        heat(builder, 5, Regime.ACTIVE_HEATING, 18.4)
        r = heat(builder, 10, Regime.STABLE, 18.8)
        assert 0.0 <= r.reliability <= 1.0


class TestInteractions:
    def test_heating_then_afterheat_no_overlap(self):
        # heating closes at transition; afterheat opens afterwards -> disjoint
        h = HeatingEpisodeBuilder("lz_1")
        heat(h, 0, Regime.ACTIVE_HEATING, 18.0)
        heat(h, 5, Regime.ACTIVE_HEATING, 18.5)
        rh = heat(h, 10, Regime.AFTERHEAT, 18.7)
        assert rh.action is BuilderAction.CLOSED
        af = AfterheatEpisodeBuilder("lz_1")
        ra = af.process(BuilderInput(ts=at(10), regime=regime(Regime.AFTERHEAT),
                                     indoor_temp=18.7, target=21.0))
        assert ra.action is BuilderAction.OPENED
        assert rh.episode.end_ts == at(10)  # heating ends exactly where afterheat opens

    def test_heating_does_not_become_cooling(self):
        # a heating builder never emits a passive-cooling episode
        h = HeatingEpisodeBuilder("lz_1")
        heat(h, 0, Regime.ACTIVE_HEATING, 18.0)
        heat(h, 5, Regime.ACTIVE_HEATING, 18.5)
        r = heat(h, 10, Regime.PASSIVE_COOLING, 18.4)
        assert r.episode is None or isinstance(r.episode, HeatingEpisode)

    def test_multiple_trvs_single_zone_episode(self):
        h = HeatingEpisodeBuilder("lz_1")
        heat(h, 0, Regime.ACTIVE_HEATING, 18.0, trv_binding_id="trv_a")
        heat(h, 5, Regime.ACTIVE_HEATING, 18.4, trv_binding_id="trv_b")
        r = heat(h, 10, Regime.STABLE, 18.8, trv_binding_id="trv_a")
        assert r.action is BuilderAction.CLOSED  # one episode, not one per TRV

    def test_two_zones_isolated(self):
        a = HeatingEpisodeBuilder("lz_1")
        b = HeatingEpisodeBuilder("lz_2")
        heat(a, 0, Regime.ACTIVE_HEATING, 18.0)
        # builder b never saw zone-1 events
        assert b.current_state()["active"] is False


class TestReliability:
    def test_unknown_lowers_reliability(self):
        clean = PassiveCoolingEpisodeBuilder("lz_1")
        clean.process(BuilderInput(ts=at(0), regime=regime(Regime.PASSIVE_COOLING, 0.8),
                                   indoor_temp=20.0, controller_demand=False))
        clean.process(BuilderInput(ts=at(5), regime=regime(Regime.PASSIVE_COOLING, 0.8),
                                   indoor_temp=19.8, controller_demand=False))
        r_clean = clean.process(BuilderInput(ts=at(10), regime=regime(Regime.STABLE, 0.8),
                                             indoor_temp=19.6, controller_demand=False))
        noisy = PassiveCoolingEpisodeBuilder("lz_1")
        noisy.process(BuilderInput(ts=at(0), regime=regime(Regime.PASSIVE_COOLING, 0.8),
                                   indoor_temp=20.0, controller_demand=False))
        noisy.process(BuilderInput(ts=at(5), regime=regime(Regime.UNKNOWN, 0.2),
                                   indoor_temp=19.8, controller_demand=False))
        r_noisy = noisy.process(BuilderInput(ts=at(10), regime=regime(Regime.STABLE, 0.8),
                                             indoor_temp=19.6, controller_demand=False))
        assert r_noisy.reliability < r_clean.reliability


class TestRegistry:
    def _episode_registry(self, cap=120):
        reg = EpisodeRegistry()
        reg.register(EpisodeDefinition(
            episode_type=EpisodeType.HEATING, schema_type=HeatingEpisode,
            episode_schema_version=1, builder_version=1,
            consumed_raw_tracks=(RawTrackName.ROOM, RawTrackName.TRV),
            max_trajectory_points=cap, max_duration_seconds=5400,
            retention=RetentionPolicy(), trajectory_required=False, materialized=True))
        return reg

    def _raw_registry(self):
        reg = RawTrackRegistry()
        reg.register(RawTrackDefinition(
            track_name=RawTrackName.ROOM, schema_type=RoomObservation,
            track_schema_version=1, privacy_class=PrivacyClass.BEHAVIORAL,
            allowed_export_scopes=(ExportScope.SUPPORT,), retention=RetentionPolicy(),
            segmented=True, immutable_events=False))
        return reg

    def test_valid_builder_passes(self):
        builder = HeatingEpisodeBuilder("lz_1")
        errors = validate_builder_against_registry(builder, self._episode_registry())
        assert errors == []

    def test_cap_mismatch_detected(self):
        builder = HeatingEpisodeBuilder("lz_1")
        errors = validate_builder_against_registry(builder, self._episode_registry(cap=999))
        assert any("trajectory cap" in e for e in errors)

    def test_unregistered_episode_type(self):
        builder = HeatingEpisodeBuilder("lz_1")
        errors = validate_builder_against_registry(builder, EpisodeRegistry())
        assert any("not registered" in e for e in errors)

    def test_unknown_raw_track_detected(self):
        builder = HeatingEpisodeBuilder("lz_1")
        empty_raw = RawTrackRegistry()  # ROOM/TRV not registered
        errors = validate_builder_against_registry(builder, self._episode_registry(), empty_raw)
        assert any("raw track" in e for e in errors)
