"""S1a Item 10 - Section 4: Retention rules and age-based pruning.

Verifies BaselineComparisonStore age pruning, RetentionPolicy validation,
and that samples beyond the age window are discarded on restore and prune().
Runs on Windows (pure Python, no HA).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.models.baseline_comparison import (
    BaselineComparisonStore,
    BaselineParams,
    NonBoostBaselineSample,
)
from custom_components.thermosmart.learning.registry import (
    InvalidDefinitionError,
    RetentionPolicy,
)

UTC = timezone.utc
T0 = datetime(2025, 1, 1, 6, 0, 0, tzinfo=UTC)


def _sample(decision_id: str, reached_at: str) -> NonBoostBaselineSample:
    return NonBoostBaselineSample(
        episode_id=f"ep_{decision_id}", decision_id=decision_id,
        learning_zone_id="lz", started_at="2025-01-01T06:00:00+00:00",
        effective_heating_onset_at="2025-01-01T06:10:00+00:00",
        target_reached_at=reached_at,
        command_to_target_duration_s=3600.0, onset_to_target_duration_s=3000.0,
        start_deficit_c=2.0, target_temperature_c=21.0, start_temperature_c=19.0,
        external_temperature_c=-5.0, heat_rate_used=1.5, heat_loss_used=0.3,
        tpi_duty=0.7, baseline_setpoint_c=21.0, comfort_time_utc=None,
        time_of_day_bucket=6, presence_context=None, device_control_type="setpoint",
        dispatch_quality="full", sensor_reliability=0.95, confounder_flags=(),
        outcome_reliability=0.9, boost_applied_c=0.0, authoritative=True,
    )


# ── 1. BaselineComparisonStore age pruning ────────────────────────────────────


class TestBaselineAgeRetention:
    def test_default_max_age_is_30_days(self):
        params = BaselineParams()
        assert params.max_age_s == 30 * 24 * 3600.0

    def test_old_samples_pruned_on_explicit_prune(self):
        store = BaselineComparisonStore("lz", params=BaselineParams(max_age_s=3600.0))
        # Add sample from 2 hours ago (older than 1-hour max_age_s)
        old_ts = (T0 - timedelta(hours=2)).isoformat()
        store.add(_sample("dec_old", old_ts))
        assert store.size == 1
        now_ts = T0.isoformat()
        store.prune(now_ts)
        assert store.size == 0

    def test_fresh_samples_kept_on_prune(self):
        store = BaselineComparisonStore("lz", params=BaselineParams(max_age_s=7200.0))
        recent_ts = (T0 - timedelta(minutes=30)).isoformat()
        store.add(_sample("dec_fresh", recent_ts))
        store.prune(T0.isoformat())
        assert store.size == 1

    def test_mixed_age_prune_keeps_fresh_removes_old(self):
        store = BaselineComparisonStore("lz", params=BaselineParams(max_age_s=3600.0))
        old_ts = (T0 - timedelta(hours=3)).isoformat()
        recent_ts = (T0 - timedelta(minutes=30)).isoformat()
        store.add(_sample("dec_old", old_ts))
        store.add(_sample("dec_fresh", recent_ts))
        assert store.size == 2
        store.prune(T0.isoformat())
        assert store.size == 1
        kept = list(store.samples())
        assert kept[0].decision_id == "dec_fresh"

    def test_old_samples_pruned_on_restore(self):
        store = BaselineComparisonStore("lz", params=BaselineParams(max_age_s=3600.0))
        old_ts = (T0 - timedelta(hours=5)).isoformat()
        store.add(_sample("dec_old", old_ts))
        blob = store.serialize()
        store2 = BaselineComparisonStore("lz", params=BaselineParams(max_age_s=3600.0))
        store2.restore(blob, now_ts=T0.isoformat())
        assert store2.size == 0

    def test_fresh_samples_kept_on_restore(self):
        store = BaselineComparisonStore("lz", params=BaselineParams(max_age_s=7200.0))
        recent_ts = (T0 - timedelta(minutes=20)).isoformat()
        store.add(_sample("dec_fresh", recent_ts))
        blob = store.serialize()
        store2 = BaselineComparisonStore("lz", params=BaselineParams(max_age_s=7200.0))
        store2.restore(blob, now_ts=T0.isoformat())
        assert store2.size == 1


# ── 2. RetentionPolicy validation ─────────────────────────────────────────────


class TestRetentionPolicyValidation:
    def test_valid_policy_no_raises(self):
        p = RetentionPolicy(max_records=100, max_age_days=30)
        assert p.max_records == 100

    def test_zero_max_records_raises(self):
        with pytest.raises(InvalidDefinitionError):
            RetentionPolicy(max_records=0)

    def test_negative_max_records_raises(self):
        with pytest.raises(InvalidDefinitionError):
            RetentionPolicy(max_records=-1)

    def test_zero_max_age_days_raises(self):
        with pytest.raises(InvalidDefinitionError):
            RetentionPolicy(max_age_days=0)

    def test_negative_max_age_days_raises(self):
        with pytest.raises(InvalidDefinitionError):
            RetentionPolicy(max_age_days=-5)

    def test_both_none_is_valid(self):
        p = RetentionPolicy()
        assert p.max_records is None
        assert p.max_age_days is None

    def test_segmented_quality_seasonal_flags(self):
        p = RetentionPolicy(segmented=True, quality_prioritized=True, seasonal_stratification=True)
        assert p.segmented is True
        assert p.quality_prioritized is True
        assert p.seasonal_stratification is True


# ── 3. Rejection tracking ─────────────────────────────────────────────────────


class TestBaselineRejectionTracking:
    def test_rejection_count_increments(self):
        store = BaselineComparisonStore("lz")
        store.record_rejection("boost_applied")
        store.record_rejection("boost_applied")
        store.record_rejection("not_reached")
        counts = store.rejection_counts()
        assert counts["boost_applied"] == 2
        assert counts["not_reached"] == 1

    def test_rejection_counts_serialized_and_restored(self):
        store = BaselineComparisonStore("lz")
        store.record_rejection("low_reliability")
        blob = store.serialize()
        store2 = BaselineComparisonStore("lz")
        store2.restore(blob)
        counts = store2.rejection_counts()
        assert counts.get("low_reliability", 0) == 1

    def test_non_authoritative_sample_rejected(self):
        store = BaselineComparisonStore("lz")
        sample = _sample("dec_x", "2025-01-01T07:00:00+00:00")
        import dataclasses
        bad = dataclasses.replace(sample, authoritative=False)
        result = store.add(bad)
        assert result is False
        assert store.size == 0

    def test_boost_applied_nonzero_rejected(self):
        store = BaselineComparisonStore("lz")
        sample = _sample("dec_x", "2025-01-01T07:00:00+00:00")
        import dataclasses
        bad = dataclasses.replace(sample, boost_applied_c=1.5)
        result = store.add(bad)
        assert result is False
        assert store.size == 0
