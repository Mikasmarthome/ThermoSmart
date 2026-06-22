"""Phase 3 tests for dirty tracking, flush policy, and snapshot/concurrency."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.thermosmart.learning.clock import FakeClock
from custom_components.thermosmart.learning.storage.persistence import (
    BackoffPolicy,
    DirtyTracker,
    FlushPolicy,
    FlushReason,
    PersistenceError,
)

UTC = timezone.utc
NOW = datetime(2026, 1, 15, 20, 0, 0, tzinfo=UTC)


class TestDirtyTracker:
    def test_mark_dirty(self):
        t = DirtyTracker(FakeClock(NOW))
        assert not t.is_dirty("k")
        t.mark_dirty("k")
        assert t.is_dirty("k")
        assert t.dirty_keys() == ("k",)

    def test_successful_save_clears_dirty(self):
        t = DirtyTracker(FakeClock(NOW))
        t.mark_dirty("k")
        token = t.begin_save("k")
        t.complete_save(token)
        assert not t.is_dirty("k")
        assert t.state("k").last_saved_utc == NOW

    def test_failed_save_keeps_dirty(self):
        t = DirtyTracker(FakeClock(NOW))
        t.mark_dirty("k")
        token = t.begin_save("k")
        t.fail_save(token, "disk error")
        assert t.is_dirty("k")
        assert t.state("k").retry_count == 1
        assert t.state("k").last_error == "disk error"
        assert t.state("k").save_in_progress is False

    def test_concurrent_save_rejected(self):
        t = DirtyTracker(FakeClock(NOW))
        t.mark_dirty("k")
        t.begin_save("k")
        with pytest.raises(PersistenceError):
            t.begin_save("k")  # already in progress

    def test_change_during_save_remains_dirty(self):
        clock = FakeClock(NOW)
        t = DirtyTracker(clock)
        t.mark_dirty("k")                 # change_seq 1
        token = t.begin_save("k")         # target_seq 1
        t.mark_dirty("k")                 # change_seq 2 (arrived during save)
        t.complete_save(token)            # acknowledges only seq 1
        assert t.is_dirty("k")            # seq 2 still pending

    def test_save_when_not_dirty_rejected(self):
        t = DirtyTracker(FakeClock(NOW))
        with pytest.raises(PersistenceError):
            t.begin_save("k")

    def test_second_save_after_change_clears(self):
        t = DirtyTracker(FakeClock(NOW))
        t.mark_dirty("k")
        t.complete_save(t.begin_save("k"))
        t.mark_dirty("k")
        t.complete_save(t.begin_save("k"))
        assert not t.is_dirty("k")


class TestFlushPolicy:
    def test_interval_not_yet_due(self):
        p = FlushPolicy(interval_seconds=900)
        clock = FakeClock(NOW)
        t = DirtyTracker(clock)
        t.mark_dirty("k")
        decision = p.decide(t.state("k"), now=NOW + timedelta(seconds=60))
        assert decision.should_flush is False

    def test_interval_due(self):
        p = FlushPolicy(interval_seconds=900)
        t = DirtyTracker(FakeClock(NOW))
        t.mark_dirty("k")
        decision = p.decide(t.state("k"), now=NOW + timedelta(seconds=901))
        assert decision.should_flush and decision.reason is FlushReason.INTERVAL

    def test_segment_full_flushes_immediately(self):
        p = FlushPolicy(interval_seconds=900)
        t = DirtyTracker(FakeClock(NOW))
        t.mark_dirty("k")
        decision = p.decide(t.state("k"), now=NOW, segment_full=True)
        assert decision.reason is FlushReason.SEGMENT_FULL

    def test_important_event_flushes_immediately(self):
        p = FlushPolicy(interval_seconds=900)
        t = DirtyTracker(FakeClock(NOW))
        t.mark_dirty("k")
        decision = p.decide(t.state("k"), now=NOW, important_event=True)
        assert decision.reason is FlushReason.IMPORTANT_EVENT

    def test_shutdown_forces_flush_when_dirty(self):
        p = FlushPolicy(interval_seconds=900)
        t = DirtyTracker(FakeClock(NOW))
        t.mark_dirty("k")
        decision = p.decide(t.state("k"), now=NOW, forced_reason=FlushReason.SHUTDOWN)
        assert decision.should_flush and decision.reason is FlushReason.SHUTDOWN

    def test_clean_store_not_flushed(self):
        p = FlushPolicy(interval_seconds=900)
        t = DirtyTracker(FakeClock(NOW))
        decision = p.decide(t.state("k"), now=NOW, forced_reason=FlushReason.SHUTDOWN)
        assert decision.should_flush is False


class TestBackoffPolicy:
    def test_no_delay_at_zero(self):
        assert BackoffPolicy().delay_for(0) == 0.0

    def test_exponential_growth_capped(self):
        b = BackoffPolicy(base_seconds=5, factor=2, max_seconds=30)
        assert b.delay_for(1) == 5
        assert b.delay_for(2) == 10
        assert b.delay_for(3) == 20
        assert b.delay_for(10) == 30  # capped
