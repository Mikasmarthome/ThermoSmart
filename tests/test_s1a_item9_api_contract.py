"""Phase S1a Item 9 — API Contract Tests for LearningShadowController.

Background: In an earlier test draft, `shadow.async_flush_now()` was called.
This method never existed — the correct API is `async_flush()`.

This was a test-authoring error, not a production bug:
  - `LearningShadowController.async_flush()` has existed since LE2 integration
  - `async_flush_now()` was never a method on any LE2 class
  - No production code used `async_flush_now()`
  - The test at test_s1a_item9_ha_real_restart.py:258 was the only occurrence
  - Fix: renamed to `async_flush()` (one line change)

This test file prevents regression: calling a non-existent flush method
will now immediately fail with a clear test error.
"""
from __future__ import annotations

import inspect
import pytest

from custom_components.thermosmart.learning.runtime.ha_integration import (
    LearningShadowController,
)
from custom_components.thermosmart.learning.runtime.lifecycle import LearningRuntime


class TestAsyncFlushApiContract:
    """Prove that the flush API is exactly async_flush(), nothing else."""

    def test_async_flush_exists_on_shadow_controller(self):
        assert hasattr(LearningShadowController, "async_flush"), \
            "LearningShadowController must have async_flush()"

    def test_async_flush_is_coroutine(self):
        method = LearningShadowController.async_flush
        assert inspect.iscoroutinefunction(method), \
            "async_flush must be a coroutine function (async def)"

    def test_async_flush_now_does_not_exist(self):
        """async_flush_now() never existed — ensure it stays removed."""
        assert not hasattr(LearningShadowController, "async_flush_now"), \
            "async_flush_now must not exist — use async_flush() instead"

    def test_async_flush_exists_on_runtime(self):
        assert hasattr(LearningRuntime, "async_flush"), \
            "LearningRuntime must have async_flush()"

    def test_async_flush_is_coroutine_on_runtime(self):
        method = LearningRuntime.async_flush
        assert inspect.iscoroutinefunction(method), \
            "LearningRuntime.async_flush must be a coroutine function"

    def test_async_flush_now_does_not_exist_on_runtime(self):
        assert not hasattr(LearningRuntime, "async_flush_now"), \
            "LearningRuntime must not have async_flush_now — use async_flush()"

    @pytest.mark.asyncio
    async def test_async_flush_callable_on_runtime(self):
        """async_flush() must be callable without error on a fresh runtime."""
        from tests.helpers_runtime import MemoryStore
        from tests.helpers_runtime_scenarios import runtime, step
        rt = runtime(store=MemoryStore())
        await rt.async_setup()
        step(rt, 0, 19.0, heating=False)
        rt.mark_dirty(important=True)
        await rt.async_flush()  # must not raise
