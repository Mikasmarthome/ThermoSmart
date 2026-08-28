"""HACS default#9081 maintainer review — item 7: cleanup must survive a Home
Assistant restart, not rely solely on an in-session timer.

Covers:
  - async_cleanup_expired_exports() removes files older than the retention
    window on a startup scan (the restart-safe path)
  - a fresh (not-yet-expired) file is left alone
  - idempotent: calling it twice in a row does not raise or double-delete
  - a missing export directory does not raise
  - a filesystem error while removing one file does not abort the scan for
    the remaining files
  - a stray/foreign file (not matching ThermoSmart's own naming pattern)
    still gets cleaned up eventually via mtime, rather than lingering forever
  - the whole scan runs inside a single executor job (no blocking I/O on the
    event loop)
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.thermosmart.export import (
    _EXPORT_CLEANUP_DELAY_S,
    _export_dir,
    async_cleanup_expired_exports,
)


def _mock_hass(tmp_path):
    hass = MagicMock()
    hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a: fn(*a))
    return hass


def _touch_with_mtime(path: Path, age_seconds: float) -> None:
    path.write_text("{}", encoding="utf-8")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))


def _named_file(export_dir: Path, kind: str, age_seconds: float) -> Path:
    ts = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).strftime("%Y%m%dT%H%M%S")
    path = export_dir / f"thermosmart_{kind}_{ts}_abcdef.json"
    path.write_text("{}", encoding="utf-8")
    return path


class TestRestartSafeCleanup:
    async def test_expired_file_removed_on_startup_scan(self, tmp_path):
        hass = _mock_hass(tmp_path)
        export_dir = Path(_export_dir(hass))
        export_dir.mkdir(parents=True)
        old_file = _named_file(export_dir, "research", _EXPORT_CLEANUP_DELAY_S + 3600)

        await async_cleanup_expired_exports(hass)

        assert not old_file.exists()

    async def test_fresh_file_is_kept(self, tmp_path):
        hass = _mock_hass(tmp_path)
        export_dir = Path(_export_dir(hass))
        export_dir.mkdir(parents=True)
        new_file = _named_file(export_dir, "support", 60)  # 1 minute old

        await async_cleanup_expired_exports(hass)

        assert new_file.exists()

    async def test_idempotent_second_call_does_not_raise(self, tmp_path):
        hass = _mock_hass(tmp_path)
        export_dir = Path(_export_dir(hass))
        export_dir.mkdir(parents=True)
        _named_file(export_dir, "research", _EXPORT_CLEANUP_DELAY_S + 100)

        await async_cleanup_expired_exports(hass)
        await async_cleanup_expired_exports(hass)  # must not raise on already-clean dir

    async def test_missing_export_directory_does_not_raise(self, tmp_path):
        hass = _mock_hass(tmp_path)
        # Directory was never created — first-ever startup, nothing exported yet.
        await async_cleanup_expired_exports(hass)  # must not raise

    async def test_filesystem_error_on_one_file_does_not_abort_scan(self, tmp_path, monkeypatch):
        hass = _mock_hass(tmp_path)
        export_dir = Path(_export_dir(hass))
        export_dir.mkdir(parents=True)
        broken = _named_file(export_dir, "research", _EXPORT_CLEANUP_DELAY_S + 100)
        also_expired = _named_file(export_dir, "support", _EXPORT_CLEANUP_DELAY_S + 100)

        real_remove = os.remove

        def flaky_remove(path):
            if path == str(broken):
                raise OSError("simulated permission error")
            return real_remove(path)

        monkeypatch.setattr(os, "remove", flaky_remove)

        await async_cleanup_expired_exports(hass)  # must not raise

        assert broken.exists()          # failed removal left in place, not crashed
        assert not also_expired.exists()  # the other expired file was still removed

    async def test_foreign_file_cleaned_up_via_mtime_fallback(self, tmp_path):
        hass = _mock_hass(tmp_path)
        export_dir = Path(_export_dir(hass))
        export_dir.mkdir(parents=True)
        stray = export_dir / "not_a_thermosmart_file.json"
        _touch_with_mtime(stray, _EXPORT_CLEANUP_DELAY_S + 100)

        await async_cleanup_expired_exports(hass)

        assert not stray.exists()

    async def test_foreign_fresh_file_not_removed(self, tmp_path):
        hass = _mock_hass(tmp_path)
        export_dir = Path(_export_dir(hass))
        export_dir.mkdir(parents=True)
        stray = export_dir / "not_a_thermosmart_file.json"
        _touch_with_mtime(stray, 60)

        await async_cleanup_expired_exports(hass)

        assert stray.exists()

    async def test_scan_runs_in_single_executor_job(self, tmp_path):
        hass = MagicMock()
        hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))
        export_dir = Path(_export_dir(hass))
        export_dir.mkdir(parents=True)
        _named_file(export_dir, "research", _EXPORT_CLEANUP_DELAY_S + 100)

        calls = []

        async def tracking_executor_job(fn, *args):
            calls.append(fn)
            return fn(*args)

        hass.async_add_executor_job = tracking_executor_job
        await async_cleanup_expired_exports(hass)
        assert len(calls) == 1
