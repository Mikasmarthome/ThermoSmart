"""Pytest configuration for ThermoSmart tests.

Two test tiers
--------------
Welle 1 — pure Python (this Windows/local dev environment):
  Tests import HA modules installed as packages, but the
  pytest-homeassistant-custom-component plugin (which provides the ``hass``,
  ``hass_storage``, and ``enable_custom_integrations`` fixtures) is DISABLED via
  ``addopts = -p no:homeassistant`` in setup.cfg.

  Tests that need those fixtures are excluded from collection on Windows below.

Welle 2+ — HA integration tests (Linux / WSL / Docker / CI):
  Run with ``--override-ini='addopts='`` to re-enable the plugin.  The HA
  fixtures then become available and the excluded files are collected normally.

Why platform-based gating instead of try/import?
  The earlier approach (``try: import homeassistant.runner``) broke when
  fcntl.py / resource.py stubs made the import succeed on Windows while the
  actual ``hass`` fixture (which needs POSIX socket.socketpair for asyncio's
  ProactorEventLoop) remains unavailable.  An explicit ``sys.platform`` check
  is honest and correct.
"""
from __future__ import annotations

import sys

# Files that require the ``hass`` / ``hass_storage`` / ``enable_custom_integrations``
# fixtures from pytest-homeassistant-custom-component.  On Windows the plugin is
# disabled (setup.cfg: addopts = -p no:homeassistant) so these tests cannot run.
# On Linux/CI the plugin is enabled via --override-ini='addopts=' and they run fully.
collect_ignore_glob: list[str] = []

if sys.platform == "win32":
    collect_ignore_glob = [
        "test_le2_ha_real_*.py",
        "test_config_flow.py",
        "test_options_flow.py",
    ]
