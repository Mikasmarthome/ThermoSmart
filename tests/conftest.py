"""Pytest configuration for ThermoSmart tests.

Welle 1 (pure Python):
  No HA fixtures needed. Tests import HA modules that are installed as a
  package, but the runner / event loop plugin is NOT loaded here since
  homeassistant.runner imports fcntl which is unavailable on Windows.

Welle 2+ (HA integration tests):
  Uncomment `pytest_plugins` below when running on Linux / WSL / CI where
  homeassistant.runner is available. The HA fixtures (hass, hass_storage,
  enable_custom_integrations) will then be available for integration tests.
"""
from __future__ import annotations

# Uncomment for Welle 2+ (requires Linux/WSL/CI — homeassistant.runner needs fcntl):
# pytest_plugins = "pytest_homeassistant_custom_component"
