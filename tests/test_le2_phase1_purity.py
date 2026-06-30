"""Proves the Phase 1 LE 2.0 modules have no Home Assistant dependency.

The modules are loaded by file path into a synthetic standalone package in a
fresh subprocess, with a meta-path finder that raises on any ``homeassistant``
import. If any Phase 1 module imported Home Assistant (directly or indirectly),
the subprocess fails.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LEARNING_DIR = (
    Path(__file__).resolve().parents[1]
    / "custom_components" / "thermosmart" / "learning"
)

_SCRIPT = r'''
import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path

learning_dir = Path(sys.argv[1])


class _BlockHomeAssistant(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "homeassistant" or name.startswith("homeassistant."):
            raise ImportError("homeassistant import is forbidden in pure LE2 modules")
        return None


sys.meta_path.insert(0, _BlockHomeAssistant())

# Synthetic standalone package so relative imports resolve without importing
# the real custom_components.thermosmart parent (which depends on HA).
pkg = types.ModuleType("ts_phase1")
pkg.__path__ = [str(learning_dir)]
sys.modules["ts_phase1"] = pkg

for name in ("clock", "contracts", "raw_schemas", "episode_schemas"):
    spec = importlib.util.spec_from_file_location(
        f"ts_phase1.{name}", learning_dir / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ts_phase1.{name}"] = mod
    spec.loader.exec_module(mod)

assert "homeassistant" not in sys.modules, "homeassistant was imported transitively"
print("PURE_OK")
'''


def test_phase1_modules_import_without_home_assistant():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(LEARNING_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "PURE_OK" in result.stdout


def test_learning_module_sources_do_not_import_home_assistant():
    # Only actual import statements matter; docstrings may mention HA in prose.
    import re

    pattern = re.compile(r"^\s*(import\s+homeassistant|from\s+homeassistant)", re.MULTILINE)
    for name in ("clock", "contracts", "raw_schemas", "episode_schemas", "__init__"):
        source = (LEARNING_DIR / f"{name}.py").read_text(encoding="utf-8")
        assert not pattern.search(source), f"{name}.py imports homeassistant"
