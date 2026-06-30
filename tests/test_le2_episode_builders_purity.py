"""Proves the LE 2.0 episode builders are pure (no HA / storage / models)."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from custom_components.thermosmart.learning.contracts import Regime
from custom_components.thermosmart.learning.builders import (
    BuilderAction,
    BuilderInput,
    HeatingEpisodeBuilder,
)
from tests.helpers_builders import at, regime

LEARNING_DIR = (
    Path(__file__).resolve().parents[1]
    / "custom_components" / "thermosmart" / "learning"
)
BUILDERS_DIR = LEARNING_DIR / "builders"

_SCRIPT = r'''
import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path

learning_dir = Path(sys.argv[1])
builders_dir = learning_dir / "builders"


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "homeassistant" or name.startswith("homeassistant."):
            raise ImportError("homeassistant import is forbidden in pure modules")
        return None


sys.meta_path.insert(0, _Block())

root = types.ModuleType("ts_p6")
root.__path__ = [str(learning_dir)]
sys.modules["ts_p6"] = root
bpkg = types.ModuleType("ts_p6.builders")
bpkg.__path__ = [str(builders_dir)]
sys.modules["ts_p6.builders"] = bpkg

for name in ("clock", "contracts", "raw_schemas", "episode_schemas", "features",
             "regime", "registry"):
    spec = importlib.util.spec_from_file_location(f"ts_p6.{name}", learning_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ts_p6.{name}"] = mod
    spec.loader.exec_module(mod)

for name in ("base", "drive_end", "heating", "afterheat", "cooling",
             "window_cooling", "outcome", "registry_check"):
    spec = importlib.util.spec_from_file_location(
        f"ts_p6.builders.{name}", builders_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ts_p6.builders.{name}"] = mod
    spec.loader.exec_module(mod)

assert "homeassistant" not in sys.modules
print("PURE_OK")
'''


def test_builders_import_without_home_assistant():
    result = subprocess.run([sys.executable, "-c", _SCRIPT, str(LEARNING_DIR)],
                            capture_output=True, text=True)
    assert result.returncode == 0, (
        f"purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert "PURE_OK" in result.stdout


def test_builder_sources_have_no_forbidden_imports():
    pat_ha = re.compile(r"^\s*(import\s+homeassistant|from\s+homeassistant)", re.MULTILINE)
    pat_storage = re.compile(r"^\s*from\s+\.\.storage", re.MULTILINE)
    pat_models = re.compile(r"^\s*from\s+\.\.models", re.MULTILINE)
    for path in BUILDERS_DIR.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert not pat_ha.search(src), f"{path.name} imports homeassistant"
        assert not pat_storage.search(src), f"{path.name} imports storage"
        assert not pat_models.search(src), f"{path.name} imports models"


def test_input_not_mutated_and_deterministic():
    inp = BuilderInput(ts=at(0), regime=regime(Regime.ACTIVE_HEATING), indoor_temp=18.0,
                       target=21.0)
    a = HeatingEpisodeBuilder("lz_1")
    b = HeatingEpisodeBuilder("lz_1")
    ra = a.process(inp)
    rb = b.process(inp)
    assert ra.action is rb.action is BuilderAction.OPENED
    assert inp.indoor_temp == 18.0 and inp.target == 21.0  # unchanged
