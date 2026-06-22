"""Proves the LE 2.0 ThermalRegimeClassifier is pure (no HA / storage / models)."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from custom_components.thermosmart.learning.contracts import Regime
from custom_components.thermosmart.learning.features import FeatureExtractor
from custom_components.thermosmart.learning.episode_schemas import Trajectory, TrajectoryPoint
from custom_components.thermosmart.learning.regime import RegimeInput, ThermalRegimeClassifier

LEARNING_DIR = (
    Path(__file__).resolve().parents[1]
    / "custom_components" / "thermosmart" / "learning"
)
STEP_MS = 5 * 60 * 1000

_SCRIPT = r'''
import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path

learning_dir = Path(sys.argv[1])


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "homeassistant" or name.startswith("homeassistant."):
            raise ImportError("homeassistant import is forbidden in pure modules")
        return None


sys.meta_path.insert(0, _Block())

pkg = types.ModuleType("ts_p5")
pkg.__path__ = [str(learning_dir)]
sys.modules["ts_p5"] = pkg

for name in ("clock", "contracts", "raw_schemas", "episode_schemas", "features", "regime"):
    spec = importlib.util.spec_from_file_location(f"ts_p5.{name}", learning_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ts_p5.{name}"] = mod
    spec.loader.exec_module(mod)

assert "homeassistant" not in sys.modules
print("PURE_OK")
'''


def test_regime_imports_without_home_assistant():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(LEARNING_DIR)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "PURE_OK" in result.stdout


def test_regime_source_has_no_forbidden_imports():
    source = (LEARNING_DIR / "regime.py").read_text(encoding="utf-8")
    patterns = [
        re.compile(r"^\s*(import\s+homeassistant|from\s+homeassistant)", re.MULTILINE),
        re.compile(r"^\s*from\s+\.storage", re.MULTILINE),
        re.compile(r"^\s*from\s+\.{1,2}models", re.MULTILINE),
        re.compile(r"^\s*from\s+\.registry", re.MULTILINE),
    ]
    for pat in patterns:
        assert not pat.search(source), f"regime.py matches forbidden import {pat.pattern!r}"


def test_input_not_mutated():
    pts = tuple(TrajectoryPoint(i * STEP_MS, float(v))
                for i, v in enumerate([20.0, 19.8, 19.6, 19.4]))
    traj = Trajectory(points=pts, max_points=120)
    fs = FeatureExtractor().extract_trajectory_features(traj)
    inp = RegimeInput(controller_demand=False, trajectory_features=fs)
    before = tuple((p.offset_ms, p.value) for p in traj.points)
    ThermalRegimeClassifier().classify(inp)
    after = tuple((p.offset_ms, p.value) for p in traj.points)
    assert before == after
    assert inp.controller_demand is False  # input object unchanged


def test_deterministic_across_instances():
    fs = FeatureExtractor().extract_trajectory_features(
        Trajectory(points=tuple(TrajectoryPoint(i * STEP_MS, float(v))
                                for i, v in enumerate([20.0, 20.3, 20.45, 20.5])),
                   max_points=120))
    inp = RegimeInput(controller_demand=False, trajectory_features=fs)
    a = ThermalRegimeClassifier().classify(inp)
    b = ThermalRegimeClassifier().classify(inp)
    assert a.regime is b.regime and a.reliability == b.reliability
