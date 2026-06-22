"""Proves the LE 2.0 ConfidenceAggregator is pure (no HA / storage / coordinator / models)."""
from __future__ import annotations

import re
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


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "homeassistant" or name.startswith("homeassistant."):
            raise ImportError("homeassistant import is forbidden in pure modules")
        return None


sys.meta_path.insert(0, _Block())

root = types.ModuleType("ts_p12")
root.__path__ = [str(learning_dir)]
sys.modules["ts_p12"] = root

for name in ("clock", "contracts", "confidence"):
    spec = importlib.util.spec_from_file_location(f"ts_p12.{name}", learning_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ts_p12.{name}"] = mod
    spec.loader.exec_module(mod)

assert "homeassistant" not in sys.modules
print("PURE_OK")
'''


def test_confidence_imports_without_home_assistant():
    result = subprocess.run([sys.executable, "-c", _SCRIPT, str(LEARNING_DIR)],
                            capture_output=True, text=True)
    assert result.returncode == 0, (
        f"purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert "PURE_OK" in result.stdout


def test_confidence_source_no_forbidden_imports():
    src = (LEARNING_DIR / "confidence.py").read_text(encoding="utf-8")
    for pat in (
        re.compile(r"^\s*(import\s+homeassistant|from\s+homeassistant)", re.MULTILINE),
        re.compile(r"^\s*from\s+\.storage", re.MULTILINE),
        re.compile(r"^\s*from\s+\.coordinator", re.MULTILINE),
        re.compile(r"^\s*from\s+\.models", re.MULTILINE),
        re.compile(r"^\s*from\s+\.builders", re.MULTILINE),
        re.compile(r"^\s*import\s+\w*coordinator", re.MULTILINE),
    ):
        assert not pat.search(src), f"confidence.py matches forbidden import {pat.pattern!r}"
