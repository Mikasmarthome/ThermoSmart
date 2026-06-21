"""Proves the LE 2.0 registry module has no Home Assistant / storage / coordinator
dependency, and imports cleanly as pure Python.
"""
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

FORBIDDEN = ("homeassistant",)


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        for bad in FORBIDDEN:
            if name == bad or name.startswith(bad + "."):
                raise ImportError(f"{name} import is forbidden in pure LE2 modules")
        return None


sys.meta_path.insert(0, _Block())

pkg = types.ModuleType("ts_phase2")
pkg.__path__ = [str(learning_dir)]
sys.modules["ts_phase2"] = pkg

for name in ("clock", "contracts", "raw_schemas", "episode_schemas", "registry"):
    spec = importlib.util.spec_from_file_location(
        f"ts_phase2.{name}", learning_dir / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ts_phase2.{name}"] = mod
    spec.loader.exec_module(mod)

assert "homeassistant" not in sys.modules
print("PURE_OK")
'''


def test_registry_imports_without_home_assistant():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(LEARNING_DIR)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "PURE_OK" in result.stdout


def test_registry_source_has_no_forbidden_imports():
    source = (LEARNING_DIR / "registry.py").read_text(encoding="utf-8")
    for forbidden in ("homeassistant", "storage", "coordinator"):
        pattern = re.compile(rf"^\s*(import\s+{forbidden}|from\s+{forbidden}|from\s+\.{forbidden})",
                             re.MULTILINE)
        assert not pattern.search(source), f"registry.py imports {forbidden}"
    # no concrete model module is imported in Phase 2
    assert "from .models" not in source
    assert "import models" not in source
