"""Phase 3 purity / scope tests for the LE 2.0 storage shell.

* The pure storage submodules import without Home Assistant.
* ``stores`` is allowed to import Home Assistant, but no storage module may
  import the coordinator or any concrete model.
* Importing the storage package creates no store.
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
STORAGE_DIR = LEARNING_DIR / "storage"

# Pure submodules that must import without Home Assistant.
PURE_STORAGE_MODULES = ("naming", "serialization", "segments", "recovery", "persistence")

_SCRIPT = r'''
import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path

learning_dir = Path(sys.argv[1])
storage_dir = learning_dir / "storage"


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "homeassistant" or name.startswith("homeassistant."):
            raise ImportError("homeassistant import is forbidden in pure modules")
        return None


sys.meta_path.insert(0, _Block())

# Build synthetic packages so relative imports resolve without the HA parent.
root = types.ModuleType("ts_p3")
root.__path__ = [str(learning_dir)]
sys.modules["ts_p3"] = root
storage_pkg = types.ModuleType("ts_p3.storage")
storage_pkg.__path__ = [str(storage_dir)]
sys.modules["ts_p3.storage"] = storage_pkg

for name in ("clock", "contracts", "raw_schemas", "episode_schemas"):
    spec = importlib.util.spec_from_file_location(f"ts_p3.{name}", learning_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ts_p3.{name}"] = mod
    spec.loader.exec_module(mod)

for name in ("naming", "serialization", "segments", "recovery", "persistence"):
    spec = importlib.util.spec_from_file_location(
        f"ts_p3.storage.{name}", storage_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ts_p3.storage.{name}"] = mod
    spec.loader.exec_module(mod)

assert "homeassistant" not in sys.modules
print("PURE_OK")
'''


def test_pure_storage_modules_import_without_home_assistant():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(LEARNING_DIR)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "PURE_OK" in result.stdout


def test_pure_modules_have_no_ha_import():
    pat = re.compile(r"^\s*(import\s+homeassistant|from\s+homeassistant)", re.MULTILINE)
    for name in PURE_STORAGE_MODULES:
        src = (STORAGE_DIR / f"{name}.py").read_text(encoding="utf-8")
        assert not pat.search(src), f"storage/{name}.py imports homeassistant"


def test_no_storage_module_imports_coordinator_or_models():
    # Match real import statements only; docstrings may mention these words.
    patterns = [
        re.compile(r"^\s*from\s+\.{1,2}coordinator", re.MULTILINE),
        re.compile(r"^\s*import\s+.*coordinator", re.MULTILINE),
        re.compile(r"^\s*from\s+\.{1,2}models", re.MULTILINE),
        re.compile(r"^\s*import\s+.*\bmodels\b", re.MULTILINE),
    ]
    for path in STORAGE_DIR.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        for pat in patterns:
            assert not pat.search(src), f"{path.name} imports {pat.pattern!r}"


def test_stores_module_may_import_home_assistant():
    src = (STORAGE_DIR / "stores.py").read_text(encoding="utf-8")
    assert "from homeassistant.helpers.storage import Store" in src
