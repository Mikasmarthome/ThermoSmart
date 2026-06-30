"""Phase 7 scope tests: init_state is pure; reset is HA-shell but coordinator-free."""
from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

LEARNING_DIR = (
    Path(__file__).resolve().parents[1]
    / "custom_components" / "thermosmart" / "learning"
)

# init_state must load purely; reset is allowed to import the HA store shell.
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

pkg = types.ModuleType("ts_p7")
pkg.__path__ = [str(learning_dir)]
sys.modules["ts_p7"] = pkg

for name in ("clock", "contracts", "raw_schemas", "episode_schemas", "init_state"):
    spec = importlib.util.spec_from_file_location(f"ts_p7.{name}", learning_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ts_p7.{name}"] = mod
    spec.loader.exec_module(mod)

assert "homeassistant" not in sys.modules
print("PURE_OK")
'''


def test_init_state_imports_without_home_assistant():
    result = subprocess.run([sys.executable, "-c", _SCRIPT, str(LEARNING_DIR)],
                            capture_output=True, text=True)
    assert result.returncode == 0, (
        f"purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert "PURE_OK" in result.stdout


def test_reset_source_scope():
    src = (LEARNING_DIR / "reset.py").read_text(encoding="utf-8")
    # reset is allowed to import the HA store shell ...
    assert "from .storage.stores import" in src
    # ... but must not import the coordinator or concrete models, or alter setup
    for forbidden in (
        re.compile(r"^\s*from\s+\.\.coordinator", re.MULTILINE),
        re.compile(r"^\s*from\s+\.coordinator", re.MULTILINE),
        re.compile(r"^\s*import\s+.*coordinator", re.MULTILINE),
        re.compile(r"^\s*from\s+\.models", re.MULTILINE),
        re.compile(r"^\s*from\s+\.\.config_flow", re.MULTILINE),
    ):
        assert not forbidden.search(src), f"reset.py matches forbidden import {forbidden.pattern!r}"


def test_importing_reset_creates_no_store():
    # Importing the module must perform no I/O / create no store.
    mod = importlib.import_module("custom_components.thermosmart.learning.reset")
    assert hasattr(mod, "ensure_v2_initialized")


def test_setup_init_file_unchanged_by_phase7():
    # Phase 7 must not wire anything into the normal setup path.
    init_src = (LEARNING_DIR.parent / "__init__.py").read_text(encoding="utf-8")
    assert "ensure_v2_initialized" not in init_src
    assert "learning.reset" not in init_src
