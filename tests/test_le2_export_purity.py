"""Phase 16 purity/scope: orchestration imports no HA/storage/coordinator, mutates nothing."""
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

root = types.ModuleType("ts_p16")
root.__path__ = [str(learning_dir)]
sys.modules["ts_p16"] = root

for name in ("clock", "contracts", "raw_schemas", "episode_schemas", "features", "registry",
             "confidence", "privacy", "export", "diagnostics"):
    spec = importlib.util.spec_from_file_location(f"ts_p16.{name}", learning_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"ts_p16.{name}"] = mod
    spec.loader.exec_module(mod)

assert "homeassistant" not in sys.modules
print("PURE_OK")
'''


def test_orchestration_imports_without_home_assistant():
    result = subprocess.run([sys.executable, "-c", _SCRIPT, str(LEARNING_DIR)],
                            capture_output=True, text=True)
    assert result.returncode == 0, (
        f"purity subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert "PURE_OK" in result.stdout


def test_orchestration_source_no_forbidden_imports():
    for fname in ("privacy.py", "export.py", "diagnostics.py"):
        src = (LEARNING_DIR / fname).read_text(encoding="utf-8")
        for pat in (
            re.compile(r"^\s*(import\s+homeassistant|from\s+homeassistant)", re.MULTILINE),
            re.compile(r"^\s*from\s+\.storage", re.MULTILINE),
            re.compile(r"^\s*from\s+\.coordinator", re.MULTILINE),
            re.compile(r"^\s*from\s+\.models", re.MULTILINE),
            re.compile(r"^\s*from\s+\.builders", re.MULTILINE),
            re.compile(r"open\(", re.MULTILINE),  # no file I/O
        ):
            assert not pat.search(src), f"{fname} matches forbidden pattern {pat.pattern!r}"


def test_export_does_not_mutate_models():
    from custom_components.thermosmart.learning.models import HeatRateModel
    from custom_components.thermosmart.learning.export import build_support_export
    from tests.helpers_orchestration import export_input, zone_input
    hr = HeatRateModel("living_room")
    before = hr.serialize_state()
    z = zone_input(models={"heat_rate": hr})
    build_support_export(export_input(zones=(z,)))
    assert hr.serialize_state() == before
