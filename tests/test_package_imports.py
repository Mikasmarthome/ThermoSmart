"""Every module of the integration imports on its own.

Guards cleanups of unused code against leaving a dangling import behind in a
module that no other test happens to load.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import custom_components.thermosmart as integration

_MODULES = sorted(
    m.name for m in pkgutil.walk_packages(integration.__path__, integration.__name__ + ".")
)


def test_modules_were_discovered():
    assert len(_MODULES) > 50


@pytest.mark.parametrize("name", _MODULES)
def test_module_imports(name):
    importlib.import_module(name)
