"""Learning storage shell (Phase 3 — capture foundation).

Pure submodules (``naming``, ``serialization``, ``segments``, ``recovery``,
``persistence``) have no Home Assistant dependency and are importable and
testable in isolation. Only ``stores`` wraps the Home Assistant ``Store`` and
is therefore HA-dependent.

Importing this package performs no I/O and creates no store. Nothing here is
wired into the running integration in Phase 3.
"""
