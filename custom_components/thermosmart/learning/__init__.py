"""ThermoSmart Learning Engine 2.0 — pure-Python foundation package.

Phase 1 contains only runtime-neutral, typed building blocks:
clock abstraction, model/prediction contracts, raw record schemas and
episode schemas. None of these modules import Home Assistant, hold runtime
state, touch storage, or influence control. They are safe to import and unit
test in isolation.

Deliberately empty of imports so the submodules can be loaded standalone
without pulling in the parent integration (which depends on Home Assistant).
"""
