# Legacy learning engine (LE v1) removal — release and migration notes

Applies to the next release after 1.3.2. No version bump is part of this change.

## What changed

* **LE v1 removed.** The legacy `learning_engine.py` and its coordinator wiring
  are gone. The `ThermoSmartCoordinator` constructor no longer takes a learning
  engine. The Learning on/off switch is owned by the coordinator.
* **Active learning runtime unchanged.** Heat rate, heat loss, afterheat, onset
  delay, outcome, forecast and boost models (the `learning/` package) behave as
  before. Control output for learned and unlearned zones is pinned by the golden
  tests in `tests/test_learned_state_golden.py` and
  `tests/test_learning_control_golden.py`.
* **Schedule periods over midnight corrected.** Evening periods that continue
  past midnight now resolve to the same period on both sides of the day change.
  The schedule logic lives in the pure module `schedule.py`.
* **Override with a disabled schedule corrected.** With the schedule switched
  off the period is the constant `always_comfort`, so a manual override no
  longer expires at a day boundary; with the schedule on it still ends at the
  real slot change.
* **Diagnostics export.** The historical keys `analytics` and
  `historical_learning_snapshot` are no longer exported. Nothing in the
  integration reads them.

## Migration

* No migration of the active learning model state is required; those stores are
  untouched.
* The old LE v1 store (`thermosmart_learning_data`) is no longer read. It is
  purged when the last zone is removed, as before.
* The Learning switch state is restored from Home Assistant restore data before
  the first refresh, as before.

## Not part of this change

No version bump, tag or release. The global 28 °C TPI setpoint cap is
unchanged and tracked for a later phase. Outstanding private-test work is
tracked in [private-test-coverage-debt.md](private-test-coverage-debt.md).
