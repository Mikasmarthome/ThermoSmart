# Private test coverage debt after the LE v1 removal

Status: open tracking item. Measured on `feature/remove-le-v1` (HEAD `69e2337`,
base `origin/develop` `20f5abb`). Nothing in the private harness was deleted,
edited or marked as passed; all numbers below come from running it as it is.

## Why the private harness no longer runs completely

The private harness (288 test files, kept outside this repository) was written
against the pre-S7 coordinator:

* `ThermoSmartCoordinator(hass, entry, weather_engine, learning_engine, *, clock)`
  — the fourth positional argument no longer exists. The coordinator is now
  `ThermoSmartCoordinator(hass, entry, weather_engine, *, clock)`; learning
  lives in the `learning/` runtime.
* `custom_components/thermosmart/learning_engine.py` (LE v1) is gone, so
  `import ... learning_engine` fails at collection.
* A few runtime dataclasses and prediction types changed in earlier phases
  (`BoostUpdateContext`, `RuntimeCycleInput`, `LiveDecisionRecord`,
  `PredictionType.RECOMMENDED_EARLY_CUTOFF`, `FORECAST_BIAS_MIN`).
* Tests using the `hass` fixture only run under the Docker setup, not in the
  plain per-file harness.

## Real scope (not 87 independent tests)

The earlier figure of "87 unclear results" cannot be reproduced and must not be
read as 87 test cases. It mixes files, collection errors and individual tests.
The measured picture, per file with `-p no:homeassistant`:

| | files | tests (passed / failed / error) |
|---|---|---|
| Private run at the S3 state | 293 listed | 3457 / 514 / 221 |
| Private run at the S7 state | 288 | 2102 / 1787 / 242 |
| Result identical S3 → S7 | 214 | — |
| Result changed S3 → S7 | 73 | — |

The 214 identical files include 127 that were already not green at S3. They
predate S7 (older API drift: unknown constructor keywords on
`BoostUpdateContext` / `RuntimeCycleInput` (≈34 files), missing modules or names
(≈71), missing `hass` fixture (14), others (≈7)) and are not S7 regressions.

### The 73 files that changed because of S7

| Cause | files | tests (passed / failed / error) |
|---|---|---|
| A. Outdated coordinator signature (`learning_engine` positional) | 54 | 323 / 1068 / 4 |
| B. Imports the removed `learning_engine` module | 14 | 32 / 172 / 13 |
| C. Other API drift, one file each (see below) | 5 | 85 / 172 / 5 |

Category C: `test_le2_boost_lifecycle` (expects `RELEASED_MODE_CHANGE`, gets
`COMPLETED`), `test_le2_early_cutoff` (`RECOMMENDED_EARLY_CUTOFF` is not
produced), `test_learning_forecast` (`FORECAST_BIAS_MIN` removed),
`test_s1a_item9_corruption_recovery` (`BoostUpdateContext` keyword),
`test_le2_ha_real_zone_removal` (needs the `hass` fixture).

### Verification of category A without touching the private tests

In a scratch copy only (not in the harness), the two helper constructors
(`tests/helpers.py`, `tests/helpers_ha_runtime.py`) were adapted to the new
signature, assigning the old mock as `coord.learning_engine`. Result for the 54
files: **1190 passed, 205 failed** (was 323 / 1068). So about 85 % of category A
is a mechanical fixture problem. Of the 205 remaining failures, about 169 are a
second, in-file constructor call and about 36 are real behaviour or API
differences that need triage (`early_cutoff_state` on `LiveDecisionRecord`,
`_ec_hold_active`, `episode_failed` expectations, forecast-bias trust values).

## Classification by intent

* **Probably obsolete LE v1 implementation tests (retire, do not port).** This
  list is based on file names and the failure cause; the content of each file
  still has to be reviewed before anything is retired:
  `test_learning_confidence`, `test_learning_heat_rate`, `test_learning_migration`,
  `test_learning_observe`, `test_learning_outcome_score`, `test_learning_schedule`,
  `test_learning_sessions`, `test_learning_storage`, `test_learning_trv_window`,
  `test_learning_weights`, `test_le2_heatloss_storage`, `test_le2_authority_freeze`
  (they exercise LE v1 internals that no longer exist). Two other category-B
  files, `test_clock_naive_timestamps` and `test_four_mode_matrix_extended`,
  only fail on the import and may hold live contracts: port, do not retire.
* **Outdated signature or fixture only (port the helper, keep the test):** the 54
  category-A files, notably `test_coordinator_*`, `test_four_mode_matrix`,
  `test_le2_tpi_*`, `test_le2_preheat_*`, `test_le2_heat_loss_authority`,
  `test_le2_adaptive_boost_*`, `test_le2_ha_*`, `test_bootstrap_override_security`,
  `test_clock_authority`, `test_window_delay_event_driven`,
  `test_temperature_units`, `test_s1a_item12..20`, `test_simulation_s1*`.
* **Still cover important behaviour contracts:** see the next section.
* **Unclear (needs triage before a decision):** the ≈36 residual failures above,
  `test_le2_boost_lifecycle`, `test_le2_early_cutoff`, `test_learning_forecast`,
  and the 127 pre-existing non-green files.

## Contracts that must be restored before the next stable release

Each row lists the private file(s) that hold the contract and the public test(s)
that already cover part of it.

| Contract | Private files | Public coverage today | Needed |
|---|---|---|---|
| Heat Loss (learned loss feeds TPI) | `test_le2_heat_loss_authority`, `test_le2_heatloss_storage` | `test_learned_state_golden`, `test_learning_control_golden`, `test_le2_storage_serialization` | restore authority/fallback cases |
| Preheat | `test_le2_preheat_authority`, `test_le2_preheat_golden`, `test_le2_onset_*` | `test_learned_state_golden` (4 cases), `test_onset_delay_scope_contract` | restore baseline-vs-learned edge cases |
| Boost | `test_le2_adaptive_boost_*`, `test_le2_boost_*`, `test_le2_b1b_dispatch_truth` | `test_model_construction_contract` only | restore lifecycle, activation, zero-emission |
| Persistence / Restore | `test_le2_ha_restart_survival`, `test_le2_ha_store_integration`, `test_s1a_item6_storage` | `test_learning_switch_restore_ordering`, `test_le2_storage_serialization`, `test_storage_*`, `test_unload_session_state` | restart survival of learned models |
| Control Authority | `test_le2_control_*`, `test_le2_live_*`, `test_le2_tpi_blend/integration/smoothing` | `test_learning_control_golden`, `test_learning_toggle_characterization`, `test_observation_mode_hardware_writes`, `test_tpi` | authority freeze, no-direct-dispatch |
| Four-Mode matrix | `test_four_mode_matrix` (73 tests), `test_four_mode_matrix_extended` | `test_four_mode_matrix`, `test_four_mode_matrix_extended` (public subset) | confirm the private 73 are covered by the public file |
| Sensor / TRV failure fallbacks | `test_s1a_item13_control_logic`, `test_s1a_item14_device_trv_capability`, `test_coordinator_integration` | `test_s1a_item14_device_trv_capability`, `test_trv_echo_and_command_ledger`, `test_device_compat_p0p1p2` | restore sensor-unavailable and TRV-failure paths |

## Rules for working this down

* Port helpers first (one change in each helper revives most of category A),
  then triage the residual ≈36.
* Never weaken an assertion to make a ported test pass. If the new behaviour is
  intended, document it and update the expected value deliberately.
* Retire only tests that check LE v1 internals; record each retirement here.
* Private tests stay private and must not appear in a public diff.
