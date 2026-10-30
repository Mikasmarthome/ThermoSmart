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
second, in-file constructor call. The remaining "≈36" were triaged test by test
in the next section; they are not S7 regressions.

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
* **Unclear:** nothing in the S3-passing set (see the triage below). Still open
  are the 127 files that were already not green at S3, and the four files that
  crash the Windows interpreter (see the triage below).

## Verified regression delta triage (S3 → S7)

Method: every private test file was run one file at a time, on three states, and
the results were compared per test ID. Scratch copies only; the private harness
and production code were not modified.

1. **S3**: production code of `origin/develop` `20f5abb` (LE v1 present).
2. **S7 raw**: production code of this branch, harness unchanged.
3. **S7 + shim**: this branch plus a scratch `conftest.py` shim that (a) accepts
   the legacy fourth constructor argument and stores it as
   `coord.learning_engine`, and (b) routes `_resolve_base_target` to a value the
   old tests inject through `learning_engine.async_get_base_target`.

| State | test IDs | passed | failed | error |
|---|---|---|---|---|
| S3 | 4255 | 3517 | 514 | 224 |
| S7 raw | 4122 | 2093 | 1787 | 242 |
| S7 + shim | 4061 | 3188 | 635 | 238 |

### Why the totals differ between S3 and S7

* **Collection drift (202 tests).** Eight files import the removed
  `learning_engine` module at module level. At S7 they produce one collection
  error each instead of their tests, so 202 passing S3 tests simply disappear
  from the S7 count. This is not a pass/fail change.
* **Signature failures (≈1150 tests).** Tests that passed at S3 fail at S7 raw
  only because the coordinator constructor changed. The shim removes this:
  3188 of the 3517 S3 passes pass again.
* **Large simulation files.** Four files (`test_s1a_item20_long_run`,
  `test_s1a_item6_long_run`, `test_s1a_item8_ab_long_run`, and the tail of
  `test_simulation_s1a_part2`) crash the Windows interpreter (access violation
  in Home Assistant's orjson encoder, Python 3.14), identically at S3 and S7, so
  they yield no per-test IDs in either state. They are not evidence either way.
  The other eight simulation files (`test_s1a_item6_*`, `test_s1a_item8_ab_30d`,
  `test_simulation_s1`) run weeks of simulated control through the real
  coordinator and give identical results at S3 and S7 + shim (for example
  14/14, 12/12, 16/16 and 8/8 passed).
* **Unchanged non-green.** 514 failed and 224 error results are identical at S3
  and S7 + shim (old API drift, already present before S7).
* The earlier "≈36" is not authoritative. There are 329 tests that pass at S3
  and not at S7 + shim; they split as follows.

### Classification of the 329 S3-passing tests that do not pass at S7 + shim

| Group | tests | Reason | Class |
|---|---|---|---|
| `test_learning_outcome_score`, `_heat_rate`, `_storage`, `_weights`, `_forecast`, `_migration`, `test_le2_authority_freeze`, `test_clock_naive_timestamps` | 202 | module imports the removed `learning_engine` (collection error) | obsolete expectation; LE v1 internals |
| `test_learning_observe`, `_trv_window`, `_confidence`, `_schedule`, `_sessions` | 116 | construct or call `LearningEngine` | obsolete expectation; LE v1 internals |
| `test_four_mode_matrix_extended::TestLE1FrozenSemantics` | 6 | assert LE v1 freeze flags and no-op methods | obsolete expectation |
| `test_le2_heatloss_storage::TestLE1EmaNotWritten`, `test_le2_heat_loss_authority::test_le1_ema_mutation_silenced_in_source`, `test_le2_boost_lifecycle::test_no_le_v1_boost_factor_affects_control`, `test_le2_early_cutoff::test_learning_engine_has_no_early_cutoff_method` | 5 | assert that LE v1 does not influence control or has no method | obsolete expectation (true by construction now; `test_early_cutoff_residue_gone` is public) |
| **Total** | **329** | | no behavioural assertion on a live code path |

The 19 tests that looked like behaviour failures after the constructor shim
(`test_le2_forecast_bias_trust` 16, `test_coordinator_recommendation` 1,
`test_le2_preheat_authority` 1, `test_le2_early_cutoff` 1) pass once the
base-target seam is honoured. They inject the base target through the removed
LE v1 object; the value was never produced by learning. Class: test seam.
The remaining failures of `LiveDecisionRecord(early_cutoff_state=...)`,
`PredictionType.RECOMMENDED_EARLY_CUTOFF`, `_ec_hold_active`, `episode_failed`
and `DecisionMode` also fail at S3 (failed → failed): pre-existing.
One shim artefact: `test_simulation_s1a_part2::TestClockGaps::test_coordinator_has_clock_attribute`
inspects the source of `__init__`, which the shim replaces. Harness problem.
No test went from failed to passed or from passed to a real behavioural failure.

### Production behaviour check

Read-only comparison of S3 and S7 production code:

* At S3 LE v1 was frozen. Its only reads on a control path were
  `is_zone_enabled` (Learning switch) and `async_get_base_target` (a pure
  schedule lookup). Preheat, TPI coefficients, boost factor, heat loss and early
  cutoff never read LE v1 (comments and code in `coordinator.py` at `20f5abb`).
* `async_get_base_target` was replaced by `schedule.resolve_base_target`. A
  differential run of the old `_schedule_target` against the new
  `schedule_slot_target` over a full week in 5-minute steps and five schedule
  configurations (including night end after midnight and inverted times) gave
  **0 differences in 10 080 samples**.
* The Learning switch moved to the coordinator; covered by
  `test_learning_switch_restore_ordering`, `test_learning_toggle_characterization`
  and `test_learning_control_golden` (public).
* The only intended behaviour changes are the ones listed in the release notes
  (period labels over midnight, `always_comfort` with a disabled schedule).

| Area | S7 effect |
|---|---|
| Heating target calculation | none (0/10 080 differences) |
| Override expiration | intended change with disabled schedule / over midnight; public tests |
| Learning enabled/disabled authority | moved, not changed; public tests |
| Active Control | none |
| Preheat, TPI, Boost, Heat Loss | none (LE v1 was never read); public golden tests |
| Persistence / Restore | LE v1 store no longer loaded, purge kept; active models untouched |
| TRV safety and fallback | none |

Conclusion: no S7 production regression confirmed. Limits: the four crashing
simulation files could not be compared, and failures that are identical at S3
and S7 were not compared by their messages.

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
  then work through the contracts table.
* Never weaken an assertion to make a ported test pass. If the new behaviour is
  intended, document it and update the expected value deliberately.
* Retire only tests that check LE v1 internals; record each retirement here.
* Private tests stay private and must not appear in a public diff.
