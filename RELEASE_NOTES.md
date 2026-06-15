# ThermoSmart – Release Notes

## v1.1.0-rc.3 – Release Candidate 3

Adds two fixes and two improvements to RC.2: a missing translation key in the Options Flow, a new global Debug Log switch on the ThermoSmart System entry, meaningful debug log output across all zones, and corrected override handling for mode and season transitions.

---

### Fix: Options Flow abort translation (`system_no_options`)

The `ThermoSmartSystemOptionsFlow` calls `async_abort(reason="system_no_options")` when a user opens the options dialog for the System entry (which has no editable options). Home Assistant looks up abort reasons for Options Flow under `options.abort.<reason>` — a separate namespace from `config.abort`. The key existed only in the `config` namespace and was missing from `options`, causing a raw key string to be displayed instead of the user-facing message.

**Fix:** Added `options.abort.system_no_options` to `strings.json` and all 24 locale translation files.

---

### New: Global Debug Log switch

A new **Debug Log** switch appears on the ThermoSmart System device card. Toggling it sets the `custom_components.thermosmart` logger to `DEBUG` at runtime — no Home Assistant restart or `configuration.yaml` edit required.

**Behaviour:**
- **On:** stores the logger's current level, sets it to `DEBUG`, logs `ThermoSmart System: Debug logging enabled` at INFO (always visible)
- **Off:** restores the previously stored level (`NOTSET` / inherited, or YAML-configured value), logs `ThermoSmart System: Debug logging disabled` at INFO
- **HA restart with switch ON:** state is restored via `RestoreEntity`; debug is re-enabled immediately during entity setup
- **YAML logger configuration is preserved:** if `custom_components.thermosmart: debug` is set in `configuration.yaml`, turning the switch off restores that level rather than resetting it

Only the `custom_components.thermosmart` parent logger is modified. All submodule loggers inherit from it. No coordinator, learning engine, or TRV control logic is affected.

---

### Improvement: Debug Logging

Meaningful diagnostic output has been added throughout the coordinator and learning engine so that enabling the Debug Log switch produces actionable log entries during normal operation.

- **Cycle summary:** one compact `DEBUG` line per zone per update cycle with mode, current temp, target, adjusted target, TRV setpoint, TPI duty cycle, preheat minutes, window state, and summer flag
- **Mode change:** logged whenever the effective heating mode transitions (e.g. `auto → vacation`), including the reason (`vacation`, `manual`, `presence`, or `schedule`)
- **TRV observation:** logged when the learning engine stores a new observation, including setpoint, indoor temp, target, and heat rate
- **Forecast/weather correction:** logged when the weather bias actively suppresses the heating target, showing raw vs. effective target and correction magnitude
- Debug logs now make it easier to understand why ThermoSmart selected a particular target, TRV setpoint, or operating mode

---

### Fix: Override Handling

Manual temperature overrides are now consistently cleared whenever the active mode changes, ensuring the newly selected mode's configured temperature always takes effect.

- Manual overrides are cleared when switching to any different heating mode (e.g. Comfort → Eco, Night → Comfort)
- Manual overrides are cleared when Vacation mode activates — prevents a stale override from bypassing frost protection
- Manual overrides are cleared when Summer Mode becomes active, both via the global switch and via automatic 72 h season detection — prevents stale overrides from reappearing after Summer Mode ends
- Schedule-slot transitions (Auto mode night ↔ comfort), Presence-away detection, and window-open suppression continue to work as before

---

### Included from RC.2

- Device-aware setpoint resolution: `target_temp_step` snap with round-half-up before write and store
- `is_active=False` guard: devices like tado, daikin_climate, netatmo skip `climate.set_temperature`
- Final clamp after step-snap: re-clamp to `min_temp` / `profile.minimum_setpoint`
- Summer mode frost setpoint tracking: `_last_written_setpoints` updated before writing 12 °C
- Debounced TRV observation save: 60-second deferred save replaces threshold-based `% 20` save

### Included from RC.1

- Window open/close delay: exact timing via `async_call_later`
- TRV setpoint sensor: immediate update on window open
- Race condition fix: `_handle_trv_change` suppressed during window open
- Preheat cap raised from 60 → 180 minutes
- Learning sensor UX: `None` on MEASUREMENT sensors + `data_status: learning` attribute

---

**Pre-release** – not marked as Latest. Current stable release remains v1.0.9.

**Test focus for RC.3 observation period:**
- Open ThermoSmart System options → verify message instead of raw key `system_no_options`
- Toggle Debug Log switch → verify `ThermoSmart System: Debug logging enabled/disabled` in log
- Restart HA with Debug Log ON → verify debug resumes immediately
- With Debug Log ON: verify per-zone cycle summary appears every update cycle in the log
- With Debug Log ON and a mode change: verify mode transition log line with reason
- Set a manual override, then switch mode → verify override is cleared and new mode temp applies
- Set a manual override, then activate Summer Mode → verify override is cleared
- All RC.2 test cases remain valid

---

## v1.0.8

### Learning Engine Data Collection

This release completes the data collection phase for Learning Engine 2.0. All
observations now carry a rich context layer that enables LE 2.0 to filter,
weight, and analyse historical data without any ambiguity.

**Observation context fields added to every learning observation:**

- **`active_control`** — distinguishes pure observation mode (`false`) from active
  TRV control (`true`). LE 2.0 can train exclusively on observations where
  ThermoSmart was actually steering the heating.
- **`window_open`** — flags observations recorded while a window was open.
  Contaminated heat-rate readings during window events can be excluded from
  thermal modelling.
- **`control_reason`** — the primary driver of the current heating target:
  `schedule`, `presence`, `manual_override`, `window_open`, `vacation`,
  `summer_mode`. Single source of truth for why a particular target was set.
- **`preheat_active`** — marks observations taken during a pre-heat ramp.
  Pre-heat observations reflect catch-up heating, not steady-state performance.
- **`heating_failure`** — flags observations where a heating failure was active.
- **`vacation`** — boolean; true when vacation/frost-protection mode was active.
  Enables LE 2.0 to optionally exclude long-absence periods from schedule learning.
- **`summer_mode`** — boolean; true when summer mode was effectively active.
- **`schedule_period`** — the active schedule slot at observation time: `comfort`,
  `night`, `eco`, `away`, or `null` during vacation/summer. Enables per-slot
  heat-rate analysis without post-hoc reconstruction from timestamps.

### Data Quality Fixes

- **TRV observation burst guard** — duplicate TRV setpoint observations within a
  5-second window are rejected. Prevents inflated TRV observation counts caused
  by rapid coordinator cycles.
- **Window cooling false-positive threshold** — window cooling events below 0.5 °C
  temperature drop are no longer recorded, eliminating sensor-noise artefacts.

### Export Analytics

The learning data export now includes additional computed metrics per zone:

- `contaminated_heat_rate_count` — number of heat-rate observations where
  `delta < −1.0 °C` (room clearly below target; reading reflects catch-up, not
  steady-state performance).
- `clean_heat_rate_mean` — mean heat rate excluding contaminated readings.
- `clean_norm_heat_rate_mean` — mean normalised heat rate excluding contaminated
  readings.

No migration required. Older observations without context fields remain valid;
absent fields are treated as unknown by LE 2.0. `export_format_version` stays 1.

### Upgrade

HACS → Integrations → ThermoSmart → Update → Restart Home Assistant.
No configuration changes required. Learning data is preserved.

---

## v1.0.7

### New Features

**TRV Temperature Source Management**
Opt-in feature for TRVs that expose a `temperature_sensor` select entity (e.g. SONOFF TRVZB). When enabled, ThermoSmart automatically switches the TRV between `internal` and `external` temperature source based on sensor availability. Falls back to `internal` with a 5-minute grace period if external sensors become unavailable. Ownership tracking ensures manual user changes are never overwritten.

**Per-Zone Analytics in Learning Data Export**
The learning data export now includes a computed `analytics` block per zone: observation span, target-change frequency, delta statistics, heat rate mean, and setpoint excess — all computed at export time with no storage changes and no migration required.

### Improvements

- **UX: Toggle descriptions** — All three device toggles (Valve Maintenance, External Room Temperature, Calibration Invert) now show descriptive helper text in all 24 supported languages.
- **UX: Toggle order** — Device options are now consistently ordered: Valve Maintenance → External Room Temperature → Calibration Invert.
- **UX: Clearer label** — The "External Room Temperature" option label was improved for clarity in all 24 languages.

### Bug Fixes

- **Calibration skip when `temperature_sensor = external`** — When a TRV is confirmed to be using its external temperature source, ThermoSmart no longer writes a calibration offset that the TRV firmware would ignore anyway.
- **Allow `comfort_temp == night_temp`** — The config flow no longer rejects equal comfort and night temperatures, enabling single-setpoint setups.

### Upgrade

HACS → Integrations → ThermoSmart → Update → Restart Home Assistant.
No configuration changes required. Learning data is preserved.

---

## v1.0.6

### New Feature — Export Learning Data

ThermoSmart can now export an anonymized snapshot of its learning data as a local JSON file. This is intended for voluntary debugging and to contribute real-world data to future Learning Engine improvements.

**Two ways to trigger the export:**
- **Button:** Press *Export Learning Data* in the ThermoSmart System device card in your HA dashboard.
- **Service:** Call `thermosmart.export_learning_data` from Developer Tools → Services.

The file is saved to `/config/www/` with a timestamped, randomized filename. It can be opened via `/local/<filename>` appended to your HA base URL.

**What is exported:**
- ThermoSmart version, export timestamp, zone count
- Per-zone: TRV count, sensor counts, feature flags (booleans)
- Per-zone: all numeric learning data (observations, rates, confidence, boost factor, …)
- Observation timestamps (needed for longitudinal analysis)

**What is NOT exported:**
- Passwords or authentication tokens
- Entity IDs, device names, or integration names
- Person names or user identifiers
- Street addresses or geographic coordinates

No data is sent anywhere automatically. You review the file locally and decide whether and with whom to share it.

### Bug Fixes

**Export notification link** — Replaced a non-functional Markdown link in the persistent notification with a plain path. Home Assistant's SPA router was intercepting relative `<a href>` clicks and closing the notification drawer instead of navigating to the file.

**Export notification text** — Clarified that the `/local/<filename>` path must be appended to the HA base URL, that the file opens as JSON in the browser, and how to save it (Save Page As… / Ctrl+S / Cmd+S).

---

## v1.0.5

### Bug Fix

**valve_opening_degree recovery now works after a full Home Assistant restart**

v1.0.4 introduced an automatic recovery that resets `valve_opening_degree` to 100%
on SONOFF TRVZB (and similar devices) if a previous ThermoSmart version had written
the TPI duty-cycle to it. That recovery ran during integration setup — before
Zigbee2MQTT entities were available on a full HA restart — so it was silently
skipped.

v1.0.5 retries the recovery across the first three coordinator refresh cycles
(≈ 5 s apart). The `_valve_opening_degree` reset now fires as soon as the relevant
entity becomes reachable, regardless of how fast Zigbee2MQTT comes online after a
restart. If no relevant entity is found within three attempts, a warning is logged.

Integration Reload behaviour is unchanged: the reset fires immediately as before.

### Upgrade

HACS → Integrations → ThermoSmart → Update → Restart Home Assistant.
No configuration changes required. Learning data is preserved.

---

## v1.0.4

### Bug Fix

**SONOFF TRVZB and similar devices: heating capacity was unintentionally limited**

ThermoSmart previously auto-detected `valve_opening_degree` entities on TRV devices
and wrote the TPI duty-cycle directly to them (0–100%). On SONOFF TRVZB (via
Zigbee2MQTT), this entity is a **maximum-opening limit config**, not a live valve
position command. The result:

- At 60% TPI duty: valve could open no more than 60% → reduced heating power
- At 0% TPI duty (idle): valve was blocked entirely → no heating possible until
  the next 5-minute update cycle

**Fix:** `valve_opening_degree` is no longer written by ThermoSmart.
Control continues via `climate.set_temperature` (setpoint boost), which is the
correct path for these devices. The existing `external_temperature_input` feed
is unaffected and continues to improve TRV accuracy.

**Automatic recovery on upgrade:** If ThermoSmart finds a `valve_opening_degree`
entity with a value below 100% on a managed TRV device, it resets it to 100%
at startup and logs the action. No manual intervention required.

### Unaffected devices
Eurotronic Spirit (`valve_position`), Tuya TS0601 (`pi_heating_demand`),
Homematic IP (`level`) — all other direct-valve patterns are unchanged.

### Upgrade
HACS → Integrations → ThermoSmart → Update → Restart Home Assistant.
Learning data is preserved.

---

## v1.0.3

### Bug Fixes

- **Current temperature displayed immediately**: `climate.current_temperature` and the dashboard card now reflect the actual sensor reading instantly on every state change. Previously the display was updated from the EMA-smoothed control value, causing a visible lag of several update cycles. Display path (raw average) and control path (EMA + spike filter) are now fully separated.

- **Current temperature available for TRV-only zones**: Zones without external temperature sensors now have a valid `current_temperature` — ThermoSmart reads the `current_temperature` attribute from TRV entities as fallback. This re-enables TPI, learning, slope calculation, and calibration for sensorless zones. External sensors always take priority.

### Documentation

- **FAQ: SONOFF TRVZB `heat` vs `auto` mode**: Added a FAQ entry explaining why Active Control users do not need to delete the TRV's internal weekly schedule — Active Control automatically switches the TRV to `heat` mode on every cycle, bypassing the internal schedule.

### Notes

No migration required. No configuration changes needed.

---

## v1.0.2

### Fixes

- Fixed validation of empty optional weather and outdoor sensor entity fields in the zone creation flow.
- Empty weather-related entity selectors now correctly accept both `None` and empty values without raising entity ID validation errors.

### Notes

This fixes an issue where creating a new zone without weather or outdoor sensor entities could fail with:

`Entity None is neither a valid entity ID nor a valid UUID`

No migration is required.

---

## v1.0.1

### Fixes

- Fixed handling of empty optional weather and outdoor sensor entity fields during zone creation and editing.
- Improved fallback handling for missing weather entities.

### Notes

Optional weather-related entity fields can now safely be left empty.

---

## v1.0.0

**First stable release — changes since v1.0.0-rc.4**

### Bug Fixes

- **Automatic summer mode indoor safety uses configured night temperature**: The indoor safety threshold that temporarily bypasses automatic summer mode was previously fixed at 16 °C regardless of zone configuration. It now uses the zone's configured night temperature (default 18 °C) as the safety threshold, with a 0.5 °C hysteresis on recovery to prevent rapid toggling around the boundary. Only applies to `summer_mode = automatic`; the manual `on` override is unaffected.

### Stable Finalization

- Version bumped to 1.0.0 — the integration has been validated across multiple RC cycles covering all core features.
- README updated for stable installation: no pre-release flag required in HACS, simplified installation steps.

---

## v1.0.0-rc.4

**Changes since v1.0.0-rc.3**

### Bug Fixes

- **Battery level not detected for Zigbee TRVs**: TRV climate entities (Z2M/ZHA) do not expose battery level as an entity attribute. The previous implementation searched only attributes, so `device_batteries` was always empty for TRV-only zones — making the card's battery warning permanently invisible. Fixed by adding a second lookup path: if no battery attribute is found on the entity itself, ThermoSmart walks the HA entity registry to find a sibling `sensor` entity with `device_class=battery` on the same device. Method 1 (attribute-based) is preserved as the primary path for backwards compatibility.

### Improvements

- **Local brand assets for Geräte & Dienste**: Added `custom_components/thermosmart/brand/` with `icon.png`, `icon@2x.png`, `logo.png`, and `logo@2x.png`. Home Assistant 2026.3+ uses these files to display the ThermoSmart logo in Devices & Services instead of the default placeholder icon. Existing `icon.png` files for HACS and README are unchanged.

### Stability

- Additional RC validation and stability improvements before v1.0.0 Stable.

---

## v1.0.0-rc.3

**Changes since v1.0.0-rc.2**

### Bug Fixes

- **Schedule times ignored (silent fallback)**: HA `TimeSelector` stores times as `HH:MM:SS` (e.g. `06:30:00`). The three schedule parsers in `coordinator.py` and `learning_engine.py` used `h, m = split(":")` unpacking, which raised `ValueError` on three-part strings. The error was silently caught and the hardcoded fallback time was used instead — causing user-configured schedule times to be ignored. Fixed by indexing `parts[0]` / `parts[1]`, which handles both `HH:MM` and `HH:MM:SS` correctly.

### Improvements

- **Mode change: immediate temperature display**: Switching heating mode (Comfort, Night, Eco, Away, Vacation) now immediately updates the target temperature sensor. Previously the card showed the old temperature for several seconds until the next coordinator refresh. `coordinator.set_mode()` now patches `adjusted_target` with the configured preset base value before `async_write_ha_state()` is called; the full weather/learning recalculation on the next refresh overwrites it with the properly adjusted figure.
- **Options flow UI strings**: Cleaned up options flow labels and descriptions across all 25 translation files for consistency.

### Stability

- Additional RC validation and stability improvements before v1.0.0 Stable.

---

## v1.0.0-rc.2

**Changes since v1.0.0-rc.1**

### Fixes

- **Manual override cleared on Away**: Manual temperature overrides are now automatically cleared when all tracked persons leave the home (Away mode).
- **Presence restore**: Presence changes correctly restore automatic schedule behavior after returning home.

### Improvements

- **Status sensor translations**: Status sensor translations now work correctly in all supported languages via `SensorDeviceClass.ENUM` and explicit state options.
- **Branding and README**: Branding assets updated; README now uses a dedicated high-resolution logo separate from the HACS integration icon.

### Stability

- Additional RC validation and stability improvements before v1.0.0 Stable.

---

## v1.0.0-rc.1

**Changes since v1.0.0-beta.23**

### Bug Fixes – Restart & Persistence

- **Override restore after restart**: Manual temperature overrides set via the climate entity are now restored after a Home Assistant restart. The override state is persisted through `extra_state_attributes` and re-applied via `RestoreEntity` on startup.
- **Vacation mode restore**: The Global Vacation switch now re-applies its restored state to all zone coordinators on startup, including coordinators that were not yet loaded when the switch entity initialised.
- **Summer mode restore**: The Global Summer Mode select now re-applies its restored state to all zone coordinators on startup with the same race-condition fix.
- **Global override registry**: The in-memory override registry (`global_vacation_override`, `global_summer_override`) is now kept in sync on every runtime toggle, not only on startup restore. Zones added dynamically will always inherit the current global state.
- **Race condition fix (system entry loads first)**: When the ThermoSmart System entry is loaded before zone entries, the restored global overrides are now written into `hass.data[DOMAIN]` and applied to each zone coordinator as it initialises.

### Translations

- **Summer Mode select states changed to machine-readable keys**: State values are now `automatic`, `on`, `off` (previously `Automatic`, `On`, `Off`). Automations and templates checking these states must be updated accordingly.
- **Translation key coverage**: Global Vacation switch and Summer Mode select now use `_attr_translation_key` with full `has_entity_name` support. Climate preset `sleep` is now translated as "Night" (and language equivalents).
- **17 additional UI languages added** (24 total): BG, CA, CS, DA, EL, ES, FI, HU, NB, PT, RO, RU, SK, SL, TR, UK, ZH-Hans — all complete, all 145 keys covered, UTF-8 without BOM.

### Home Assistant Quality Improvements

- `set_active_control` log level changed from `WARNING` to `INFO` — toggling active control is an expected user action, not an error condition.
- Unused constant `CONF_HEATING_ZONE` removed from `const.py`.
- `sw_version=VERSION` added to the global ThermoSmart System device info (visible in HA device registry for both the Summer Mode select and the Vacation switch).

### RC Workflow

- GitHub Actions release workflow now correctly marks `v*-rc.*` tags as pre-releases (previously only `alpha`, `beta`, `bN` suffixes were recognised).

---

## v1.0.0-beta.23

- Global Summer Mode: `switch.thermosmart_global_summer` replaced by `select.thermosmart_summer_mode` (three-state: Automatic / On / Off). Old entity automatically removed from the Entity Registry on upgrade.
- TPI control with learned coefficients
- Outcome scoring to improve learning quality
- Slope-based window detection
- Weather-aware setpoints with forecast suppression
- Automatic pre-heating
