# ThermoSmart – Release Notes

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
