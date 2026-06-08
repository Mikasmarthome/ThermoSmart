# ThermoSmart – Release Notes

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
