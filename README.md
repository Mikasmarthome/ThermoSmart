<p align="center">
  <img src="https://raw.githubusercontent.com/Mikasmarthome/ThermoSmart/main/assets/thermosmart-logo.png" alt="ThermoSmart" width="256"/>
</p>

<h1 align="center">ThermoSmart</h1>
<p align="center"><strong>Self-learning, weather-aware heating control for Home Assistant</strong></p>

<p align="center">
  <a href="https://hacs.xyz"><img src="https://img.shields.io/badge/HACS-Custom-orange.svg" alt="HACS Custom"/></a>
  <a href="https://github.com/Mikasmarthome/ThermoSmart/releases"><img src="https://img.shields.io/badge/version-v1.0.4-blue.svg" alt="Version"/></a>
  <img src="https://img.shields.io/badge/status-stable-brightgreen.svg" alt="Stable"/>
  <a href="https://www.home-assistant.io"><img src="https://img.shields.io/badge/HA-2024.1%2B-brightgreen.svg" alt="HA min"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"/></a>
</p>

---

> ⚠️ **Use at your own risk.** ThermoSmart is not affiliated with Home Assistant or Nabu Casa. It controls physical heating devices. Always set a safe minimum temperature and verify behaviour in Observation mode before enabling Active Control.

---

ThermoSmart observes how your building heats and cools, then uses that data — together with weather forecasts and presence detection — to control your TRVs. **One config entry = one heating zone.**

## Why ThermoSmart?

- **Learns your building** — heating rate, heat loss and weather sensitivity derived from real observations in your home, not factory defaults
- **Adapts to conditions** — outdoor temperature, wind, solar radiation and forecasts feed into every decision
- **Starts passively** — Observation mode collects data without touching your existing setup
- **Fully local** — no cloud, no subscription

**Features:** automatic pre-heating · TPI control with learned coefficients · weather-aware setpoints · slope-based window detection · outcome scoring to improve learning quality

---

## Dashboard Card

A dedicated Lovelace card is available: **[thermosmart-card](https://github.com/Mikasmarthome/thermosmart-card)**

Temperature ring, drag-to-set, mode buttons, history sparkline, confidence bar — install separately via HACS (Frontend).

---

## Installation

### 1 — Add Custom Repository

HACS → Integrations → **⋮** → **Custom repositories** → URL: `https://github.com/Mikasmarthome/ThermoSmart` → Category: **Integration** → Add

### 2 — Download

HACS → Integrations → find **ThermoSmart** → **Download** → select `v1.0.4` → Download

### 3 — Restart and Add

Restart Home Assistant, then: **Settings → Integrations → Add Integration → ThermoSmart**

---

## Setup

Add **ThermoSmart System** first (Summer mode select + Vacation switch), then add each heating zone.

**Observation mode is recommended to start.** With Active Control off, ThermoSmart watches your existing TRVs and builds an initial thermal model without changing anything in your setup. Use the first day or two to verify that sensors, temperatures, and schedules are configured correctly — then enable Active Control whenever you're ready. Learning continues in both modes; Observation mode is not a prerequisite.

### Zone Configuration (4 steps)

| Step | Fields |
|---|---|
| **1 – Devices** | Zone name · TRVs · Temperature sensors *(optional)* · Humidity sensors · Window / door sensors · Window delays · Valve maintenance · Calibration invert |
| **2 – Temperatures** | Comfort · Night · Away · Vacation · Eco · Tolerance · Weekday/weekend schedule times |
| **3 – Presence** | Person entities · Home zone · Learning on/off |
| **4 – Weather** | Weather entity · Outdoor temp/humidity · Wind speed · Solar radiation · Precipitation |

> **External temperature sensors are optional.** Since v1.0.3, ThermoSmart automatically uses the TRV's built-in temperature sensor as fallback when no external sensor is configured. Adding a dedicated room sensor improves accuracy, but is not required to get started.

---

## Entities

### Per Zone
| Entity | Description |
|---|---|
| `climate.thermosmart_*` | Virtual thermostat — target/current temp, HVAC mode, presets |
| `select.*_mode` | Heating mode: Auto / Comfort / Eco / Night / Away / Vacation |
| `switch.*_active_control` | Active Control on/off |
| `switch.*_learning` | Learning algorithm on/off |
| `sensor.*_status` | Zone status |
| `sensor.*_confidence` | Learning progress 0–100 % |
| `sensor.*_adjusted_target` | Configured target temperature (schedule or mode) |
| `sensor.*_preheat_minutes` | Calculated preheat lead time in minutes |

Additional diagnostic sensors (disabled by default): TRV setpoint · TPI duty-cycle · Weather offset · Temperature slope · Heat loss · Heating power · Solar gain · TRV observations · Window cooling rate · EMA temperature

### Global (ThermoSmart System)
| Entity | Description |
|---|---|
| `select.thermosmart_summer_mode` | Summer mode — `automatic` / `on` / `off` |
| `switch.thermosmart_vacation_mode` | Vacation mode — all zones → vacation temperature |

---

## Supported Devices

### Developer-tested
| Device | Protocol | Notes |
|---|---|---|
| SONOFF TRVZB | Zigbee via Zigbee2MQTT | Controlled via setpoint boost (`climate.set_temperature`) and `external_temperature_input` for improved TRV-internal accuracy. `valve_opening_degree` is a max-opening limit on this device — ThermoSmart does not write TPI duty-cycle to it. A motor-protection workaround is applied on close. |

### Untested — setpoint boost expected to work
Any `climate` entity supporting `set_temperature`: Danfoss Ally, Eurotronic Spirit, Tuya TS0601, generic ZHA / Z-Wave / Matter TRVs.
ThermoSmart falls back to setpoint boost on devices without a recognised valve entity.

> [Open an issue](https://github.com/Mikasmarthome/ThermoSmart/issues/new) if you test a device — community results are very welcome.

### Not recommended
**Tado / Netatmo and other cloud-based thermostats:** cloud AI periodically overrides setpoints. ThermoSmart would learn the cloud's behaviour rather than your building's.

### Protocol compatibility

| Protocol | Obs. Mode | Active Control | Direct Valve | Auto-Calibration |
|---|---|---|---|---|
| **Zigbee2MQTT** | ✅ | ✅ | ✅ auto-detected¹ | ✅ auto-detected¹ |
| **ZHA** | ✅ | ✅ setpoint boost | ⚠️ device-dependent | ⚠️ device-dependent |
| **Z-Wave JS** | ✅ | ✅ setpoint boost | ❌ | ⚠️ rarely exposed |
| **Matter** | ✅ | ✅ setpoint boost | ❌ | ❌ |
| **Homematic IP** (local) | ✅ | ✅ setpoint boost | ⚠️ `level` pattern, untested | ⚠️ `temperature_offset` |
| **Fritz!DECT / Bosch** | ✅ | ✅ setpoint boost | ❌ | ❌ |
| **Tado / Netatmo** | ⚠️ | ⚠️ not recommended | ❌ | ❌ |

¹ Auto-detection finds a matching entity and writes values to it — it does not guarantee physical function on every device.

**Direct valve control** auto-detects writable `number` entities matching: `pi_heating_demand`, `valve_position`, `heating_demand`, or `level`. Note: `valve_opening_degree` (used on some devices as a max-opening limit, not a live position) is intentionally excluded and not written by ThermoSmart.

**Setpoint boost** converts TPI duty-cycle to `setpoint = target + duty% × 8 °C` via `climate.set_temperature`. Works on all devices; less precise than direct valve control.

---

## Summer & Vacation Mode

### Summer Mode (`select.thermosmart_summer_mode`)

| Option | Behaviour |
|---|---|
| `automatic` | Activates when the 72 h rolling outdoor average exceeds **18 °C**; deactivates below **15 °C** (3 °C hysteresis prevents rapid toggling) |
| `on` | Forces summer mode — heating disabled, frost protection (12 °C) active |
| `off` | Forces winter mode — normal heating regardless of outdoor temperature |

In summer mode all normal heating is suspended. TRVs are held at 12 °C to keep valves exercised.

### Vacation Mode (`switch.thermosmart_vacation_mode`)

Sets all zones to the configured vacation temperature (default 12 °C). Takes priority over schedule and presence detection. Independent of summer mode.

---

## Status Values

| Status | Meaning |
|---|---|
| `observation_mode` | Active Control OFF, learning ON |
| `heating` | Actively heating toward target |
| `idle` | Target reached, maintaining |
| `preheating` | Heating before scheduled comfort time |
| `window_open` | Window detected (sensor or slope), heating paused |
| `away` | All persons away |
| `vacation` | Vacation mode active |
| `summer` | Summer mode — frost protection only |
| `heating_failure` | Heating commanded but temperature falling 35+ min |
| `disabled` | Active Control OFF, learning OFF |

---

## FAQ

**How do I update ThermoSmart?**  
HACS → Integrations → ThermoSmart → **Update** → Restart Home Assistant. Learning data is preserved across updates.

**What is Observation mode?**  
Active Control OFF: ThermoSmart reads what your existing controller does and learns from it — nothing in your setup changes. Switch to Active Control when you're ready.

**How long until the learning algorithm is effective?**  
Learning builds from active heating observations and works in both Observation mode and Active Control. Phase 1 (<5 obs): physical fallback only. Phase 2 (5–50): first patterns emerge. Phase 3 (50+): learning dominates. How quickly you reach each phase depends on how often your heating runs — results improve steadily as observations accumulate.

**A sensor goes offline — what happens?**  
ThermoSmart averages the remaining sensors. Temperature decisions pause only if all sensors in a zone are unavailable simultaneously.

**Where is the learning data stored?**  
`/config/.storage/thermosmart_learning_data` — safe to share for debugging.

**SONOFF TRVZB via Zigbee2MQTT — `heat` vs `auto` mode and internal weekly schedule**  
Zigbee2MQTT exposes two HVAC modes on the SONOFF TRVZB:

- **`heat`** — manual setpoint mode: the TRV follows the setpoint written by ThermoSmart (or any other controller)
- **`auto`** — the TRV follows its own internal weekly schedule and ignores externally written setpoints

When ThermoSmart is in **Active Control**, it calls `climate.set_temperature` on every cycle. This automatically switches the TRV to `heat` mode, so the internal weekly schedule is bypassed — the TRV follows ThermoSmart's setpoints instead.

**You do not need to delete the internal schedule.** It has no effect while the TRV is in `heat` mode. If Active Control is ever disabled and the TRV is switched back to `auto` manually, the internal schedule becomes active again.

In **Observation mode** ThermoSmart does not write setpoints, so the TRV stays in whatever mode it was last set to (typically `auto` after initial pairing). Observations and learning are recorded correctly regardless of the TRV's active mode.

---

## Breaking Changes

### v1.0.0-rc.1 — Summer mode state values changed to lowercase

The state values of `select.thermosmart_summer_mode` were changed from `Automatic` / `On` / `Off` to `automatic` / `on` / `off`.

**Automations and templates that check these states must be updated.** For example: `state: "Automatic"` → `state: "automatic"`.

---

### v1.0.0-beta.23 — Summer mode switch replaced by select

| Removed entity | New entity |
|---|---|
| `switch.thermosmart_global_summer` | `select.thermosmart_summer_mode` |

The old entity is automatically removed from the Entity Registry on upgrade. **Dashboards and automations referencing the old entity must be updated manually.**  
New options: `automatic` · `on` · `off`

---

## Languages

24 UI languages supported: BG · CA · CS · DA · DE · EL · EN · ES · FI · FR · HU · IT · NB · NL · PL · PT · RO · RU · SK · SL · SV · TR · UK · ZH-Hans

Additional languages welcome via Pull Request (`custom_components/thermosmart/translations/`).

---

## Changelog

Full release history: [RELEASE_NOTES.md](RELEASE_NOTES.md)

---

## Contributing

**Bugs** → [GitHub Issues](https://github.com/Mikasmarthome/ThermoSmart/issues/new): HA version, ThermoSmart version, what happened, log lines (`Settings → System → Logs → thermosmart`)

**Features** → [Issues](https://github.com/Mikasmarthome/ThermoSmart/issues/new) with label `enhancement`

**Especially welcome:** Testing with more TRV models · Additional translations · Device quirk patterns

[Discussions →](https://github.com/Mikasmarthome/ThermoSmart/discussions)

---

## License

MIT — see [LICENSE](LICENSE)
