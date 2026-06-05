<p align="center">
  <img src="icon.png" alt="ThermoSmart" width="180"/>
</p>

<h1 align="center">ThermoSmart</h1>
<p align="center"><strong>Self-learning, weather-aware heating control for Home Assistant</strong></p>

<p align="center">
  <a href="https://hacs.xyz"><img src="https://img.shields.io/badge/HACS-Custom-orange.svg" alt="HACS Custom"/></a>
  <a href="https://github.com/Mikasmarthome/ThermoSmart/releases"><img src="https://img.shields.io/badge/version-v1.0.1--beta.2-orange.svg" alt="Version"/></a>
  <img src="https://img.shields.io/badge/status-beta-red.svg" alt="Beta"/>
  <a href="https://www.home-assistant.io"><img src="https://img.shields.io/badge/HA-2024.1%2B-brightgreen.svg" alt="HA min"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Mikasmarthome/ThermoSmart" alt="License"/></a>
</p>

---

> ⚠️ **Use at your own risk.** ThermoSmart is not affiliated with Home Assistant or Nabu Casa. It controls physical heating devices. Always set a safe minimum temperature and verify behaviour in Observation mode before enabling Active Control.

> 🧪 **Beta (v1.0.1-beta.2)** — Core features are functional. The learning algorithm improves with each heating season. Please report issues on GitHub — it helps a lot.

---

ThermoSmart observes how your building heats and cools, then uses that data — together with weather forecasts and presence detection — to control your TRVs precisely. **One config entry = one heating zone.**

- **Observation mode** — runs passively alongside any existing controller, learns without changing anything
- **TPI control** with coefficients auto-derived from your building's actual thermal data
- **Automatic pre-heating** based on learned heating rate and outdoor conditions
- **Weather-aware** — outdoor temp, wind, solar radiation and forecast all feed into the target
- **Outcome Scoring** — grades every heating session; slow or overshooting sessions are down-weighted in learning
- **Fully local** — no cloud, no subscription

---

## Dashboard Card

A dedicated Lovelace card is available: **[thermosmart-card](https://github.com/Mikasmarthome/thermosmart-card)**

Temperature ring, drag-to-set, mode buttons, 3-hour history sparkline, learning confidence bar and diagnostic chips — all in one card. Install separately via HACS (Frontend).

---

## Supported Devices

### Tested
| Device | Protocol | Direct Valve Control |
|---|---|---|
| SONOFF TRVZB | Zigbee (via Zigbee2MQTT) | ✅ `valve_opening_degree` |

### Should work (untested)
Any HA `climate` entity supporting `set_temperature` — Danfoss Ally, Eurotronic Spirit, Tuya TS0601, generic Z-Wave / Zigbee TRVs.

> Core setpoint control should work on all of these. Direct valve control and auto-calibration depend on what each device exposes. [Open an issue](https://github.com/Mikasmarthome/ThermoSmart/issues/new) if you test one.

**Direct valve control** is auto-detected for TRVs exposing a writable `number` entity matching: `valve_opening_degree`, `pi_heating_demand`, `valve_position`, or `level`.

---

## Compatibility

| Protocol | Obs. Mode | Active Control | Direct Valve | Auto-Calibration |
|---|---|---|---|---|
| **Zigbee2MQTT** | ✅ | ✅ Full | ✅ Auto-detected | ✅ Auto-detected |
| **ZHA** | ✅ | ✅ Setpoint boost | ⚠️ Device-dependent | ⚠️ Device-dependent |
| **Z-Wave JS** | ✅ | ✅ Setpoint boost | ❌ | ⚠️ Rarely exposed |
| **Matter** | ✅ | ✅ Setpoint boost | ❌ | ❌ |
| **Homematic IP** (local) | ✅ | ✅ Setpoint boost | ✅ `level` auto-detected | ⚠️ `temperature_offset` |
| **Fritz!DECT** | ✅ | ✅ Setpoint boost | ❌ | ❌ |
| **Bosch Smart Home** | ✅ | ✅ Setpoint boost | ❌ | ❌ |
| **Tado / Netatmo** | ⚠️ | ⚠️ | ❌ | ❌ |
| **generic_thermostat / MQTT HVAC** | ✅ | ✅ Setpoint boost | ❌ | ❌ |

**Setpoint boost mode:** TPI duty-cycle is converted to `setpoint = target + duty% × 8°C` and written via `climate.set_temperature`. Works on all devices; less precise than direct valve control.

> **Tado / Netatmo:** Cloud AI periodically overrides setpoints. In Observation mode ThermoSmart would learn the cloud AI's behaviour rather than your building's. Not recommended.

---

## Installation

1. **HACS** → Integrations → ⋮ → **Custom Repositories**
2. URL: `https://github.com/Mikasmarthome/ThermoSmart` → Integration → Add
3. Download **ThermoSmart** and restart Home Assistant
4. **Settings → Integrations → Add Integration → ThermoSmart**

---

## Setup

Add **ThermoSmart System** first (global Summer + Vacation switches), then add each heating zone.

**Recommended:** Start with Active Control **OFF** (Observation mode). ThermoSmart learns from your existing setup without changing anything. Switch to Active Control after a few weeks.

### Zone Configuration (4 steps)

| Step | Fields |
|---|---|
| **1 – Devices** | Zone name · TRVs · Temperature sensors · Humidity sensors · Window sensors · Window delays · Valve maintenance · Calibration invert |
| **2 – Temperatures** | Comfort (21°C) · Night (18°C) · Away (17°C) · Vacation (12°C) · Eco (19°C) · Tolerance (0.5°C) · Weekday/weekend schedule times |
| **3 – Presence** | Person entities · Home zone · Learning on/off |
| **4 – Weather** | Weather entity · Outdoor temp/humidity · Wind speed · Solar radiation · Precipitation |

---

## Entities

### Per Zone
| Entity | Description |
|---|---|
| `climate.thermosmart_*` | Virtual thermostat — target/current temp, HVAC mode, presets |
| `select.*_heizmodus` | Heating mode: Auto / Comfort / Eco / Night / Away / Vacation |
| `switch.*_aktive_steuerung` | Active Control on/off |
| `switch.*_lernmodus` | Learning algorithm on/off |
| `sensor.*_status` | Zone status |
| `sensor.*_lernfortschritt` | Learning confidence 0–100% |
| `sensor.*_zieltemperatur` | Adjusted target temperature (incl. weather correction) |
| `sensor.*_vorheizzeit` | Calculated preheat time in minutes |

Additional diagnostic sensors (disabled by default): TRV setpoint · TPI duty-cycle · Weather offset · Temperature slope · Heat loss · Heating power · Solar gain · TRV observations · Window cooling rate · EMA temperature

### Global (ThermoSmart System)
| Entity | Description |
|---|---|
| `switch.thermosmart_sommer_modus` | Summer mode — all zones → frost protection |
| `switch.thermosmart_urlaubsmodus` | Vacation mode — all zones → vacation temperature |

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

**Do I need weather data?**
No. ThermoSmart falls back to physical estimates. Accuracy improves with a weather entity configured.

**What is Observation mode?**
Active Control OFF: ThermoSmart reads what your existing controller does and learns from it — nothing in your setup changes. Switch to Active Control when you're ready.

**How long until the learning algorithm is effective?**
Confidence builds from **heating observations** (room temperature actively rising). Phase 1 (<5): physical fallback. Phase 2 (5–50): first patterns. Phase 3 (50+): learning dominates. Expect several weeks of regular heating. Exact timelines have not yet been validated at scale.

**A sensor goes offline — what happens?**
ThermoSmart averages the remaining sensors. Temperature decisions pause only if all sensors are unavailable.

**Where is the learning data stored?**
`/config/.storage/thermosmart_learning_data` — safe to share for debugging.

---

## Languages

DE · EN · FR · NL · PL · SV · IT — all complete. Additional languages welcome via Pull Request (`custom_components/thermosmart/translations/`).

---

## Contributing

**Bugs** → [GitHub Issues](https://github.com/Mikasmarthome/ThermoSmart/issues/new): HA version, ThermoSmart version, what happened, log lines (`Settings → System → Logs → thermosmart`)

**Features** → [Issues](https://github.com/Mikasmarthome/ThermoSmart/issues/new) with label `enhancement`

**Especially welcome:** Testing with more TRV models · Additional translations · Device quirk patterns

[Discussions →](https://github.com/Mikasmarthome/ThermoSmart/discussions)

---

## License

MIT — see [LICENSE](LICENSE)
