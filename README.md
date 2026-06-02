# ThermoSmart

**AI-powered, weather-aware heating control for Home Assistant**

ThermoSmart replaces static heating schedules with a self-learning controller that adapts to your actual habits, outdoor conditions, and building characteristics.

---

## Features

| Feature | Description |
|---|---|
| 🌤️ **Weather Engine** | Adjusts target temperatures based on outdoor temp, wind chill, and forecast |
| 🧠 **Learning Engine** | Observes your home over time, derives optimal targets and preheat times |
| ⏱️ **Preheat Optimiser** | Starts heating early so rooms are warm *when you need them*, not after |
| 🪟 **Window Detection** | Automatically pauses heating when windows are open |
| 📊 **HA Sensors** | Exposes adjusted target, preheat minutes, confidence %, and weather offset per zone |
| 🏖️ **Holiday Mode** | Compatible with existing `input_boolean.urlaubsmodus` automations |

---

## Zones (auto-configured for Mikasmarthome)

- Wohnbereich
- Schlafbereich
- Badezimmer
- Gäste-WC
- Keller (4 thermostats)

---

## Installation

1. Copy `custom_components/thermosmart/` into your HA config directory.
2. Restart Home Assistant.
3. Go to **Settings → Integrations → Add Integration → ThermoSmart**.
4. Select your weather entity (e.g. `weather.wetterstation`) and optional outdoor sensor.

---

## Architecture

```
ThermoSmartCoordinator          (polls every 5 min)
├── WeatherEngine               (reads HA weather entity + outdoor sensor)
│   ├── async_get_data()        → outdoor temp, wind, condition, forecast
│   └── compute_temperature_offset() → °C offset based on conditions
├── LearningEngine              (persisted via HA Storage)
│   ├── async_observe()         → records each coordinator run
│   ├── async_get_base_target() → learned or schedule-based target
│   └── async_get_preheat_minutes() → adaptive lead time
└── Sensor platform             → 4 HA sensors per zone
    ├── Zieltemperatur          (adjusted target °C)
    ├── Vorheizzeit             (preheat minutes)
    ├── Lernfortschritt         (confidence %)
    └── Wetterkorrektur         (weather offset °C)
```

---

## Roadmap

- [ ] `number` platform: per-zone temperature overrides via UI
- [ ] `switch` platform: per-zone enable/disable
- [ ] `select` platform: manual mode selector (auto / comfort / away / night)
- [ ] Forecast-based preheating (cold night ahead → start earlier)
- [ ] Solar gain detection (sunny south windows → reduce heating)
- [ ] HA Energy dashboard integration

---

## License

MIT © 2026 Mikasmarthome
