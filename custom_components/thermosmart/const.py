"""Constants for the ThermoSmart integration."""

DOMAIN = "thermosmart"
VERSION = "0.0.4"

# Config entry keys
CONF_CLIMATE_ENTITIES = "climate_entities"
CONF_WEATHER_ENTITY = "weather_entity"
CONF_OUTDOOR_TEMP_SENSOR = "outdoor_temp_sensor"
CONF_LEARNING_ENABLED = "learning_enabled"
CONF_HEATING_ZONE = "heating_zone"

# Default values
DEFAULT_WEATHER_ENTITY = "weather.home"
DEFAULT_LEARNING_ENABLED = True
DEFAULT_SCAN_INTERVAL = 300  # seconds

# Zone names (matching your setup)
ZONES = {
    "wohnbereich": {
        "name": "Wohnbereich",
        "climate_entity": "climate.wohnbereich_thermostat_steuerung",
        "window_sensor": "binary_sensor.fenster_wohnbereich",
        "temp_sensor": "sensor.temperatur_wohnbereich_durchschnitt",
        "humidity_sensor": "sensor.luftfeuchtigkeit_wohnbereich_durchschnitt",
    },
    "schlafbereich": {
        "name": "Schlafbereich",
        "climate_entity": "climate.schlafbereich_thermostat_steuerung",
        "window_sensor": "binary_sensor.fenster_schlafbereich",
        "temp_sensor": "sensor.temperatur_schlafbereich_durchschnitt",
        "humidity_sensor": "sensor.luftfeuchtigkeit_schlafbereich_durchschnitt",
    },
    "badezimmer": {
        "name": "Badezimmer",
        "climate_entity": "climate.badezimmer_thermostat_steuerung",
        "window_sensor": "binary_sensor.fenster_badezimmer",
        "temp_sensor": "sensor.temperatursensor_badezimmer_temperature",
        "humidity_sensor": "sensor.temperatursensor_badezimmer_humidity",
    },
    "gaste_wc": {
        "name": "Gäste-WC",
        "climate_entity": "climate.gaste_wc_thermostat_steuerung",
        "window_sensor": None,
        "temp_sensor": "sensor.temperatursensor_gaste_wc_temperature",
        "humidity_sensor": "sensor.temperatursensor_gaste_wc_humidity",
    },
    "keller": {
        "name": "Keller",
        "climate_entity": [
            "climate.keller_1_thermostat_steuerung",
            "climate.keller_2_thermostat_steuerung",
            "climate.keller_hinten_thermostat_steuerung",
            "climate.waschkuche_thermostat_steuerung",
        ],
        "window_sensor": "binary_sensor.fenster_keller_raum_1",
        "temp_sensor": "sensor.temperatursensor_keller_1_temperature",
        "humidity_sensor": "sensor.temperatursensor_keller_1_humidity",
    },
}

# Heating schedule defaults (°C)
TEMP_COMFORT = 21.0
TEMP_NIGHT = 18.0
TEMP_AWAY = 17.0
TEMP_FROST_PROTECTION = 12.0
TEMP_BATHROOM_COMFORT = 22.0
TEMP_BATHROOM_NIGHT = 18.0
TEMP_GASTE_WC_COMFORT = 20.0
TEMP_KELLER_COMFORT = 15.0

# Weather-based adjustments
WEATHER_COLD_THRESHOLD = 0.0     # °C outdoor — boost heating
WEATHER_MILD_THRESHOLD = 10.0   # °C outdoor — reduce heating
WEATHER_WARM_THRESHOLD = 18.0   # °C outdoor — consider disabling
WEATHER_HOT_THRESHOLD = 22.0    # °C outdoor — disable heating

WEATHER_BOOST_OFFSET = 1.5      # °C added when very cold
WEATHER_REDUCE_OFFSET = -1.0    # °C subtracted when mild

# Wind chill factor: extra boost when windy + cold
WIND_THRESHOLD_MS = 10.0        # m/s
WIND_CHILL_BOOST = 0.5          # °C extra when cold AND windy

# Learning algorithm settings
LEARNING_WINDOW_DAYS = 30       # Tage für Konfidenz-Berechnung (Daten bleiben dauerhaft)
LEARNING_MIN_SAMPLES = 5        # Mindest-Beobachtungen bevor Lernen aktiv wird
LEARNING_DECAY_HALFLIFE = 90    # Halbwertszeit in Tagen (ältere Daten verlieren Gewicht)
PREHEAT_MAX_MINUTES = 60        # maximale Vorheizzeit
PREHEAT_MIN_DELTA = 2.0         # °C Differenz ab der Vorheizung ausgelöst wird

# Präsenz & Automatik
PRESENCE_PERSONS = ["person.resident_1", "person.resident_2"]
PRESENCE_PREHEAT_ZONE = "zone.heizungszone"   # Zone → Heizung startet beim Reinfahren
VACATION_BOOLEAN = "input_boolean.urlaubsmodus"
OUTDOOR_TEMP_SENSOR_EXTRA = "sensor.outdoor_temperature"  # eigene Wetterstation

# Sommer/Winter-Erkennung (aus deinen Automationen übernommen)
SUMMER_THRESHOLD = 18.0   # °C Außentemperatur über X Stunden → Sommer
WINTER_THRESHOLD = 15.0   # °C Außentemperatur unter X Stunden → Heizung an
SEASON_HOURS = 72         # Stunden die der Schwellwert überschritten sein muss

# Heizmodi
HEATING_MODE_AUTO     = "auto"
HEATING_MODE_COMFORT  = "comfort"
HEATING_MODE_NIGHT    = "night"
HEATING_MODE_AWAY     = "away"
HEATING_MODE_VACATION = "vacation"
HEATING_MODES = [
    HEATING_MODE_AUTO,
    HEATING_MODE_COMFORT,
    HEATING_MODE_NIGHT,
    HEATING_MODE_AWAY,
    HEATING_MODE_VACATION,
]

# Komforttemperaturen pro Zone und Modus
ZONE_TEMPS: dict[str, dict[str, float]] = {
    "wohnbereich":  {"comfort": 21.0, "night": 18.0, "away": 17.0, "vacation": 12.0},
    "schlafbereich":{"comfort": 21.0, "night": 18.0, "away": 17.0, "vacation": 12.0},
    "badezimmer":   {"comfort": 22.0, "night": 18.0, "away": 17.0, "vacation": 12.0},
    "gaste_wc":     {"comfort": 20.0, "night": 18.0, "away": 17.0, "vacation": 12.0},
    "keller":       {"comfort": 15.0, "night": 15.0, "away": 15.0, "vacation": 12.0},
}

# Wochentags-Zeitplan (aus deinen Automationen extrahiert)
# Format: [{"time": "HH:MM", "temp": float}]
SCHEDULE: dict[str, dict[str, list]] = {
    "wohnbereich": {
        "weekday": [
            {"time": "05:30", "temp": 21.0},
            {"time": "22:00", "temp": 18.0},
        ],
        "weekend": [
            {"time": "06:30", "temp": 21.0},
            {"time": "22:00", "temp": 18.0},
        ],
    },
    "schlafbereich": {
        "weekday": [
            {"time": "05:30", "temp": 21.0},
            {"time": "22:00", "temp": 18.0},
        ],
        "weekend": [
            {"time": "06:30", "temp": 21.0},
            {"time": "22:00", "temp": 18.0},
        ],
    },
    "badezimmer": {
        "weekday": [
            {"time": "05:30", "temp": 22.0},
            {"time": "22:00", "temp": 18.0},
        ],
        "weekend": [
            {"time": "06:30", "temp": 22.0},
            {"time": "22:00", "temp": 18.0},
        ],
    },
    "gaste_wc": {
        "weekday": [
            {"time": "06:00", "temp": 20.0},
            {"time": "22:00", "temp": 18.0},
        ],
        "weekend": [
            {"time": "06:00", "temp": 20.0},
            {"time": "22:00", "temp": 18.0},
        ],
    },
    "keller": {
        "weekday": [{"time": "00:00", "temp": 15.0}],
        "weekend": [{"time": "00:00", "temp": 15.0}],
    },
}

# Storage keys
STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}_learning_data"

# Platforms
PLATFORMS = ["sensor", "number", "switch", "select"]

# Attributes
ATTR_ZONE = "zone"
ATTR_TARGET_TEMP = "target_temperature"
ATTR_CURRENT_TEMP = "current_temperature"
ATTR_OUTDOOR_TEMP = "outdoor_temperature"
ATTR_WEATHER_CONDITION = "weather_condition"
ATTR_WIND_SPEED = "wind_speed"
ATTR_PREHEAT_MINUTES = "preheat_minutes"
ATTR_LEARNING_CONFIDENCE = "learning_confidence"
ATTR_WEATHER_OFFSET = "weather_offset"
ATTR_LAST_UPDATED = "last_updated"

# Icons
ICON_THERMOSMART = "mdi:thermostat-auto"
ICON_ZONE = "mdi:home-thermometer"
ICON_LEARNING = "mdi:brain"
ICON_WEATHER_ADJUST = "mdi:weather-partly-cloudy"
ICON_PREHEAT = "mdi:timer-play"
