"""Temperature-unit normalization for external inputs.

ThermoSmart always computes internally in Celsius: °C for temperature, °C/h
for Heat Rate/Heat Loss, °C for Boost Offset/TPI/Afterheat/Onset/Outcome.
Every external temperature value (sensor state, weather/climate attribute,
TRV setpoint) must be normalized to °C exactly once, right where it enters
the integration — via Home Assistant's own ``TemperatureConverter``, never
private °F/°C math, so Kelvin and any future unit HA adds are handled too.

Two entity-attribute conventions exist in Home Assistant and callers must
pick the right one:
  - ``sensor.*`` entities carry their own unit in
    ``state.attributes["unit_of_measurement"]``.
  - ``weather.*`` entities carry their own unit in
    ``state.attributes["temperature_unit"]`` (a dedicated attribute, not
    ``unit_of_measurement``).
  - ``climate.*`` entities (current_temperature/temperature/min_temp/
    max_temp) expose NO unit attribute at all — Home Assistant's climate
    platform (``homeassistant.helpers.temperature.display_temp``) already
    converts these to ``hass.config.units.temperature_unit`` (the HA system
    unit) before they ever reach ``state.attributes``. Passing no ``unit``
    for these correctly falls back to the system unit.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.unit_conversion import TemperatureConverter

_VALID_UNITS = {
    UnitOfTemperature.CELSIUS,
    UnitOfTemperature.FAHRENHEIT,
    UnitOfTemperature.KELVIN,
}
_INVALID_STATES = {"", "unknown", "unavailable", "none"}

# Plausibility band for an already-internal-Celsius temperature reading.
# Wide enough for any real room/TRV/outdoor sensor, narrow enough to reject
# garbage/sentinel values a misbehaving device can still report even though
# the raw float parses and the unit conversion "succeeds" — e.g. Zigbee
# error sentinels like 127°C / 126.5°C, or -100°C used by some firmwares as
# an error code. A unit-conversion bug (e.g. an unconverted °F reading
# mistaken for °C) also tends to land outside this band, so the guard
# doubles as a cheap cross-check on the Temperature-Unit-Fix.
PLAUSIBLE_TEMPERATURE_MIN_C = -30.0
PLAUSIBLE_TEMPERATURE_MAX_C = 50.0


def to_internal_temperature_c(
    hass: HomeAssistant, value: Any, *, unit: Optional[str] = None,
) -> Optional[float]:
    """Normalize an external temperature reading to Celsius.

    ``value`` is typically ``state.state`` or a raw attribute value; any
    unparseable/missing/placeholder form (``None``, empty string,
    ``"unknown"``/``"unavailable"``) returns ``None`` — never raises.

    ``unit`` should be the source's own temperature unit when known (e.g.
    a sensor's ``unit_of_measurement`` or a weather entity's
    ``temperature_unit`` attribute). When omitted or not a recognized
    temperature unit, falls back to ``hass.config.units.temperature_unit``
    (the HA system unit) — the correct assumption for climate attributes,
    which HA itself already normalizes to the system unit.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _INVALID_STATES:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    source_unit = unit if unit in _VALID_UNITS else hass.config.units.temperature_unit
    try:
        return TemperatureConverter.convert(numeric, source_unit, UnitOfTemperature.CELSIUS)
    except HomeAssistantError:
        return None


def from_internal_temperature_c(hass: HomeAssistant, value_c: Optional[float]) -> Optional[float]:
    """Convert an internal Celsius value to the HA system unit for dispatch.

    Home Assistant's ``climate.set_temperature`` service always interprets
    the ``temperature`` field as being in ``hass.config.units.temperature_unit``
    (verified against ``homeassistant.components.climate.async_service_temperature_set``,
    which converts the service value from the system unit to the target
    entity's own unit before validating min/max) — never the target entity's
    native unit directly. This must run right before dispatch, on the final
    value only, to avoid a double conversion.
    """
    if value_c is None:
        return None
    system_unit = hass.config.units.temperature_unit
    if system_unit not in _VALID_UNITS or system_unit == UnitOfTemperature.CELSIUS:
        return value_c
    return TemperatureConverter.convert(value_c, UnitOfTemperature.CELSIUS, system_unit)


def is_plausible_temperature_c(value: Optional[float], *, context: Optional[str] = None) -> bool:
    """Return True if an already-internal-Celsius reading looks physically real.

    Rejects ``None``, non-numeric input, NaN/inf, and anything outside
    ``[PLAUSIBLE_TEMPERATURE_MIN_C, PLAUSIBLE_TEMPERATURE_MAX_C]``. Never
    raises. ``context`` is an optional caller-supplied label (e.g. a sensor
    or entity id) for the caller's own logging — this function is pure and
    never logs itself, matching ``to_internal_temperature_c``'s style.

    Does not distinguish room/TRV/outdoor use — one shared band is simpler
    and already comfortably covers all three (indoor setpoints run roughly
    5–35°C, outdoor sensors rarely exceed -30..50°C even in extreme climates).
    """
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if math.isnan(numeric) or math.isinf(numeric):
        return False
    return PLAUSIBLE_TEMPERATURE_MIN_C <= numeric <= PLAUSIBLE_TEMPERATURE_MAX_C


def state_temperature_unit(state: Optional[State]) -> Optional[str]:
    """Return a sensor-style state's own ``unit_of_measurement``, if present."""
    if state is None:
        return None
    return state.attributes.get("unit_of_measurement")
