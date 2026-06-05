"""TPI (Time Proportional Integrator) Regler für ThermoSmart.

Berechnet einen Duty-Cycle (0–100%) aus Raumtemperaturfehler und
Außentemperatur-Delta.  Universell einsetzbar:

  • TRVs mit direkter Ventilsteuerung  → Duty-Cycle direkt auf Ventil schreiben
  • TRVs ohne Ventilsteuerung (TRVZB)  → Duty-Cycle in Boost-Setpoint umrechnen

Formel (etablierter TPI-Ansatz):
  duty = coef_int × (target − room) + coef_ext × (target − outdoor)
  duty_pct = clamp(duty × 100, 0, 100)

Koeffizienten werden automatisch aus gelernten Gebäudedaten abgeleitet:
  coef_int ≈ heat_loss_rate / heat_rate
  coef_ext ≈ coef_int / 50

Das macht TS's TPI besonders: keine manuell gesetzten
Koeffizienten nötig – alles läuft aus den eigenen Lerndaten.
"""
from __future__ import annotations

import logging

_LOGGER = logging.getLogger(__name__)

# Sicherheitsgrenzen
_COEF_INT_MIN = 0.15
_COEF_INT_MAX = 1.2
_COEF_EXT_MIN = 0.002
_COEF_EXT_MAX = 0.06


def compute_tpi(
    target: float,
    room: float,
    outdoor: float | None,
    coef_int: float,
    coef_ext: float,
) -> float:
    """Berechnet TPI Duty-Cycle in Prozent (0.0 – 100.0).

    Args:
        target:    Zieltemperatur (°C)
        room:      aktuelle Raumtemperatur (°C)
        outdoor:   aktuelle Außentemperatur (°C) – None falls unbekannt
        coef_int:  Koeffizient interner Fehler (Einheit: 1/°C)
        coef_ext:  Koeffizient externer Fehler (Einheit: 1/°C)

    Returns:
        Duty-Cycle in % – 0.0 bei Ziel erreicht oder überschritten,
        100.0 bei sehr großem Fehler.
    """
    error_int = target - room
    duty = coef_int * error_int

    if outdoor is not None:
        error_ext = target - outdoor
        duty += coef_ext * error_ext

    duty_pct = duty * 100.0
    result = max(0.0, min(100.0, round(duty_pct, 1)))

    _LOGGER.debug(
        "TPI: target=%.1f room=%.1f outdoor=%s "
        "error_int=%.2f coef_int=%.3f coef_ext=%.4f → duty=%.1f%%",
        target, room,
        f"{outdoor:.1f}" if outdoor is not None else "n/a",
        error_int, coef_int, coef_ext, result,
    )
    return result


def duty_to_setpoint(target: float, duty_pct: float, max_boost: float = 8.0) -> float:
    """Konvertiert TPI Duty-Cycle in einen TRV-Boost-Setpoint.

    Für TRVs die keinen direkten Ventilprozent-Befehl unterstützen:
    Das TRV bekommt einen erhöhten Setpoint statt den Duty-Cycle direkt.

        trv_setpoint = target + (duty_pct / 100) × max_boost

    Beispiel: duty=70%, target=21°C, max_boost=8°C → setpoint=26.6°C
    """
    boost = (duty_pct / 100.0) * max_boost
    setpoint = min(target + boost, target + max_boost, 28.0)
    return round(setpoint, 1)


def estimate_coefficients(
    heat_rate: float | None,
    heat_loss_rate: float | None,
) -> tuple[float, float]:
    """Leitet TPI-Koeffizienten aus gelernten Gebäudedaten ab.

    Physikalische Herleitung:
      Bei 1°C Raumfehler muss der Heizkörper mindestens soviel leisten wie
      das Gebäude verliert:
        coef_int = heat_loss_rate / heat_rate

      Beispiel: heat_loss=0.02°C/min, heat_rate=0.05°C/min
        coef_int = 0.4  → 40% Ventilöffnung pro 1°C Fehler

      coef_ext ergänzt den Außentemperatureinfluss als ~1/50 von coef_int.

    Fallback: Standard-Werte wenn keine Lerndaten verfügbar.
    """
    from .const import TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT

    if not heat_rate or heat_rate <= 0 or not heat_loss_rate or heat_loss_rate <= 0:
        return TPI_COEF_INT_DEFAULT, TPI_COEF_EXT_DEFAULT

    coef_int = heat_loss_rate / heat_rate
    coef_int = round(max(_COEF_INT_MIN, min(_COEF_INT_MAX, coef_int)), 3)
    coef_ext = round(max(_COEF_EXT_MIN, min(_COEF_EXT_MAX, coef_int / 50.0)), 4)

    _LOGGER.debug(
        "TPI Koeffizienten aus Lerndaten: "
        "heat_rate=%.5f°C/min heat_loss=%.5f°C/min "
        "→ coef_int=%.3f coef_ext=%.4f",
        heat_rate, heat_loss_rate, coef_int, coef_ext,
    )
    return coef_int, coef_ext
