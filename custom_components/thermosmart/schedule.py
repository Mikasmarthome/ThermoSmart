"""Pure schedule resolution — the single source of truth for the day/night
comfort schedule.

Pure functions only: an explicit ``now_local`` is passed in (no clock read), no
Home Assistant dependency, no learning dependency, no hidden mutable state. The
coordinator's ``_current_schedule_period``, ``_minutes_until_next_comfort`` and
base-target resolution all delegate here.

Schedule model: a weekday and a weekend
pair of morning/night boundaries. Outside ``[morning, night)`` the zone is in
the night block; inside it, the comfort block. Defaults: weekday 06:00/22:00,
weekend 08:00/23:00.

Period labels are only used to detect a change of schedule slot (a manual
override ends with the slot it was set in) and as observation context. With the
schedule disabled there are no slots, so the label is the constant
``always_comfort``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .const import (
    CONF_SCHED_WD_MORNING,
    CONF_SCHED_WD_NIGHT,
    CONF_SCHED_WE_MORNING,
    CONF_SCHED_WE_NIGHT,
    CONF_SCHEDULE_ENABLED,
    HEATING_MODE_AUTO,
)

_WD_MORNING_DEFAULT = "06:00"
_WD_NIGHT_DEFAULT = "22:00"
_WE_MORNING_DEFAULT = "08:00"
_WE_NIGHT_DEFAULT = "23:00"


def _hhmm_to_minutes(raw: object, fallback: str) -> int:
    """Parse ``"HH:MM"`` to minutes-since-midnight; fall back on malformed input.

    Mirrors the former inline ``t()`` helpers exactly, including catching
    ValueError/AttributeError/IndexError and re-parsing the fallback string.
    """
    try:
        parts = str(raw).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, AttributeError, IndexError):
        h, m = fallback.split(":")
        return int(h) * 60 + int(m)


def _day_slots(is_weekend: bool, cfg: dict) -> tuple[int, int]:
    """Return ``(morning_minutes, night_minutes)`` for a weekday or weekend day."""
    if is_weekend:
        morning = _hhmm_to_minutes(cfg.get(CONF_SCHED_WE_MORNING, _WE_MORNING_DEFAULT), _WE_MORNING_DEFAULT)
        night = _hhmm_to_minutes(cfg.get(CONF_SCHED_WE_NIGHT, _WE_NIGHT_DEFAULT), _WE_NIGHT_DEFAULT)
    else:
        morning = _hhmm_to_minutes(cfg.get(CONF_SCHED_WD_MORNING, _WD_MORNING_DEFAULT), _WD_MORNING_DEFAULT)
        night = _hhmm_to_minutes(cfg.get(CONF_SCHED_WD_NIGHT, _WD_NIGHT_DEFAULT), _WD_NIGHT_DEFAULT)
    return morning, night


def _slots(now_local: datetime, cfg: Optional[dict]) -> tuple[bool, int, int, int]:
    """Return ``(is_weekend, cur_minutes, morning_minutes, night_minutes)``."""
    cfg = cfg or {}
    is_weekend = now_local.weekday() >= 5
    cur = now_local.hour * 60 + now_local.minute
    morning, night = _day_slots(is_weekend, cfg)
    return is_weekend, cur, morning, night


def current_schedule_period(now_local: datetime, cfg: Optional[dict]) -> str:
    """Return the active slot label: ``wd_night``/``wd_comfort``/``we_night``/``we_comfort``.

    When the schedule is disabled there are no slots and the zone is always in
    comfort: the label is the constant ``always_comfort``, independent of the
    day, so no slot change (and no override expiry) can happen. The night that
    runs past midnight keeps the day-type of the evening it started on, so a
    manual override set in the evening does not expire at midnight purely
    because the calendar day flips (it ends at the next morning, as intended).
    """
    cfg = cfg or {}
    if not cfg.get(CONF_SCHEDULE_ENABLED, True):
        return "always_comfort"
    is_weekend, cur, morning, night = _slots(now_local, cfg)
    today = "we" if is_weekend else "wd"
    if morning <= cur < night:
        return f"{today}_comfort"
    if cur >= night:
        return f"{today}_night"
    # cur < morning: the early-morning night is the continuation of the previous
    # calendar day's evening night — label it with that day's type.
    yesterday_weekend = ((now_local.weekday() - 1) % 7) >= 5
    return f"{'we' if yesterday_weekend else 'wd'}_night"


def minutes_until_next_comfort(now_local: datetime, cfg: Optional[dict]) -> Optional[int]:
    """Minutes until the next comfort window opens, or ``None`` if already inside it
    (or if the schedule is disabled — comfort is then always active, so there is
    no upcoming transition to pre-heat toward)."""
    cfg = cfg or {}
    if not cfg.get(CONF_SCHEDULE_ENABLED, True):
        return None
    _is_weekend, cur, morning, night = _slots(now_local, cfg)
    if morning <= cur < night:
        return None
    if cur < morning:
        return morning - cur
    # cur >= night: the next comfort window is the NEXT calendar day's morning,
    # which may be a different day-type (weekday vs weekend) than today.
    tomorrow_weekend = ((now_local.weekday() + 1) % 7) >= 5
    tomorrow_morning, _night = _day_slots(tomorrow_weekend, cfg)
    return (24 * 60 - cur) + tomorrow_morning


def schedule_slot_target(
    now_local: datetime,
    cfg: Optional[dict],
    *,
    comfort_temp: float,
    night_temp: float,
) -> float:
    """Return the configured target for the current slot (comfort vs night)."""
    _is_weekend, cur, morning, night = _slots(now_local, cfg)
    if cur < morning or cur >= night:
        return night_temp
    return comfort_temp


def resolve_base_target(
    now_local: datetime,
    mode: str,
    *,
    comfort_temp: float = 21.0,
    night_temp: float = 18.0,
    away_temp: float = 17.0,
    vacation_temp: float = 12.0,
    eco_temp: float,
    cfg: Optional[dict] = None,
) -> float:
    """Resolve the base target temperature for ``mode``.

    Non-AUTO modes map straight to their configured temperature (unknown modes
    fall back to night). In AUTO the target follows the schedule slot, unless
    the schedule is disabled, in which case it is always comfort.
    """
    mode_temps = {
        "comfort": comfort_temp,
        "night": night_temp,
        "away": away_temp,
        "vacation": vacation_temp,
        "eco": eco_temp,
    }
    if mode != HEATING_MODE_AUTO:
        return mode_temps.get(mode, night_temp)

    if (cfg or {}).get(CONF_SCHEDULE_ENABLED, True):
        return schedule_slot_target(
            now_local, cfg, comfort_temp=comfort_temp, night_temp=night_temp
        )
    return comfort_temp
