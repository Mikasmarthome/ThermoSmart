"""MetricsCollector — records per-step simulation metrics for S1 invariant checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DayMetrics:
    """Metrics accumulated within a single simulated calendar day (UTC)."""
    day_idx: int
    comfort_steps: int = 0
    cold_steps: int = 0
    overshoot_steps: int = 0
    heating_steps: int = 0
    service_calls: int = 0
    boost_applied_steps: int = 0
    total_heating_wh: float = 0.0


@dataclass
class MetricsSummary:
    total_steps: int = 0
    comfort_steps: int = 0
    cold_steps: int = 0
    overshoot_steps: int = 0
    boost_applied_steps: int = 0
    total_heating_wh: float = 0.0
    boost_offset_sum_c: float = 0.0
    # S1a Part 2 extensions
    service_call_count: int = 0
    adaptation_mode_counts: dict = field(default_factory=dict)
    daily: list = field(default_factory=list)   # list[DayMetrics]

    @property
    def comfort_fraction(self) -> float:
        return self.comfort_steps / max(1, self.total_steps)

    @property
    def cold_fraction(self) -> float:
        return self.cold_steps / max(1, self.total_steps)

    @property
    def overshoot_fraction(self) -> float:
        return self.overshoot_steps / max(1, self.total_steps)

    @property
    def mean_boost_offset_c(self) -> float:
        return self.boost_offset_sum_c / max(1, self.boost_applied_steps)

    def dominant_mode(self) -> Optional[str]:
        """Return the most frequent adaptation_mode observed, or None."""
        if not self.adaptation_mode_counts:
            return None
        return max(self.adaptation_mode_counts, key=self.adaptation_mode_counts.get)


@dataclass
class _StepTrace:
    ts: str
    room_temp: float
    commanded_setpoint: float
    outdoor_temp: float
    trv_setpoint: Optional[float]
    effective_target: Optional[float]
    boost_offset_c: float
    applied_boost_offset_c: float
    schedule_period: Optional[str]
    is_heating: bool


class MetricsCollector:
    """Accumulates per-step metrics and retains a bounded trace for diagnostics."""

    def __init__(
        self,
        comfort_temp: float = 21.0,
        night_temp: float = 18.0,
        comfort_tolerance: float = 0.5,
        safety_floor_offset: float = -2.0,
        overshoot_ceiling_offset: float = 1.5,
        max_trace_len: int = 500,
    ) -> None:
        self.comfort_temp = comfort_temp
        self.night_temp = night_temp
        self._comfort_threshold = comfort_temp - comfort_tolerance
        self._cold_threshold = night_temp + safety_floor_offset
        self._overshoot_threshold = comfort_temp + overshoot_ceiling_offset
        self._max_trace = max_trace_len
        self._summary = MetricsSummary()
        self._trace: list[_StepTrace] = []
        self._current_day_idx: int = -1
        self._current_day: Optional[DayMetrics] = None

    def _get_or_create_day(self, sim_time: datetime) -> DayMetrics:
        """Return the DayMetrics bucket for the given simulation time."""
        s = self._summary
        day_idx = int((sim_time.timestamp() // 86400))
        if self._current_day is None or day_idx != self._current_day_idx:
            idx = len(s.daily)
            self._current_day = DayMetrics(day_idx=idx)
            self._current_day_idx = day_idx
            s.daily.append(self._current_day)
        return self._current_day

    def record(
        self,
        sim_time: datetime,
        zone: dict,
        room,
        outdoor_temp: float,
        step_s: float,
        service_calls_this_step: int = 0,
    ) -> None:
        room_temp = room.room_temp
        boost_offset = zone.get("boost_offset_c") or 0.0
        applied_boost = zone.get("applied_boost_offset_c") or 0.0
        adaptation_mode = zone.get("adaptation_mode") or "unknown"

        s = self._summary
        s.total_steps += 1
        if room_temp >= self._comfort_threshold:
            s.comfort_steps += 1
        if room_temp < self._cold_threshold:
            s.cold_steps += 1
        if room_temp > self._overshoot_threshold:
            s.overshoot_steps += 1
        if applied_boost > 0.0:
            s.boost_applied_steps += 1
            s.boost_offset_sum_c += applied_boost
        if room.is_heating:
            wh = room.effective_heating_w * step_s / 3600.0
            s.total_heating_wh += wh
        s.service_call_count += service_calls_this_step
        s.adaptation_mode_counts[adaptation_mode] = (
            s.adaptation_mode_counts.get(adaptation_mode, 0) + 1
        )

        day = self._get_or_create_day(sim_time)
        if room_temp >= self._comfort_threshold:
            day.comfort_steps += 1
        if room_temp < self._cold_threshold:
            day.cold_steps += 1
        if room_temp > self._overshoot_threshold:
            day.overshoot_steps += 1
        if room.is_heating:
            day.heating_steps += 1
            day.total_heating_wh += room.effective_heating_w * step_s / 3600.0
        day.service_calls += service_calls_this_step
        if applied_boost > 0.0:
            day.boost_applied_steps += 1

        if len(self._trace) < self._max_trace:
            self._trace.append(_StepTrace(
                ts=sim_time.isoformat(),
                room_temp=round(room_temp, 2),
                commanded_setpoint=room.commanded_setpoint,
                outdoor_temp=outdoor_temp,
                trv_setpoint=zone.get("trv_setpoint"),
                effective_target=zone.get("effective_target"),
                boost_offset_c=boost_offset,
                applied_boost_offset_c=applied_boost,
                schedule_period=zone.get("schedule_period"),
                is_heating=room.is_heating,
            ))

    def summarize(self) -> MetricsSummary:
        return self._summary

    def last_trace(self) -> list[_StepTrace]:
        return list(self._trace)
