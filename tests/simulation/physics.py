"""VirtualRoomPhysics — discrete thermal model for the S1 simulation harness.

Physical model (per step dt_s seconds):
    Q_in  = effective_heating_W + afterheat_W + internal_gain_W + solar_gain_W
    Q_out = heat_loss_w_per_k × (T_room - T_outdoor)
    dT    = (Q_in - Q_out) × dt_s / (thermal_mass_kj_per_k × 1000)

The TRV thermostat switches heating ON when commanded_setpoint > room_temp + deadband.
Supply delay is modeled as a linear warm-up ramp over supply_delay_s seconds.
Afterheat decays exponentially with time constant afterheat_tau_s after switch-off.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class RoomProfile:
    thermal_mass_kj_per_k: float = 4000.0
    heat_loss_w_per_k: float = 70.0
    heating_power_w: float = 1200.0
    supply_delay_s: float = 300.0
    afterheat_tau_s: float = 600.0
    sensor_noise_k: float = 0.05
    trv_step: float = 0.5
    device_min: float = 5.0
    device_max: float = 35.0
    deadband_k: float = 0.1
    internal_gain_w: float = 30.0


class VirtualRoomPhysics:
    """Discrete thermal model of a single heated room."""

    def __init__(self, profile: RoomProfile, initial_temp: float = 18.0) -> None:
        self.profile = profile
        self.room_temp: float = initial_temp
        self._commanded: float = initial_temp
        self._heating_on: bool = False
        self._warmup_elapsed: float = 0.0
        self._afterheat_w: float = 0.0

    # ── Setpoint command ────────────────────────────────────────────────

    def set_setpoint(self, setpoint: float) -> None:
        """Apply a new commanded setpoint from coordinator dispatch."""
        p = self.profile
        clamped = max(p.device_min, min(p.device_max, setpoint))
        self._commanded = round(clamped / p.trv_step) * p.trv_step

    @property
    def commanded_setpoint(self) -> float:
        return self._commanded

    # ── Physics step ────────────────────────────────────────────────────

    def step(self, dt_s: float, outdoor_temp: float, solar_gain_w: float = 0.0) -> None:
        """Advance thermal model by dt_s seconds."""
        p = self.profile
        should_heat = self._commanded > self.room_temp + p.deadband_k

        # State transitions
        if should_heat and not self._heating_on:
            self._heating_on = True
            self._warmup_elapsed = 0.0
            self._afterheat_w = 0.0
        elif not should_heat and self._heating_on:
            fraction = min(1.0, self._warmup_elapsed / max(1.0, p.supply_delay_s))
            self._afterheat_w = p.heating_power_w * fraction
            self._heating_on = False

        # Effective heat input
        if self._heating_on:
            self._warmup_elapsed += dt_s
            fraction = min(1.0, self._warmup_elapsed / max(1.0, p.supply_delay_s))
            effective_w = p.heating_power_w * fraction
        else:
            effective_w = 0.0
            if self._afterheat_w > 0.5:
                self._afterheat_w *= math.exp(-dt_s / max(1.0, p.afterheat_tau_s))
            else:
                self._afterheat_w = 0.0

        Q_in = effective_w + self._afterheat_w + p.internal_gain_w + solar_gain_w
        Q_out = p.heat_loss_w_per_k * (self.room_temp - outdoor_temp)
        dT = (Q_in - Q_out) * dt_s / (p.thermal_mass_kj_per_k * 1000.0)
        self.room_temp = round(self.room_temp + dT, 4)

    # ── Sensor ──────────────────────────────────────────────────────────

    def read_sensor(self, noise: bool = True) -> float:
        """Return room temperature as a sensor reading (0.1 °C quantized)."""
        val = self.room_temp
        if noise and self.profile.sensor_noise_k > 0:
            val += random.gauss(0, self.profile.sensor_noise_k)
        return round(val * 10) / 10

    @property
    def is_heating(self) -> bool:
        return self._heating_on

    @property
    def effective_heating_w(self) -> float:
        if not self._heating_on:
            return 0.0
        fraction = min(1.0, self._warmup_elapsed / max(1.0, self.profile.supply_delay_s))
        return self.profile.heating_power_w * fraction
