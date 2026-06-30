"""VirtualRoomPhysics — discrete thermal model for the S1 simulation harness.

Physical model (per step dt_s seconds):
    Q_in  = effective_heating_W + afterheat_W + internal_gain_W + solar_gain_W
    Q_out = heat_loss_w_per_k × (T_room - T_outdoor)   [window multiplies loss]
    dT    = (Q_in - Q_out) × dt_s / (thermal_mass_kj_per_k × 1000)

The TRV thermostat switches heating ON when commanded_setpoint > room_temp + deadband.
Supply delay is modeled as a linear warm-up ramp over supply_delay_s seconds.
Afterheat decays exponentially with time constant afterheat_tau_s after switch-off.
Window-open multiplies heat_loss_w_per_k by window_loss_factor (default 5×).

Named profiles for Item 6 scenarios:
    PROFILE_FAST_INSULATED  — quick-response, well-insulated room (Profile A)
    PROFILE_SLOW_INERTIA    — heavy thermal mass, long supply delay, significant afterheat (Profile B)
    PROFILE_DIFFICULT       — high heat loss, variable external temp sensitivity, noisy sensor (Profile C)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


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
    window_loss_factor: float = 5.0   # multiplier on heat_loss when window is open
    name: str = "standard"            # human-readable identifier for test reports


# ── Named profiles for Item 6 ─────────────────────────────────────────────────

# Profile A: Fast, well-insulated room
# Low thermal mass → heats quickly; low heat loss → stays warm; short supply delay;
# minimal afterheat.  Closest to a modern energy-efficient room with small radiator.
PROFILE_FAST_INSULATED = RoomProfile(
    name="fast_insulated",
    thermal_mass_kj_per_k=2500.0,   # lighter construction
    heat_loss_w_per_k=30.0,         # well-insulated, ~20 m² modern
    heating_power_w=900.0,          # modest radiator
    supply_delay_s=180.0,           # 3-min warm-up
    afterheat_tau_s=300.0,          # minor coasting
    sensor_noise_k=0.03,
    deadband_k=0.1,
    internal_gain_w=40.0,
)

# Profile B: Slow, high-inertia room
# Heavy mass → slow response; long supply delay; significant afterheat → overshoot risk.
# Typical of old building with large cast-iron radiator.
PROFILE_SLOW_INERTIA = RoomProfile(
    name="slow_inertia",
    thermal_mass_kj_per_k=7000.0,   # heavy construction + furnishings
    heat_loss_w_per_k=55.0,         # moderate insulation
    heating_power_w=2000.0,         # large radiator
    supply_delay_s=900.0,           # 15-min warm-up (boiler start)
    afterheat_tau_s=1800.0,         # 30-min afterheat decay
    sensor_noise_k=0.05,
    deadband_k=0.15,
    internal_gain_w=25.0,
)

# Profile C: Difficult room (high heat loss, variable, noisy)
# Used to test robustness: confounded episodes, window events, sensor noise, availability gaps.
# heating_power_w=3000 W is required to reach the 20°C setpoint at the coldest winter point
# (outdoor nadir ~-6°C at base_winter_c=-3°C): T_max = -6 + 3000/90 ≈ 27°C.
# With 1500 W the equilibrium would be only ~10°C — thermodynamically impossible to achieve
# setpoint — so this is 3000 W to make the scenario "difficult but achievable".
PROFILE_DIFFICULT = RoomProfile(
    name="difficult",
    thermal_mass_kj_per_k=3500.0,   # medium mass
    heat_loss_w_per_k=90.0,         # poorly insulated, north-facing
    heating_power_w=3000.0,         # strong heater required to overcome high heat loss
    supply_delay_s=600.0,           # 10-min warm-up
    afterheat_tau_s=900.0,          # 15-min afterheat
    sensor_noise_k=0.12,            # significant sensor noise
    deadband_k=0.2,
    internal_gain_w=20.0,
)

# Profile D: Clean afterheat scenario — strong heater, low noise, long afterheat decay.
# Designed for spring conditions (outdoor ≈ 5°C → ΔT ≈ 16.5K):
#   At cut-off: Q_in = 2500W afterheat + 20W internal = 2520W;
#               Q_out = 70 × 16.5 = 1155W; net = +1365W → slope ≈ +0.016°C/min >> threshold
#   Post-peak cooling: Q_out − Q_in = 1155 − 20 = 1135W → slope ≈ −0.014°C/min > 0.01
#   → PASSIVE_COOLING regime triggers → AfterheatEpisodeBuilder closes via CLOSED_PEAK_COOLING
PROFILE_CLEAN_AFTERHEAT = RoomProfile(
    name="clean_afterheat",
    thermal_mass_kj_per_k=5000.0,
    heat_loss_w_per_k=70.0,
    heating_power_w=2500.0,
    supply_delay_s=600.0,
    afterheat_tau_s=2400.0,
    sensor_noise_k=0.01,
    deadband_k=0.1,
    internal_gain_w=20.0,
)

# Profile E: Clean heat-loss scenario — high loss, low noise, no windows.
# At night setback (room=21→10°C, outdoor=-5°C, ΔT=26K at start):
#   Cooling rate = 80 × 26 / 3_000_000 × 60 ≈ 0.041°C/min >> 0.01 threshold
PROFILE_CLEAN_HEAT_LOSS = RoomProfile(
    name="clean_heat_loss",
    thermal_mass_kj_per_k=3000.0,
    heat_loss_w_per_k=80.0,
    heating_power_w=2500.0,
    supply_delay_s=300.0,
    afterheat_tau_s=600.0,
    sensor_noise_k=0.02,
    deadband_k=0.1,
    internal_gain_w=20.0,
)

# Profile F: Clean supply-delay scenario — heavy mass, long ramp, very clean sensor.
# Supply delay = 1200s (20 min) → onset signal at step ~5 (25 min from command)
PROFILE_CLEAN_SUPPLY_DELAY = RoomProfile(
    name="clean_supply_delay",
    thermal_mass_kj_per_k=8000.0,
    heat_loss_w_per_k=50.0,
    heating_power_w=2000.0,
    supply_delay_s=1200.0,
    afterheat_tau_s=600.0,
    sensor_noise_k=0.02,
    deadband_k=0.1,
    internal_gain_w=20.0,
)


class VirtualRoomPhysics:
    """Discrete thermal model of a single heated room."""

    def __init__(self, profile: RoomProfile, initial_temp: float = 18.0) -> None:
        self.profile = profile
        self.room_temp: float = initial_temp
        self._commanded: float = initial_temp
        self._heating_on: bool = False
        self._warmup_elapsed: float = 0.0
        self._afterheat_w: float = 0.0
        self._window_open: bool = False

    # ── Window control ───────────────────────────────────────────────────

    def set_window_open(self, is_open: bool) -> None:
        """Open or close the window, adjusting effective heat loss."""
        self._window_open = is_open

    @property
    def window_open(self) -> bool:
        return self._window_open

    # ── Ground-truth physics accessors ──────────────────────────────────

    @property
    def ground_truth_heat_rate_c_per_h(self) -> float:
        """Net temperature rise rate at reference condition (room 18 °C, outdoor 0 °C).

        The HeatRateModel learns the *observed* net rate during real heating episodes,
        which equals (heating_power + internal_gain - heat_loss × ΔT) / thermal_mass.
        Comparing against the gross rate (heating_power only) would be apples-to-oranges:
        net < gross by the heat-loss fraction.

        Reference: room_temp=18 °C, outdoor_temp=0 °C → ΔT=18 K.
        This matches the typical early-morning preheat start in a winter simulation
        and is confirmed by Profile C's learned value (≈ 1.44 °C/h) converging to
        exactly this reference within 2 % already after 90 days.

        Values by profile (approximate):
          Profile A (fast_insulated): ≈ 0.58 °C/h
          Profile B (slow_inertia):   ≈ 0.53 °C/h
          Profile C (difficult):      ≈ 1.44 °C/h
        """
        p = self.profile
        _ref_delta_k = 18.0  # room=18 °C minus outdoor=0 °C
        net_w = p.heating_power_w + p.internal_gain_w - p.heat_loss_w_per_k * _ref_delta_k
        net_w = max(net_w, 0.0)
        return (net_w / (p.thermal_mass_kj_per_k * 1000.0)) * 3600.0

    @property
    def ground_truth_heat_loss_w_per_k(self) -> float:
        """Reference heat loss coefficient — matches model's `heat_loss_w_per_k`."""
        return self.profile.heat_loss_w_per_k

    @property
    def ground_truth_supply_delay_s(self) -> float:
        return self.profile.supply_delay_s

    @property
    def ground_truth_afterheat_tau_s(self) -> float:
        return self.profile.afterheat_tau_s

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
        # Window open: effective heat loss multiplied by window_loss_factor
        effective_loss_w_per_k = (p.heat_loss_w_per_k * p.window_loss_factor
                                  if self._window_open else p.heat_loss_w_per_k)
        Q_out = effective_loss_w_per_k * (self.room_temp - outdoor_temp)
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
