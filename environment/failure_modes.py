# environment/failure_modes.py
#
# Failure injection into the true environment state.
# Three failure modes for v1:
#   1. fuel_breach       — UAV runs out of fuel mid-mission
#   2. sensor_blackout   — sensor fails in a zone
#   3. weather_spike     — wind spikes beyond operational limit
#
# These are NOT random in the final sense — they are triggered by
# real causal conditions (underestimation, bad planning) plus a
# probabilistic trigger to keep it interesting across runs.
#
# Each failure returns a FailureEvent dataclass.

from dataclasses import dataclass, field
from typing import List, Optional
import random

from config.settings import (
    FUEL_CONSUMPTION_PER_KM,
    FUEL_HEADWIND_MULTIPLIER,
    WIND_FUEL_THRESHOLD_KMH,
    WEATHER_ABORT_WIND_KMH,
    WEATHER_SPIKE_PROBABILITY,
    WEATHER_SPIKE_GROWTH_PER_ITER,
    SENSOR_BLACKOUT_PROBABILITY,
    SENSOR_BLACKOUT_GROWTH_PER_ITER,
    WIND_ERROR_GROWTH_PER_ITER,
)


@dataclass
class FailureEvent:
    failure_type: str       # "fuel_breach" | "sensor_blackout" | "weather_spike"
    zone_id: Optional[str]  # which zone was affected (None for global failures)
    description: str        # human-readable cause
    caused_by: str          # root cause label for critic
    step_index: int         # which step in the execution this occurred


def check_fuel_breach(
    fuel_remaining: float,
    distance_km: float,
    wind_kmh: float,
    step_index: int,
) -> Optional[FailureEvent]:
    """
    Compute fuel consumed for a travel segment.
    Headwind multiplier applied if wind exceeds threshold.
    Returns a FailureEvent if UAV runs out of fuel.
    """
    multiplier = FUEL_HEADWIND_MULTIPLIER if wind_kmh > WIND_FUEL_THRESHOLD_KMH else 1.0
    fuel_needed = distance_km * FUEL_CONSUMPTION_PER_KM * multiplier

    if fuel_needed > fuel_remaining:
        return FailureEvent(
            failure_type="fuel_breach",
            zone_id=None,
            description=(
                f"Fuel exhausted mid-transit. "
                f"Required {fuel_needed:.3f} units, had {fuel_remaining:.3f}. "
                f"Wind={wind_kmh:.1f} km/h applied headwind multiplier={multiplier}."
            ),
            caused_by="underestimated_fuel_consumption",
            step_index=step_index,
        )
    return None


def check_sensor_blackout(
    zone_id: str,
    step_index: int,
    iteration: int,
) -> Optional[FailureEvent]:
    """
    Probabilistic sensor failure.
    Probability grows with iteration (worsening conditions).
    """
    prob = min(0.9, SENSOR_BLACKOUT_PROBABILITY + (iteration - 1) * SENSOR_BLACKOUT_GROWTH_PER_ITER)
    if random.random() < prob:
        return FailureEvent(
            failure_type="sensor_blackout",
            zone_id=zone_id,
            description=(
                f"Sensor blackout in zone {zone_id}. "
                f"Scan data lost. Iteration {iteration} degraded sensor reliability."
            ),
            caused_by="sensor_degradation",
            step_index=step_index,
        )
    return None


def check_weather_spike(
    current_wind_kmh: float,
    step_index: int,
    iteration: int,
) -> tuple[float, Optional[FailureEvent]]:
    """
    Probabilistic weather spike.
    Spike magnitude grows with iteration.
    Returns (new_wind, FailureEvent or None).
    """
    prob = min(0.8, WEATHER_SPIKE_PROBABILITY + (iteration - 1) * WEATHER_SPIKE_GROWTH_PER_ITER)
    if random.random() < prob:
        spike = random.uniform(
            WIND_ERROR_GROWTH_PER_ITER * iteration,
            WIND_ERROR_GROWTH_PER_ITER * iteration * 2.5,
        )
        new_wind = current_wind_kmh + spike

        if new_wind > WEATHER_ABORT_WIND_KMH:
            return new_wind, FailureEvent(
                failure_type="weather_spike",
                zone_id=None,
                description=(
                    f"Wind spiked from {current_wind_kmh:.1f} to {new_wind:.1f} km/h. "
                    f"Exceeds abort threshold of {WEATHER_ABORT_WIND_KMH} km/h."
                ),
                caused_by="unmodeled_weather_dynamics",
                step_index=step_index,
            )
        return new_wind, None  # spike but not abort-level
    return current_wind_kmh, None