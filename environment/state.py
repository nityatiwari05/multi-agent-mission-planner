# environment/state.py
#
# Two separate state objects:
#   TrueEnvironmentState  — ground truth, only the Executor sees this
#   ObservedEnvironmentState — noisy/partial view, what the Planner gets
#
# The gap between them is the core of the system's intelligence signal.
# The gap grows each iteration (dynamic, not fixed).
#
# CHANGE: added MismatchRecord + build_mismatch_record(). Previously the
# Planner only ever saw a single fresh ObservedEnvironmentState each
# iteration, with no memory of how wrong the last observation turned out
# to be. That's WHY the Planner looked like it "wasn't learning" across
# iterations (underestimating wind by +5, then +9, then +13, in the same
# direction every time) — it never had the prior error available to react
# to. The Coordinator now builds one of these after every iteration (using
# numbers the Critic already computes) and threads the growing list
# forward into the next generate_plan() call.

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import random
import copy

from config.settings import (
    BASE_WIND_OBSERVATION_ERROR_KMH,
    WIND_ERROR_GROWTH_PER_ITER,
    BASE_FUEL_OBSERVATION_ERROR,
    FUEL_ERROR_GROWTH_PER_ITER,
    SENSOR_NOISE_STD,
)


@dataclass
class ZoneState:
    """State of a single scan zone."""
    zone_id: str
    distance_km: float          # distance from base to this zone
    scan_required: bool = True
    scan_completed: bool = False
    sensor_blackout: bool = False   # true state only


@dataclass
class TrueEnvironmentState:
    """
    Ground truth environment state.
    Only the Executor (simulator) has access to this.
    """
    wind_kmh: float                         # true wind speed
    fuel_remaining: float                   # true fuel (0.0 – 1.0)
    zones: Dict[str, ZoneState] = field(default_factory=dict)
    uav_position: str = "base"              # current zone id or "base"
    mission_elapsed_steps: int = 0
    active_failures: List[str] = field(default_factory=list)  # e.g. ["sensor_blackout:ZoneB"]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class ObservedEnvironmentState:
    """
    What the Planner Agent sees — noisy, partial, potentially wrong.
    Generated from TrueEnvironmentState with added observation error.
    """
    wind_kmh: float             # underestimated wind
    fuel_remaining: float       # overestimated fuel
    zones: Dict[str, dict]      # zones without blackout info
    uav_position: str
    observation_iteration: int  # which replan iteration this was generated at

    def to_dict(self) -> dict:
        return {
            "wind_kmh": round(self.wind_kmh, 2),
            "fuel_remaining": round(self.fuel_remaining, 3),
            "zones": self.zones,
            "uav_position": self.uav_position,
            "observation_iteration": self.observation_iteration,
        }


@dataclass
class MismatchRecord:
    """
    One iteration's worth of observed-vs-true error, captured AFTER
    execution (when the true value is finally knowable for post-incident
    review). This is what gives the Planner an actual trend to react to,
    instead of re-deriving the same "wind looks fine" belief from a fresh
    noisy snapshot every time.
    """
    iteration: int
    wind_observed: float
    wind_true: float
    fuel_observed: float
    fuel_true: float

    @property
    def wind_underestimate(self) -> float:
        return round(self.wind_true - self.wind_observed, 2)

    @property
    def fuel_overestimate(self) -> float:
        return round(self.fuel_observed - self.fuel_true, 3)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "wind_underestimate_kmh": self.wind_underestimate,
            "fuel_overestimate_fraction": self.fuel_overestimate,
        }


def build_mismatch_record(
    iteration: int,
    observed_state: "ObservedEnvironmentState",
    true_state: "TrueEnvironmentState",
) -> MismatchRecord:
    return MismatchRecord(
        iteration=iteration,
        wind_observed=observed_state.wind_kmh,
        wind_true=true_state.wind_kmh,
        fuel_observed=observed_state.fuel_remaining,
        fuel_true=true_state.fuel_remaining,
    )


def summarize_mismatch_trend(history: List[MismatchRecord]) -> dict:
    """
    Turns a list of per-iteration MismatchRecords into an explicit trend
    summary: is the wind underestimate growing, shrinking, or flat across
    iterations? Same for the fuel overestimate. Computed in code (not left
    for the model to eyeball a list of numbers and guess), and handed to
    the Planner as a short, unambiguous instruction-relevant fact.

    Returns {} if there isn't at least 2 points to compare.
    """
    if len(history) < 2:
        return {}

    wind_errors = [r.wind_underestimate for r in history]
    fuel_errors = [r.fuel_overestimate for r in history]

    def trend(values: List[float]) -> str:
        # Strictly increasing across the whole history = "worsening".
        # Strictly decreasing = "improving". Anything else = "flat".
        if all(b > a for a, b in zip(values, values[1:])):
            return "worsening"
        if all(b < a for a, b in zip(values, values[1:])):
            return "improving"
        return "flat"

    return {
        "wind_underestimate_trend": trend(wind_errors),
        "wind_underestimate_by_iteration": wind_errors,
        "fuel_overestimate_trend": trend(fuel_errors),
        "fuel_overestimate_by_iteration": fuel_errors,
        "consecutive_iterations_observed": len(history),
    }


def generate_observed_state(
    true_state: TrueEnvironmentState,
    iteration: int,
) -> ObservedEnvironmentState:
    """
    Derive the observed state from true state.
    Observation error grows with each replan iteration (dynamic gap).

    iteration=1 → small error
    iteration=2 → medium error
    iteration=3 → large error (conditions actively worsening)
    """
    wind_error = BASE_WIND_OBSERVATION_ERROR_KMH +  random.uniform(-2.0, 2.0)
    fuel_error = BASE_FUEL_OBSERVATION_ERROR + random.uniform(-2.0, 2.0)

    observed_wind = max(0.0, true_state.wind_kmh - wind_error)

    # Planner over-estimates fuel (thinks it has more than it does)
    observed_fuel = min(1.0, true_state.fuel_remaining + fuel_error)

    # Zones: strip blackout info, add sensor noise to distance
    observed_zones = {}
    for zid, z in true_state.zones.items():
        noise = random.gauss(0, SENSOR_NOISE_STD)
        observed_zones[zid] = {
            "zone_id": zid,
            "distance_km": round(max(0.1, z.distance_km + noise), 2),
            "scan_required": z.scan_required,
            "scan_completed": z.scan_completed,
            # blackout is NOT visible to planner
        }

    return ObservedEnvironmentState(
        wind_kmh=round(observed_wind, 2),
        fuel_remaining=round(observed_fuel, 3),
        zones=observed_zones,
        uav_position=true_state.uav_position,
        observation_iteration=iteration,
    )


def build_initial_true_state(mission: dict) -> TrueEnvironmentState:
    """
    Build the initial TrueEnvironmentState from the mission JSON input.
    Adds randomized true values that are worse than what the mission declares.
    """
    constraints = mission.get("constraints", {})
    zones_input = mission.get("zones", [])

    # True wind is always higher than what the mission declares
    declared_wind = constraints.get("wind_kmh", 10)
    true_wind = declared_wind + random.uniform(
        BASE_WIND_OBSERVATION_ERROR_KMH,
        BASE_WIND_OBSERVATION_ERROR_KMH + WIND_ERROR_GROWTH_PER_ITER
    )

    # True fuel is always lower than declared
    declared_fuel = constraints.get("fuel_level", 1.0)
    true_fuel = max(0.1, declared_fuel - random.uniform(
        BASE_FUEL_OBSERVATION_ERROR,
        BASE_FUEL_OBSERVATION_ERROR + FUEL_ERROR_GROWTH_PER_ITER
    ))

    zones = {}
    for z in zones_input:
        zid = z["zone_id"]
        zones[zid] = ZoneState(
            zone_id=zid,
            distance_km=z["distance_km"],
            scan_required=z.get("scan_required", True),
        )

    return TrueEnvironmentState(
        wind_kmh=round(true_wind, 2),
        fuel_remaining=round(true_fuel, 3),
        zones=zones,
    )