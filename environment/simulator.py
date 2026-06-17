# environment/simulator.py
#
# The Executor — pure rule-based Python, NO LLM.
# Takes a plan (list of actions) and a TrueEnvironmentState,
# simulates execution step by step, and returns:
#   - updated TrueEnvironmentState
#   - execution log (list of step dicts)
#   - list of FailureEvents
#   - bool: mission_succeeded
#
# The Executor is the only component that sees TrueEnvironmentState.

import copy
from typing import List, Tuple, Dict, Any

from environment.state import TrueEnvironmentState, ZoneState
from environment.failure_modes import (
    FailureEvent,
    check_fuel_breach,
    check_sensor_blackout,
    check_weather_spike,
)
from config.settings import (
    FUEL_CONSUMPTION_PER_KM,
    FUEL_HEADWIND_MULTIPLIER,
    WIND_FUEL_THRESHOLD_KMH,
    WEATHER_ABORT_WIND_KMH,
)


def execute_plan(
    plan: dict,
    true_state: TrueEnvironmentState,
    iteration: int,
) -> Tuple[TrueEnvironmentState, List[Dict[str, Any]], List[FailureEvent], bool]:
    """
    Execute a plan against the true environment state.

    Args:
        plan: structured plan dict from Planner Agent
        true_state: ground truth environment (mutated in place copy)
        iteration: current replan iteration (1-indexed)

    Returns:
        (updated_true_state, execution_log, failures, mission_succeeded)
    """
    state = copy.deepcopy(true_state)
    steps = plan.get("steps", [])
    execution_log: List[Dict[str, Any]] = []
    failures: List[FailureEvent] = []
    mission_succeeded = False

    for i, step in enumerate(steps):
        action = step.get("action", "")
        step_log: Dict[str, Any] = {
            "step_index": i,
            "action": action,
            "state_before": {
                "fuel": round(state.fuel_remaining, 3),
                "wind_kmh": round(state.wind_kmh, 2),
                "position": state.uav_position,
            },
            "outcome": "pending",
            "failures": [],
        }

        # ── Weather check (before every step) ──────────────────────────
        new_wind, weather_event = check_weather_spike(
            current_wind_kmh=state.wind_kmh,
            step_index=i,
            iteration=iteration,
        )
        state.wind_kmh = new_wind
        if weather_event:
            failures.append(weather_event)
            step_log["failures"].append(weather_event.__dict__)
            step_log["outcome"] = "aborted_weather"
            step_log["state_after"] = _state_snapshot(state)
            execution_log.append(step_log)
            # Mission aborted — wind too high
            break

        # ── Action dispatch ──────────────────────────────────────────────
        if action == "takeoff":
            step_log["outcome"] = _handle_takeoff(state)

        elif action == "navigate":
            zone_id = step.get("zone_id", "")
            fuel_event = _handle_navigate(state, zone_id, i)
            if fuel_event:
                failures.append(fuel_event)
                step_log["failures"].append(fuel_event.__dict__)
                step_log["outcome"] = "failed_fuel_breach"
                step_log["state_after"] = _state_snapshot(state)
                execution_log.append(step_log)
                break  # UAV is grounded
            step_log["outcome"] = f"navigated_to_{zone_id}"

        elif action == "scan":
            zone_id = step.get("zone_id", "")
            sensor_event = _handle_scan(state, zone_id, i, iteration)
            if sensor_event:
                failures.append(sensor_event)
                step_log["failures"].append(sensor_event.__dict__)
                step_log["outcome"] = f"scan_failed_blackout_{zone_id}"
            else:
                step_log["outcome"] = f"scan_success_{zone_id}"

        elif action == "return_to_base":
            fuel_event = _handle_return(state, i)
            if fuel_event:
                failures.append(fuel_event)
                step_log["failures"].append(fuel_event.__dict__)
                step_log["outcome"] = "failed_fuel_breach_return"
                step_log["state_after"] = _state_snapshot(state)
                execution_log.append(step_log)
                break
            step_log["outcome"] = "returned_to_base"

        elif action == "land":
            step_log["outcome"] = "landed"
            state.uav_position = "base"
            # Check if all required zones were scanned
            all_scanned = all(
                z.scan_completed
                for z in state.zones.values()
                if z.scan_required
            )
            mission_succeeded = all_scanned and len(failures) == 0
            step_log["mission_complete"] = mission_succeeded

        else:
            step_log["outcome"] = f"unknown_action_{action}"

        state.mission_elapsed_steps += 1
        step_log["state_after"] = _state_snapshot(state)
        execution_log.append(step_log)

    return state, execution_log, failures, mission_succeeded


# ── Private helpers ──────────────────────────────────────────────────────────

def _state_snapshot(state: TrueEnvironmentState) -> dict:
    return {
        "fuel": round(state.fuel_remaining, 3),
        "wind_kmh": round(state.wind_kmh, 2),
        "position": state.uav_position,
        "zones_scanned": [
            zid for zid, z in state.zones.items() if z.scan_completed
        ],
    }


def _handle_takeoff(state: TrueEnvironmentState) -> str:
    state.uav_position = "airborne"
    # Small fuel cost for takeoff
    state.fuel_remaining = max(0.0, state.fuel_remaining - 0.02)
    return "takeoff_success"


def _handle_navigate(
    state: TrueEnvironmentState,
    zone_id: str,
    step_index: int,
) -> "FailureEvent | None":
    zone = state.zones.get(zone_id)
    if not zone:
        # Unknown zone — treat as 1km hop
        distance = 1.0
    else:
        distance = zone.distance_km

    fuel_event = check_fuel_breach(
        fuel_remaining=state.fuel_remaining,
        distance_km=distance,
        wind_kmh=state.wind_kmh,
        step_index=step_index,
    )
    if fuel_event:
        state.fuel_remaining = 0.0
        return fuel_event

    # Consume fuel
    multiplier = FUEL_HEADWIND_MULTIPLIER if state.wind_kmh > WIND_FUEL_THRESHOLD_KMH else 1.0
    consumed = distance * FUEL_CONSUMPTION_PER_KM * multiplier
    state.fuel_remaining = max(0.0, state.fuel_remaining - consumed)
    state.uav_position = zone_id
    return None


def _handle_scan(
    state: TrueEnvironmentState,
    zone_id: str,
    step_index: int,
    iteration: int,
) -> "FailureEvent | None":
    zone = state.zones.get(zone_id)
    if not zone:
        return None

    sensor_event = check_sensor_blackout(zone_id, step_index, iteration)
    if sensor_event:
        zone.sensor_blackout = True
        return sensor_event

    zone.scan_completed = True
    return None


def _handle_return(
    state: TrueEnvironmentState,
    step_index: int,
) -> "FailureEvent | None":
    """Return to base — estimate distance as avg of all zone distances."""
    if not state.zones:
        distance = 1.0
    else:
        distance = sum(z.distance_km for z in state.zones.values()) / len(state.zones)

    fuel_event = check_fuel_breach(
        fuel_remaining=state.fuel_remaining,
        distance_km=distance,
        wind_kmh=state.wind_kmh,
        step_index=step_index,
    )
    if fuel_event:
        state.fuel_remaining = 0.0
        return fuel_event

    multiplier = FUEL_HEADWIND_MULTIPLIER if state.wind_kmh > WIND_FUEL_THRESHOLD_KMH else 1.0
    consumed = distance * FUEL_CONSUMPTION_PER_KM * multiplier
    state.fuel_remaining = max(0.0, state.fuel_remaining - consumed)
    state.uav_position = "base"
    return None