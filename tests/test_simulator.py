# tests/test_simulator.py
#
# Unit tests for environment/simulator.py (the Executor).
#
# Weather and sensor checks are probabilistic by design (see
# environment/failure_modes.py), so most tests here monkeypatch
# environment.simulator.check_weather_spike / check_sensor_blackout
# directly, rather than fighting with global random state via seeds.
# That keeps these tests decoupled from whatever the current probability
# constants happen to be tuned to (they were just retuned once already -
# these tests shouldn't need to change again if that happens a second
# time), and isolates exactly the orchestration logic of the Executor
# itself: fuel accounting, abort conditions, and the success criteria.
#
# Fuel math (check_fuel_breach) is deterministic, so those tests exercise
# the real function rather than mocking it.

import sys
from pathlib import Path

# Add the project root to sys.path so local modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import environment.simulator as sim
from config.settings import FUEL_CONSUMPTION_PER_KM
from environment.failure_modes import FailureEvent
from environment.state import TrueEnvironmentState, ZoneState


def make_true_state(wind_kmh=10.0, fuel_remaining=1.0, zones=None):
    zones = zones if zones is not None else {
        "ZoneA": ZoneState(zone_id="ZoneA", distance_km=3, scan_required=True),
        "ZoneB": ZoneState(zone_id="ZoneB", distance_km=4, scan_required=True),
    }
    return TrueEnvironmentState(wind_kmh=wind_kmh, fuel_remaining=fuel_remaining, zones=zones)


SIMPLE_PLAN = {
    "steps": [
        {"action": "takeoff"},
        {"action": "navigate", "zone_id": "ZoneA"},
        {"action": "scan", "zone_id": "ZoneA"},
        {"action": "navigate", "zone_id": "ZoneB"},
        {"action": "scan", "zone_id": "ZoneB"},
        {"action": "return_to_base"},
        {"action": "land"},
    ]
}


def _no_weather(current_wind_kmh, step_index, iteration):
    """Stand-in for check_weather_spike that never triggers a spike."""
    return current_wind_kmh, None


def _no_sensor(zone_id, step_index, iteration):
    """Stand-in for check_sensor_blackout that never triggers a blackout."""
    return None


def _neutralize_weather_and_sensor(monkeypatch):
    monkeypatch.setattr(sim, "check_weather_spike", _no_weather)
    monkeypatch.setattr(sim, "check_sensor_blackout", _no_sensor)


def test_clean_mission_with_enough_fuel_succeeds(monkeypatch):
    _neutralize_weather_and_sensor(monkeypatch)
    true_state = make_true_state(wind_kmh=10.0, fuel_remaining=1.0)

    updated, log, failures, succeeded = sim.execute_plan(SIMPLE_PLAN, true_state, iteration=1)

    assert succeeded is True
    assert failures == []
    assert updated.zones["ZoneA"].scan_completed is True
    assert updated.zones["ZoneB"].scan_completed is True
    assert len(log) == len(SIMPLE_PLAN["steps"])


def test_insufficient_fuel_triggers_breach_and_grounds_the_mission(monkeypatch):
    _neutralize_weather_and_sensor(monkeypatch)
    true_state = make_true_state(wind_kmh=10.0, fuel_remaining=0.01)

    updated, log, failures, succeeded = sim.execute_plan(SIMPLE_PLAN, true_state, iteration=1)

    assert succeeded is False
    assert any(f.failure_type == "fuel_breach" for f in failures)
    assert len(log) < len(SIMPLE_PLAN["steps"])  # aborted partway, not all steps attempted


def test_headwind_increases_real_fuel_consumption(monkeypatch):
    _neutralize_weather_and_sensor(monkeypatch)
    calm = make_true_state(wind_kmh=5.0, fuel_remaining=1.0)
    windy = make_true_state(wind_kmh=20.0, fuel_remaining=1.0)

    updated_calm, *_ = sim.execute_plan(SIMPLE_PLAN, calm, iteration=1)
    updated_windy, *_ = sim.execute_plan(SIMPLE_PLAN, windy, iteration=1)

    fuel_used_calm = 1.0 - updated_calm.fuel_remaining
    fuel_used_windy = 1.0 - updated_windy.fuel_remaining
    assert fuel_used_windy > fuel_used_calm


def test_fuel_breach_can_occur_specifically_on_the_return_leg(monkeypatch):
    _neutralize_weather_and_sensor(monkeypatch)
    # Comfortable margin to reach both zones and scan them (needs ~0.30),
    # but not enough left over to fly home afterward (needs ~0.44 total).
    true_state = make_true_state(wind_kmh=5.0, fuel_remaining=0.35)

    updated, log, failures, succeeded = sim.execute_plan(SIMPLE_PLAN, true_state, iteration=1)

    assert succeeded is False
    assert any(f.failure_type == "fuel_breach" for f in failures)
    return_step = next((s for s in log if s["action"] == "return_to_base"), None)
    assert return_step is not None
    assert return_step["outcome"] == "failed_fuel_breach_return"


def test_sensor_blackout_blocks_required_zone_but_does_not_ground_the_uav(monkeypatch):
    monkeypatch.setattr(sim, "check_weather_spike", _no_weather)
    monkeypatch.setattr(
        sim, "check_sensor_blackout",
        lambda zone_id, step_index, iteration: FailureEvent(
            failure_type="sensor_blackout", zone_id=zone_id,
            description="forced for test", caused_by="test", step_index=step_index,
        ),
    )
    true_state = make_true_state(wind_kmh=10.0, fuel_remaining=1.0)

    updated, log, failures, succeeded = sim.execute_plan(SIMPLE_PLAN, true_state, iteration=1)

    assert succeeded is False
    assert any(f.failure_type == "sensor_blackout" for f in failures)
    assert updated.zones["ZoneA"].scan_completed is False
    # a sensor blackout is "soft" - the mission keeps flying and still lands
    assert updated.uav_position == "base"
    assert len(log) == len(SIMPLE_PLAN["steps"])


def test_weather_spike_aborts_immediately_before_any_action_resolves(monkeypatch):
    monkeypatch.setattr(
        sim, "check_weather_spike",
        lambda current_wind_kmh, step_index, iteration: (
            999.0,
            FailureEvent(
                failure_type="weather_spike", zone_id=None,
                description="forced for test", caused_by="test", step_index=step_index,
            ),
        ),
    )
    true_state = make_true_state(wind_kmh=10.0, fuel_remaining=1.0)

    updated, log, failures, succeeded = sim.execute_plan(SIMPLE_PLAN, true_state, iteration=1)

    assert succeeded is False
    assert any(f.failure_type == "weather_spike" for f in failures)
    assert len(log) == 1  # aborted on the very first step, before takeoff even resolved
    assert log[0]["outcome"] == "aborted_weather"


def test_required_zone_never_visited_fails_mission_with_no_explicit_failure_event(monkeypatch):
    _neutralize_weather_and_sensor(monkeypatch)
    plan_missing_zone_b = {
        "steps": [
            {"action": "takeoff"},
            {"action": "navigate", "zone_id": "ZoneA"},
            {"action": "scan", "zone_id": "ZoneA"},
            {"action": "return_to_base"},
            {"action": "land"},
        ]
    }
    true_state = make_true_state(wind_kmh=10.0, fuel_remaining=1.0)

    updated, log, failures, succeeded = sim.execute_plan(plan_missing_zone_b, true_state, iteration=1)

    assert succeeded is False
    assert failures == []  # nothing "broke" environmentally - ZoneB was simply never attempted
    assert updated.zones["ZoneB"].scan_completed is False


def test_optional_zone_can_be_skipped_without_failing_the_mission(monkeypatch):
    _neutralize_weather_and_sensor(monkeypatch)
    zones = {
        "ZoneA": ZoneState(zone_id="ZoneA", distance_km=3, scan_required=True),
        "ZoneB": ZoneState(zone_id="ZoneB", distance_km=10, scan_required=False),
    }
    true_state = make_true_state(wind_kmh=10.0, fuel_remaining=1.0, zones=zones)
    plan_required_only = {
        "steps": [
            {"action": "takeoff"},
            {"action": "navigate", "zone_id": "ZoneA"},
            {"action": "scan", "zone_id": "ZoneA"},
            {"action": "return_to_base"},
            {"action": "land"},
        ]
    }

    updated, log, failures, succeeded = sim.execute_plan(plan_required_only, true_state, iteration=1)

    assert succeeded is True
    assert updated.zones["ZoneB"].scan_completed is False  # optional, never attempted, and that's fine


def test_return_to_base_cost_uses_average_of_all_zone_distances(monkeypatch):
    _neutralize_weather_and_sensor(monkeypatch)
    zones = {
        "Near": ZoneState(zone_id="Near", distance_km=1, scan_required=True),
        "Far": ZoneState(zone_id="Far", distance_km=9, scan_required=False),
    }
    true_state = make_true_state(wind_kmh=5.0, fuel_remaining=1.0, zones=zones)  # calm, no headwind multiplier
    plan = {
        "steps": [
            {"action": "takeoff"},
            {"action": "navigate", "zone_id": "Near"},
            {"action": "scan", "zone_id": "Near"},
            {"action": "return_to_base"},
            {"action": "land"},
        ]
    }

    updated, log, failures, succeeded = sim.execute_plan(plan, true_state, iteration=1)
    return_step = next(s for s in log if s["action"] == "return_to_base")

    # average distance across ALL zones (visited or not) = (1 + 9) / 2 = 5km
    expected_return_cost = 5 * FUEL_CONSUMPTION_PER_KM
    actual_cost = return_step["state_before"]["fuel"] - return_step["state_after"]["fuel"]
    assert abs(actual_cost - expected_return_cost) < 0.01