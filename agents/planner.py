# agents/planner.py
#
# Planner Agent - LLM-based task decomposition.
#
# The Planner only ever sees ObservedEnvironmentState (noisy/partial) -
# never TrueEnvironmentState. It does NOT estimate fuel/time per step -
# environment/simulator.py is the sole source of truth for physics. The
# Planner's job is purely sequencing and triage: which zones to attempt,
# in what order, and which optional zones to skip if conditions look risky.
#
# CHANGE: generate_plan() now accepts `mismatch_history` (a list of
# MismatchRecord from environment.state) and `failure_history` (the
# Coordinator's failure-type counts). Both are summarized into plain,
# code-computed facts - a wind/fuel error TREND and an explicit
# ESCALATION LEVEL - and placed in the prompt as instructions, not left
# for the model to infer from a wall of raw numbers. This is the direct
# fix for "the planner doesn't update its beliefs" and "no strategy shift
# after repeated failures": the escalation level changes what the prompt
# literally tells the model to do.

from typing import Optional, List

from agents.llm_client import call_structured
from environment.state import summarize_mismatch_trend

VALID_ACTIONS = {"takeoff", "navigate", "scan", "return_to_base", "land"}

SYSTEM_PROMPT = """You are the Planner Agent for a UAV mission planning system.

Output ONLY a single JSON object, no markdown fences, no commentary outside
the JSON, with this exact shape:

{
  "steps": [
    {"action": "takeoff"},
    {"action": "navigate", "zone_id": "<zone_id>"},
    {"action": "scan", "zone_id": "<zone_id>"},
    {"action": "return_to_base"},
    {"action": "land"}
  ],
  "assumptions": ["<short string describing what you assumed about wind/fuel, and how you adjusted for any error trend or escalation level given to you>"]
}

Rules:
- The first step must be "takeoff". The last two steps must be
  "return_to_base" then "land".
- Every required zone MUST be covered by a "navigate" step immediately
  followed by a "scan" step for that same zone_id.
- Optional zones are NOT required for mission success - only include them
  if fuel and wind conditions look comfortable. Skipping an optional zone
  is a valid, often correct way to protect mission success.
- Keep the plan as short as possible: every extra step is extra exposure to
  weather and sensor risk, and that risk only grows on later attempts, so a
  lean plan that nails the required zones beats an ambitious one that risks
  them.
- Visit required zones BEFORE optional ones, so the core mission is secured
  before conditions can change further mid-flight.
- If you are given feedback from a failed previous attempt (look for
  "critic_feedback" below), you MUST change the plan to address it - do not
  repeat the same approach.

If you are given a "mismatch_trend":
- If wind_underestimate_trend is "worsening", you must NOT assume the
  observed wind figure is accurate - treat the true wind as meaningfully
  higher than what you're shown, and plan a shorter, lower-risk route than
  the observed numbers alone would justify.
- If fuel_overestimate_trend is "worsening", treat your observed fuel
  figure as optimistic - do not plan a route that only works if the
  observed fuel number is exactly right.

If you are given an "escalation_level", you must follow its instruction
exactly - it is not optional:
- escalation_level 0: plan normally.
- escalation_level 1 (one prior failure): drop all optional zones
  regardless of how safe conditions look, and order required zones to
  minimize total flight time.
- escalation_level 2 (two or more prior failures, or a repeated failure
  type): plan the absolute minimum-risk mission - only the required
  zones, the shortest possible route between them, and no optional
  zones under any circumstance. Your "assumptions" field must state that
  you are in minimum-risk mode and why.

Do NOT suggest, imply, or assume control over anything outside the plan
itself. You cannot change the simulation, the weather, the wind forecast,
sensor hardware, or fuel levels - you can only decide WHICH zones to
visit, in WHAT order, and whether to skip optional ones. Never write an
assumption like "increase fuel level" or "adjust the wind model" - those
are not actions available to you.
"""


def _validate_plan(parsed: dict) -> None:
    steps = parsed.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Plan must contain a non-empty 'steps' list")
    for step in steps:
        if not isinstance(step, dict) or step.get("action") not in VALID_ACTIONS:
            raise ValueError(f"Invalid or missing action in step: {step}")
        if step["action"] in ("navigate", "scan") and not step.get("zone_id"):
            raise ValueError(f"'{step['action']}' step requires a non-empty 'zone_id': {step}")
    if steps[0]["action"] != "takeoff":
        raise ValueError("First step must be 'takeoff'")
    if [s["action"] for s in steps[-2:]] != ["return_to_base", "land"]:
        raise ValueError("Last two steps must be 'return_to_base' then 'land'")


def compute_escalation_level(failure_history: Optional[dict]) -> int:
    """
    Pure function of the Coordinator's failure_history dict, computed in
    code so the Planner prompt gets an unambiguous integer instead of
    being trusted to "notice" that failures are piling up.

    0 -> no prior failures this mission
    1 -> exactly one prior failure, no type repeated yet
    2 -> two or more prior failures, OR any failure type has repeated
    """
    if not failure_history:
        return 0
    total = sum(failure_history.values())
    repeated = any(count > 1 for count in failure_history.values())
    if repeated or total >= 2:
        return 2
    if total == 1:
        return 1
    return 0


def generate_plan(
    mission: dict,
    observed_state,
    previous_plan: Optional[dict] = None,
    critic_feedback: Optional[dict] = None,
    mismatch_history: Optional[List] = None,
    failure_history: Optional[dict] = None,
) -> dict:
    required_zones = [zid for zid, z in observed_state.zones.items() if z.get("scan_required", True)]
    optional_zones = [zid for zid, z in observed_state.zones.items() if not z.get("scan_required", True)]

    escalation_level = compute_escalation_level(failure_history)
    trend = summarize_mismatch_trend(mismatch_history or [])

    user_prompt = (
        f"Mission objective: {mission.get('objective', '')}\n"
        f"Observed wind speed: {observed_state.wind_kmh} km/h\n"
        f"Observed fuel remaining: {observed_state.fuel_remaining} (fraction of a full tank)\n"
        f"Required zones (must scan): {required_zones}\n"
        f"Optional zones (scan only if conditions look safe): {optional_zones}\n"
        f"Zone details: {observed_state.zones}\n"
        f"escalation_level: {escalation_level}\n"
    )

    if trend:
        user_prompt += f"mismatch_trend: {trend}\n"

    if previous_plan is not None and critic_feedback is not None:
        user_prompt += (
            "\nThis is a REPLAN after a failed attempt.\n"
            f"previous_plan: {previous_plan}\n"
            f"critic_feedback: {critic_feedback}\n"
        )

    return call_structured(SYSTEM_PROMPT, user_prompt, validator=_validate_plan)