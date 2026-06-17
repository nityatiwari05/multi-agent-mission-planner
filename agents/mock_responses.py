# agents/mock_responses.py
#
# Deterministic, schema-shaped responses for offline testing of the full
# closed loop with no Ollama or OpenAI running at all. Activated by setting
# LLM_BACKEND=mock (e.g. `LLM_BACKEND=mock python main.py`).
#
# This is NOT meant to be "smart" - it's a stand-in that proves the
# plumbing (Planner -> Executor -> Critic -> Coordinator -> replan) works
# correctly. Real intelligence comes from swapping this for Ollama/OpenAI.
#
# Deliberately a little naive on the first attempt (includes every zone,
# required and optional) and more conservative on a replan (drops optional
# zones once it's seen critic feedback) - this is what lets a single run
# demonstrate the "fail, then improve via replan" path end to end.

import json
import re


def mock_complete(system_prompt: str, user_prompt: str) -> str:
    if "Planner Agent" in system_prompt:
        return json.dumps(_mock_plan(user_prompt))
    if "Critic Agent" in system_prompt:
        return json.dumps(_mock_critic(user_prompt))
    raise ValueError("mock_complete: received a prompt it doesn't know how to handle")


def _mock_plan(user_prompt: str) -> dict:
    required = _extract_list(user_prompt, "Required zones (must scan):")
    optional = _extract_list(user_prompt, "Optional zones (scan only if conditions look safe):")

    is_replan = "critic_feedback" in user_prompt

    zones_to_visit = list(required) if is_replan else list(required) + list(optional)
    if not zones_to_visit:
        zones_to_visit = ["ZoneA"]

    steps = [{"action": "takeoff"}]
    for zid in zones_to_visit:
        steps.append({"action": "navigate", "zone_id": zid})
        steps.append({"action": "scan", "zone_id": zid})
    steps.append({"action": "return_to_base"})
    steps.append({"action": "land"})

    assumption = (
        "replan: dropped optional zones to conserve fuel after critic feedback"
        if is_replan else
        "mock planner v1: attempt every zone, required and optional"
    )

    return {"steps": steps, "assumptions": [assumption]}


def _mock_critic(user_prompt: str) -> dict:
    lower = user_prompt.lower()
    wind_obs, wind_true, fuel_obs, fuel_true = _extract_comparison(user_prompt)
    repeat = _extract_repeat(user_prompt)

    if "fuel_breach" in lower:
        return {
            "what_failed": ["fuel_breach"],
            "state_mismatch": [
                {
                    "variable": "fuel_remaining", "observed": fuel_obs, "true": fuel_true,
                    "impact": "Planner believed it had more fuel margin than it actually did",
                },
                {
                    "variable": "wind_kmh", "observed": wind_obs, "true": wind_true,
                    "impact": "higher true wind increased real fuel burn beyond what was forecast",
                },
            ],
            "causal_chain": [
                {"stage": "planning", "issue": "plan included an optional zone on top of both required zones",
                 "evidence": f"observed fuel={fuel_obs}"},
                {"stage": "reality_gap", "issue": "true fuel was lower and true wind higher than forecast",
                 "evidence": f"true fuel={fuel_true}, true wind={wind_true}"},
                {"stage": "execution", "issue": "fuel required exceeded what was available",
                 "evidence": "fuel_breach event recorded mid-transit"},
            ],
            "why_it_failed": (
                "The plan attempted more zones than the true fuel budget allowed under the "
                "true wind conditions - the forecast understated how tight the margin really was."
            ),
            "suggested_fixes": [
                {"fix": "drop optional (non-required) zones from the plan", "targets": "fuel_overcommitment"},
                {"fix": "scan required zones before any optional ones", "targets": "mission_incomplete_risk"},
            ],
            "repeat_failure": repeat if repeat and repeat["type"] == "fuel_breach" else None,
            "severity": "major",
            "confidence": 0.8,
        }

    if "weather_spike" in lower:
        return {
            "what_failed": ["weather_spike"],
            "state_mismatch": [],
            "causal_chain": [
                {"stage": "execution", "issue": "wind crossed the operational abort threshold mid-mission",
                 "evidence": "weather_spike event recorded"},
            ],
            "why_it_failed": (
                "Wind crossed the operational abort threshold mid-mission - not something the "
                "original plan could have fully prevented."
            ),
            "suggested_fixes": [
                {"fix": "shorten the plan to only required zones", "targets": "weather_exposure_window"},
            ],
            "repeat_failure": repeat if repeat and repeat["type"] == "weather_spike" else None,
            "severity": "major",
            "confidence": 0.6,
        }

    if "sensor_blackout" in lower:
        return {
            "what_failed": ["sensor_blackout"],
            "state_mismatch": [],
            "causal_chain": [
                {"stage": "execution", "issue": "a required zone's sensor failed during scan",
                 "evidence": "sensor_blackout event recorded"},
            ],
            "why_it_failed": "A required zone's sensor failed during scan - a hardware/environmental fault, not a sequencing error.",
            "suggested_fixes": [
                {"fix": "no plan change fixes this on its own - retry once is reasonable", "targets": "sensor_blackout"},
            ],
            "repeat_failure": repeat if repeat and repeat["type"] == "sensor_blackout" else None,
            "severity": "minor",
            "confidence": 0.5,
        }

    return {
        "what_failed": ["incomplete_required_zones"],
        "state_mismatch": [],
        "causal_chain": [
            {"stage": "planning", "issue": "one or more required zones were never included or scanned",
             "evidence": "no matching environmental failure event found in the log"},
        ],
        "why_it_failed": (
            "One or more required zones were never marked scanned, and no specific "
            "environmental failure event was logged - likely a planning gap."
        ),
        "suggested_fixes": [
            {"fix": "verify every required zone has a navigate+scan pair in the plan", "targets": "incomplete_required_zones"},
        ],
        "repeat_failure": None,
        "severity": "major",
        "confidence": 0.5,
    }


def _extract_list(text: str, label: str) -> list:
    """Pulls a Python-list-looking value off a single labeled line, e.g.
    'Required zones (must scan): ['ZoneA', 'ZoneB']' -> ['ZoneA', 'ZoneB']."""
    line = next((l for l in text.splitlines() if l.startswith(label)), "")
    if not line:
        return []
    return re.findall(r"'([^']+)'", line) or re.findall(r'"([^"]+)"', line)


def _extract_comparison(text: str):
    """Pulls the precomputed observed/true numbers back out of the prompt -
    the mock's state_mismatch should be grounded in the same numbers a real
    model would be looking at."""
    wind_match = re.search(r"wind_kmh: observed=([\d.]+), true=([\d.]+)", text)
    fuel_match = re.search(r"fuel_remaining: observed=([\d.]+), true=([\d.]+)", text)
    wind_obs = float(wind_match.group(1)) if wind_match else 0.0
    wind_true = float(wind_match.group(2)) if wind_match else 0.0
    fuel_obs = float(fuel_match.group(1)) if fuel_match else 0.0
    fuel_true = float(fuel_match.group(2)) if fuel_match else 0.0
    return wind_obs, wind_true, fuel_obs, fuel_true


def _extract_repeat(text: str):
    m = re.search(r"repeated across earlier iterations \(type -> count\): \{'(\w+)': (\d+)\}", text)
    if not m:
        return None
    return {"type": m.group(1), "count": int(m.group(2)), "implication": "risk increasing across iterations"}