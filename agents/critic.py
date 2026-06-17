# agents/critic.py
#
# Critic Agent - post-mission diagnosis.
#
# Structured causal schema: an explicit observed-vs-true state_mismatch
# (grounded in numbers computed in code, not left for the model to
# calculate itself), a step-by-step causal_chain, fix -> cause
# traceability, explicit repeat handling, and a confidence score.
#
# CHANGE: added a deterministic post-processing filter,
# _enforce_plan_level_fixes(). The system prompt already told the model
# fixes must be Planner-actionable, but an 8B model still produced fixes
# like "adjust the simulation model" / "increase the estimated fuel
# level" - instructions the Planner has no way to act on, since it cannot
# touch the simulation, the wind forecast, or fuel levels. A prompt
# instruction alone isn't enforcement; this filter is. It scans every fix
# string for "world-level" language (touches the simulation, the
# forecast, sensor hardware, fuel levels themselves) and replaces any
# fix that matches with a safe, generic plan-level fallback so a bad fix
# never reaches the Planner's prompt as a stated instruction.

from typing import List

from agents.llm_client import call_structured

VALID_SEVERITIES = {"none", "minor", "major", "critical"}

# Phrases that indicate the model is trying to exert control it doesn't
# have - i.e. anything that isn't "which zones, what order, skip what".
_WORLD_LEVEL_PATTERNS = [
    "adjust the simulation", "adjust simulation", "change the simulation",
    "fix the simulation", "simulation model",
    "adjust the wind", "adjust wind model", "change the wind", "fix the wind",
    "wind model", "wind forecast model",
    "increase the estimated fuel", "increase fuel level", "increase the fuel level",
    "increase fuel estimate", "adjust fuel level", "change fuel level",
    "improve sensor", "fix the sensor", "replace the sensor", "upgrade sensor",
    "sensor hardware",
    "lower the abort threshold", "raise the abort threshold", "change the abort threshold",
    "adjust the threshold",
]

# Fallback fixes, keyed by the failure type they're attached to via
# `targets`. Used only when a proposed fix is rejected by the filter -
# these are intentionally generic but always genuinely actionable by the
# Planner (sequencing/triage only).
_SAFE_FALLBACKS = {
    "fuel_breach": "Drop optional zones and shorten the route to reduce total fuel consumption.",
    "weather_spike": "Shorten the plan and scan required zones earlier to reduce exposure time to worsening weather.",
    "sensor_blackout": "Scan required zones earlier in the sequence and drop optional zones to reduce total exposure.",
    "incomplete_required_zones": "Verify every required zone has an explicit navigate+scan pair before optional zones are added.",
}
_DEFAULT_FALLBACK = "Shorten the plan to required zones only and minimize total flight distance."


def _is_world_level(fix_text: str) -> bool:
    lowered = fix_text.lower()
    return any(pattern in lowered for pattern in _WORLD_LEVEL_PATTERNS)


def _enforce_plan_level_fixes(parsed: dict) -> dict:
    """
    Mutates parsed["suggested_fixes"] in place: any fix whose text reads
    as world-level control gets swapped for a safe plan-level fallback,
    chosen by matching `targets` against _SAFE_FALLBACKS. Returns parsed
    for convenience.
    """
    fixes = parsed.get("suggested_fixes", [])
    cleaned = []
    for fix in fixes:
        if isinstance(fix, dict):
            text = fix.get("fix", "")
            target = fix.get("targets", "")
        else:
            # Tolerate older/looser shapes (plain string fixes) defensively.
            text = str(fix)
            target = ""

        if _is_world_level(text):
            replacement = _SAFE_FALLBACKS.get(target, _DEFAULT_FALLBACK)
            cleaned.append({
                "fix": replacement,
                "targets": target or "general_risk_reduction",
                "_rewritten_from_world_level": text,
            })
        else:
            cleaned.append(fix if isinstance(fix, dict) else {"fix": text, "targets": target})

    parsed["suggested_fixes"] = cleaned
    return parsed


SYSTEM_PROMPT = """You are the Critic Agent for a UAV mission planning system.

You receive the plan, what the Planner assumed (the observed/forecast
state), the true state as it actually was at planning time (only available
now, for post-incident review), a precomputed observed-vs-true comparison,
the state at the end of execution (which can differ from the planning-time
true state - e.g. fuel drained, wind that spiked mid-flight), the
execution log, any failure events, and whether any failure type has
recurred across earlier iterations of this same mission.

Output ONLY a single JSON object, no markdown fences, no commentary outside
the JSON, with this exact shape:

{
  "what_failed": ["<failure_type strings>"],
  "state_mismatch": [
    {"variable": "<e.g. fuel_remaining>", "observed": <number>, "true": <number>, "impact": "<one sentence>"}
  ],
  "causal_chain": [
    {"stage": "planning" | "reality_gap" | "execution", "issue": "<what went wrong at this stage>", "evidence": "<a number or fact from the trace>"}
  ],
  "why_it_failed": "<1-2 sentence summary of the causal chain above>",
  "suggested_fixes": [
    {"fix": "<concrete, Planner-actionable change>", "targets": "<which failure/cause this fix addresses>"}
  ],
  "repeat_failure": {"type": "<failure_type>", "count": <int>, "implication": "<one sentence>"},
  "severity": "none" | "minor" | "major" | "critical",
  "confidence": <number between 0.0 and 1.0>
}

Rules:
- "state_mismatch" must use the precomputed observed-vs-true numbers you
  were given - do not invent numbers. Leave it as an empty list only if
  the failure genuinely had nothing to do with a forecast being wrong
  (e.g. a sensor hardware fault, a weather spike beyond what any forecast
  window could plausibly have caught).
- "causal_chain" must cite an actual number or fact from the trace at each
  stage - not a restatement like "the plan was wrong" with nothing backing
  it.
- "suggested_fixes" must be changes the Planner can actually make in its
  NEXT plan: which zones to visit, what order to visit them in, and
  whether to skip optional zones. The Planner has NO other levers. NEVER
  suggest changing the simulation, the wind/weather model, fuel levels,
  sensor hardware, or any abort/safety threshold - the Planner cannot act
  on any of those, and a fix worded that way will be discarded. Every fix
  must be phrasable as "skip X", "visit X before Y", "scan X earlier", or
  "shorten the route by dropping X". Each fix's "targets" must name the
  specific cause it addresses.
- "repeat_failure": set this to the repeat info you were given if a
  failure type has recurred across iterations of THIS mission; otherwise
  set it to null. Do not invent a repeat that wasn't given to you.
- Before assigning severity, reason explicitly: could a different zone
  ordering or pruning have avoided this failure? If yes, it is NOT
  critical (use "major" or below). If no plan change could plausibly have
  helped, use "critical".
- "confidence" reflects how well-grounded your causal_chain actually is in
  the evidence you were given - not how badly the mission turned out.
"""


def _validate_critic(parsed: dict) -> None:
    required_keys = {
        "what_failed", "state_mismatch", "causal_chain", "why_it_failed",
        "suggested_fixes", "repeat_failure", "severity", "confidence",
    }
    missing = required_keys - parsed.keys()
    if missing:
        raise ValueError(f"Critic output missing keys: {missing}")
    if parsed["severity"] not in VALID_SEVERITIES:
        raise ValueError(f"Invalid severity '{parsed['severity']}'")
    for list_field in ("what_failed", "state_mismatch", "causal_chain", "suggested_fixes"):
        if not isinstance(parsed[list_field], list):
            raise ValueError(f"'{list_field}' must be a list")
    if parsed["repeat_failure"] is not None and not isinstance(parsed["repeat_failure"], dict):
        raise ValueError("'repeat_failure' must be an object or null")
    try:
        confidence = float(parsed["confidence"])
    except (TypeError, ValueError):
        raise ValueError("'confidence' must be a number")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("'confidence' must be between 0.0 and 1.0")


def evaluate_mission(
    plan: dict,
    observed_state,
    true_state_initial,
    true_state_final,
    execution_log: List[dict],
    failures: List,
    failure_history: dict,
) -> dict:
    repeats = {ft: c for ft, c in failure_history.items() if c > 1}

    # Precomputed in code, not left for the model to calculate. Uses the
    # INITIAL true state (what was actually true at planning time) against
    # the observed/forecast state the Planner saw - that's the genuine
    # forecast-error comparison. The post-execution true state (e.g. fuel
    # drained to 0, or wind that spiked mid-flight) is a different thing -
    # "what happened during the flight" - and is given separately below for
    # situational context, not folded into this comparison.
    comparison_lines = [
        f"  wind_kmh: observed={observed_state.wind_kmh}, true={true_state_initial.wind_kmh} "
        f"(diff={true_state_initial.wind_kmh - observed_state.wind_kmh:+.2f})",
        f"  fuel_remaining: observed={observed_state.fuel_remaining}, true={true_state_initial.fuel_remaining} "
        f"(diff={true_state_initial.fuel_remaining - observed_state.fuel_remaining:+.3f})",
    ]

    user_prompt = (
        f"Plan: {plan}\n"
        f"Observed/forecast state given to Planner: {observed_state.to_dict()}\n"
        f"True state at planning time (only available now, for post-incident review): {true_state_initial.to_dict()}\n"
        f"Precomputed observed-vs-true comparison (at planning time):\n" + "\n".join(comparison_lines) + "\n"
        f"State at the end of execution (e.g. fuel drained, or wind that may have spiked mid-flight): {true_state_final.to_dict()}\n"
        f"Execution log: {execution_log}\n"
        f"Failure events this attempt: {[f.__dict__ for f in failures]}\n"
        f"Failure types repeated across earlier iterations (type -> count): {repeats or 'none'}\n"
    )

    parsed = call_structured(SYSTEM_PROMPT, user_prompt, validator=_validate_critic)
    return _enforce_plan_level_fixes(parsed)