# coordinator/coordinator.py
#
# Coordinator - orchestrates the full closed loop:
#     Planner -> Executor -> Critic -> decide(stop/replan) -> repeat
#
# Rule-based by design: validate the loop and the environment dynamics
# with a hard-coded policy first, log real episodes, and only then
# consider swapping `_decide` for a learned policy. Everything upstream
# (Planner, Executor, Critic) stays untouched when that swap happens -
# `_decide` is the only function with that responsibility.
#
# CHANGE: now builds a MismatchRecord after every iteration and threads
# the growing list (`mismatch_history`) into generate_plan(), alongside
# failure_history. This is what gives the Planner an actual trend to
# react to across replans, instead of a single fresh noisy snapshot each
# time with no memory of how wrong the last one turned out to be.

import time
import uuid
from typing import Dict, List

from config.settings import (
    MAX_REPLAN_ITERATIONS,
    REWARD_MISSION_SUCCESS,
    REWARD_ZONE_SCANNED,
    REWARD_STEP_PENALTY,
    REWARD_FAILURE_PENALTY,
    REWARD_CONSTRAINT_VIOLATION,
)
from environment.state import build_initial_true_state, generate_observed_state, build_mismatch_record
from environment.simulator import execute_plan
from agents.planner import generate_plan
from agents.critic import evaluate_mission
from memory.episode_logger import EpisodeLogger


def run_mission(mission: dict, verbose: bool = False) -> dict:
    """
    Runs one mission end-to-end through the closed loop and returns a
    summary dict. Every iteration is logged to
    logs/<episode_id>_episode.jsonl, with a final
    logs/<episode_id>_summary.json.
    """
    episode_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    logger = EpisodeLogger(episode_id)

    # Built ONCE - every iteration is a fresh attempt at the SAME mission,
    # not a continuation of a failed flight. Only the per-step failure
    # probabilities (via the `iteration` argument into execute_plan) model
    # "conditions getting worse the longer this drags on".
    true_state = build_initial_true_state(mission)

    plan = None
    critic_report = None
    failure_history: Dict[str, int] = {}
    mismatch_history: List = []
    iteration_records: List[dict] = []
    final_decision = None

    for iteration in range(1, MAX_REPLAN_ITERATIONS + 1):
        observed_state = generate_observed_state(true_state, iteration)

        plan = generate_plan(
            mission=mission,
            observed_state=observed_state,
            previous_plan=plan,
            critic_feedback=critic_report,
            mismatch_history=mismatch_history,
            failure_history=failure_history,
        )

        updated_true_state, execution_log, failures, mission_succeeded = execute_plan(
            plan=plan, true_state=true_state, iteration=iteration,
        )

        # Record this iteration's observed-vs-true error BEFORE failure
        # bookkeeping below, using the planning-time true_state (not
        # updated_true_state) - this mirrors exactly what the Critic
        # compares against, so the trend the Planner sees next iteration
        # matches the trend the Critic is reasoning about this iteration.
        mismatch_history.append(build_mismatch_record(iteration, observed_state, true_state))

        for f in failures:
            failure_history[f.failure_type] = failure_history.get(f.failure_type, 0) + 1

        if mission_succeeded:
            critic_report = {
                "what_failed": [], "state_mismatch": [], "causal_chain": [],
                "why_it_failed": "Mission succeeded - all required zones scanned with no failures.",
                "suggested_fixes": [], "repeat_failure": None, "severity": "none", "confidence": 1.0,
            }
        else:
            critic_report = evaluate_mission(
                plan=plan,
                observed_state=observed_state,
                true_state_initial=true_state,
                true_state_final=updated_true_state,
                execution_log=execution_log,
                failures=failures,
                failure_history=failure_history,
            )

        decision = _decide(
            mission_succeeded=mission_succeeded,
            critic_report=critic_report,
            failures=failures,
            failure_history=failure_history,
            iteration=iteration,
            max_iterations=MAX_REPLAN_ITERATIONS,
        )

        reward = _compute_reward(mission_succeeded, execution_log, failures)

        record = {
            "iteration": iteration,
            "plan": plan,
            "observed_state": observed_state.to_dict(),
            "true_state_after": updated_true_state.to_dict(),
            "mismatch_record": mismatch_history[-1].to_dict(),
            "execution_log": execution_log,
            "failures": [f.__dict__ for f in failures],
            "critic_report": critic_report,
            "decision": decision,
            "mission_succeeded": mission_succeeded,
            "reward": reward,
        }
        iteration_records.append(record)
        logger.log_iteration(record)

        if verbose:
            _print_iteration(record)

        final_decision = decision
        if decision["action"] != "replan":
            break

    summary = {
        "episode_id": episode_id,
        "mission_objective": mission.get("objective", ""),
        "total_iterations": len(iteration_records),
        "final_action": final_decision["action"],
        "final_reasoning": final_decision["reasoning"],
        "succeeded": iteration_records[-1]["mission_succeeded"],
        "total_reward": sum(r["reward"] for r in iteration_records),
    }
    logger.log_summary(summary)
    return summary


def _decide(
    mission_succeeded: bool,
    critic_report: dict,
    failures: list,
    failure_history: Dict[str, int],
    iteration: int,
    max_iterations: int,
) -> dict:
    if mission_succeeded:
        return {"action": "stop_success", "reasoning": f"Mission succeeded on iteration {iteration}."}

    severity = critic_report.get("severity", "major")

    # Failure probabilities in environment/failure_modes.py grow with each
    # iteration - so a failure type recurring isn't "still unlucky", it's
    # "more likely to happen again than the first time". Treat repeats as
    # unrecoverable rather than burning further iterations against rising
    # odds with no structural fix available.
    repeated = [f.failure_type for f in failures if failure_history.get(f.failure_type, 0) > 1]
    if repeated:
        return {
            "action": "stop_unrecoverable",
            "reasoning": f"'{repeated[0]}' has now recurred across iterations - risk is rising, not falling, so further retries are unlikely to help.",
        }

    if severity == "critical":
        return {
            "action": "stop_unrecoverable",
            "reasoning": f"Critic flagged a critical, plan-unfixable failure: {critic_report.get('why_it_failed')}",
        }

    if iteration >= max_iterations:
        return {
            "action": "stop_max_iterations",
            "reasoning": f"Reached the iteration cap ({max_iterations}) without a successful mission.",
        }

    return {
        "action": "replan",
        "reasoning": f"Failure was '{severity}' and plan-fixable - sending Critic feedback back to the Planner.",
    }


def _compute_reward(mission_succeeded: bool, execution_log: list, failures: list) -> float:
    """
    Scalar reward per iteration, using the REWARD_* constants already
    defined in config/settings.py. Not consumed by anything yet (the
    Coordinator's policy is still rule-based, per the build plan) - this is
    what gets logged now so a future RL training pass has (state, action,
    reward) trajectories to learn from without re-running everything.
    """
    reward = REWARD_MISSION_SUCCESS if mission_succeeded else 0.0
    zones_scanned = sum(1 for step in execution_log if str(step.get("outcome", "")).startswith("scan_success"))
    reward += zones_scanned * REWARD_ZONE_SCANNED
    reward += len(execution_log) * REWARD_STEP_PENALTY
    reward += len(failures) * REWARD_FAILURE_PENALTY
    if not mission_succeeded:
        reward += REWARD_CONSTRAINT_VIOLATION
    return reward


def _print_iteration(record: dict) -> None:
    print(f"\n-- Iteration {record['iteration']} --")
    print(f"  Mission succeeded: {record['mission_succeeded']}")
    mm = record.get("mismatch_record", {})
    if mm:
        print(f"  This iteration's error: wind underestimated by {mm['wind_underestimate_kmh']} km/h, "
              f"fuel overestimated by {mm['fuel_overestimate_fraction']}")
    if record["failures"]:
        for f in record["failures"]:
            print(f"  FAILURE: {f['failure_type']} - {f['description']}")
    cr = record["critic_report"]
    if cr.get("what_failed"):
        print(f"  Critic severity: {cr['severity']} (confidence: {cr.get('confidence', 'n/a')})")
        for m in cr.get("state_mismatch", []):
            print(f"  Mismatch: {m['variable']} observed={m['observed']} true={m['true']} - {m['impact']}")
        for step in cr.get("causal_chain", []):
            print(f"  [{step['stage']}] {step['issue']} (evidence: {step['evidence']})")
        print(f"  Why: {cr['why_it_failed']}")
        for fix in cr.get("suggested_fixes", []):
            if isinstance(fix, dict):
                rewritten = " [auto-corrected: model proposed a non-actionable fix]" if fix.get("_rewritten_from_world_level") else ""
                print(f"  Fix: {fix.get('fix')} (targets: {fix.get('targets')}){rewritten}")
            else:
                print(f"  Fix: {fix}")
        if cr.get("repeat_failure"):
            rf = cr["repeat_failure"]
            print(f"  Repeat: {rf.get('type')} x{rf.get('count')} - {rf.get('implication')}")
    print(f"  Reward: {record['reward']}")
    print(f"  Decision: {record['decision']['action']} - {record['decision']['reasoning']}")