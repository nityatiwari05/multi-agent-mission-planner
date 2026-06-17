"""
main.py - CLI entry point for the UAV multi-agent mission planner.

Usage:
    python main.py
    python main.py --mission missions/inspect_zone_a.json
    python main.py --mission missions/inspect_zone_a.json --verbose

    # Offline demo / test - no Ollama or OpenAI needed:
    LLM_BACKEND=mock python main.py --verbose
"""
import argparse
import json
import random

from coordinator.coordinator import run_mission


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a UAV mission through the multi-agent planner.")
    parser.add_argument("--mission", default="missions/inspect_zone_a.json", help="Path to a mission JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration detail as the loop runs")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed Python's random module for reproducible environment behavior "
             "(true wind/fuel draws and failure rolls). With LLM_BACKEND=mock this "
             "makes the whole run fully deterministic; with a real LLM the Planner's "
             "wording can still vary slightly run to run.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    with open(args.mission) as f:
        mission = json.load(f)

    summary = run_mission(mission, verbose=args.verbose)

    print("\n" + "=" * 60)
    print(f"Episode:    {summary['episode_id']}")
    print(f"Objective:  {summary['mission_objective']}")
    print(f"Iterations: {summary['total_iterations']}")
    print(f"Outcome:    {summary['final_action']}")
    print(f"Reasoning:  {summary['final_reasoning']}")
    print(f"Succeeded:  {summary['succeeded']}")
    print(f"Total reward: {summary['total_reward']}")
    print("=" * 60)
    print(f"Full episode log: logs/{summary['episode_id']}_episode.jsonl")
    print(f"Summary:          logs/{summary['episode_id']}_summary.json")


if __name__ == "__main__":
    main()