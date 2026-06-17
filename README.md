# UAV Multi-Agent Mission Planner

A closed-loop multi-agent mission planning system for UAVs.
Uses LLM-based Planner + Critic agents, rule-based Executor + Coordinator,
with a hidden true environment state vs. observed state gap.

## Architecture

```
User Input (JSON)
      ↓
Planner Agent  (LLM - generates structured plan)
      ↓
Executor       (rule-based simulation, hidden true state)
      ↓
Critic Agent   (LLM - diagnoses failures, suggests fixes)
      ↓
Coordinator    (rule-based - retry / adjust / hard-stop at 3)
      ↓
JSONL Episode Log + CLI Summary
```

## Setup

```bash
pip install -r requirements.txt

# Make sure Ollama is running with llama3.1:8b
ollama pull llama3.1:8b
ollama serve
```

## Run

```bash
# Run with default mission
python main.py

# Run with custom mission file
python main.py --mission missions/inspect_zone_a.json

# Run with verbose logging
python main.py --mission missions/inspect_zone_a.json --verbose
```

## Failure Scenarios Tested

- **Fuel breach**: UAV runs out of fuel mid-mission due to underestimated consumption
- **Sensor blackout**: Sensor fails in a zone, scan data lost
- **Weather spike**: Wind spikes beyond UAV operational threshold

## Key Design Decisions

- **Hidden true state**: Planner sees observed (noisy) environment; executor runs on true state
- **Dynamic gap**: Observation error changes each iteration (worsening conditions)
- **Episode logging**: Every step logged as JSONL for future RL training
- **Learning path**: Rule-based coordinator → validate → log trajectories → Q-learning

## Folder Structure

```
multi_agennt_mission_planner/
├── main.py                  # Entry point
├── config/
│   └── settings.py          # All tuneable constants
├── agents/
│   ├── planner.py           # LLM Planner Agent
│   └── critic.py            # LLM Critic Agent
├── environment/
│   ├── state.py             # True state + observed state dataclasses
│   ├── simulator.py         # Executor - simulates action outcomes
│   └── failure_modes.py     # Failure injection logic
├── coordinator/
│   └── coordinator.py       # Orchestration + replan loop
├── memory/
│   └── episode_logger.py    # JSONL episode logger
├── missions/
│   └── inspect_zone_a.json  # Example mission input
├── logs/                    # Auto-created, stores run logs
├── tests/
│   └── test_simulator.py    # Unit tests
└── requirements.txt
```