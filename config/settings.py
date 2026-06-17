# config/settings.py
# All tunable constants in one place.
# Swap OLLAMA → OpenAI by changing LLM_BACKEND and LLM_MODEL.

import os

# ──────────────────────────────────────────────
# LLM Backend
# ──────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")   # "ollama" | "openai" | "mock"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL = "https://api.openai.com/v1"

LLM_TIMEOUT = 120          # seconds per LLM call
LLM_MAX_RETRIES = 2        # retries on HTTP error

# ──────────────────────────────────────────────
# Coordinator
# ──────────────────────────────────────────────
MAX_REPLAN_ITERATIONS = 3   # hard stop

# ──────────────────────────────────────────────
# Executor / Simulator
# ──────────────────────────────────────────────
# Fuel
#
# FIX: fuel_remaining is a 0.0-1.0 fraction of a full tank (see
# environment/state.py), but FUEL_CONSUMPTION_PER_KM=1.2 was calibrated for
# a much larger unit scale - at 1.2 "units" per km, a single 3km leg
# (3.6 units) would instantly exceed an entire full tank (1.0), so every
# mission would fuel-breach on step one regardless of plan quality. Rescaled
# so a full tank covers a realistic ~20km under calm conditions and
# meaningfully less under headwind - tight enough that skipping an optional
# zone is a real, fuel-saving decision, not free.
FUEL_CONSUMPTION_PER_KM = 0.04       # fraction of full tank per km (true)
FUEL_HEADWIND_MULTIPLIER = 1.5       # extra consumption when wind > threshold
WIND_FUEL_THRESHOLD_KMH = 15        # wind above this increases fuel use

# Weather
WEATHER_ABORT_WIND_KMH = 35          # wind above this = mission abort
#
# FIX: this is checked before EVERY step (see environment/simulator.py),
# not once per iteration. With a ~7-9 step plan, a 0.35 per-step chance
# compounds to a ~63% abort rate on iteration 1 alone, and ~98%+ by
# iteration 2 - weather drowns out fuel_breach/sensor_blackout entirely and
# the mission almost never gets a chance to succeed. Lowered so iteration 1
# is mostly safe (~2% abort) while iteration 2+ is still meaningfully
# riskier (~32-67%, growing via WEATHER_SPIKE_GROWTH_PER_ITER below) -
# keeps the "conditions get worse the longer this drags on" intent, but
# lets fuel/sensor failures and successful missions actually surface.
WEATHER_SPIKE_PROBABILITY = 0.06     # chance of a spike per step, iteration 1
#
# Was hardcoded as `+ (iteration - 1) * 0.15` inside check_weather_spike.
# At 0.15, iteration 2 jumped to a ~64% abort rate even with the base above
# lowered - any replan was nearly as doomed as doing nothing. 0.05 gives a
# smoother climb (iter1 ~2%, iter2 ~32%, iter3 ~67%): still a real "don't
# dawdle" pressure, but iteration 2 is a genuine second chance, not a
# coin-flip-against-the-house.
WEATHER_SPIKE_GROWTH_PER_ITER = 0.05

# Sensor
#
# NOTE: mission success requires zero failures of any kind (see
# environment/simulator.py's `land` handler), so this probability applies
# independently per required zone per scan. At 0.3, two required zones
# alone give a ~51% chance of at least one blackout before weather or fuel
# even enter the picture. Lowered so it's a real but not dominant risk.
SENSOR_BLACKOUT_PROBABILITY = 0.12   # per zone, per execution, iteration 1
#
# Was hardcoded as `+ (iteration - 1) * 0.1`. 0.06 gives a smoother climb
# (iter1 ~23%, iter2 ~33%, iter3 ~42% with 2 required zones) instead of a
# sharper one - consistent with the same "gentler iteration 2" goal as the
# weather growth rate above.
SENSOR_BLACKOUT_GROWTH_PER_ITER = 0.06
SENSOR_NOISE_STD = 0.05              # gaussian noise on sensor readings (0–1 scale)

# ──────────────────────────────────────────────
# Observation Gap (hidden state vs observed)
# Gap grows each iteration to simulate worsening conditions
# ──────────────────────────────────────────────
BASE_WIND_OBSERVATION_ERROR_KMH = 1   # iteration 1 underestimation
WIND_ERROR_GROWTH_PER_ITER = 4        # added per iteration
BASE_FUEL_OBSERVATION_ERROR = 0.05    # fraction of true fuel, iteration 1
FUEL_ERROR_GROWTH_PER_ITER = 0.03

# ──────────────────────────────────────────────
# Reward / Scoring
# ──────────────────────────────────────────────
REWARD_MISSION_SUCCESS = 100
REWARD_ZONE_SCANNED = 15
REWARD_STEP_PENALTY = -2
REWARD_FAILURE_PENALTY = -25
REWARD_CONSTRAINT_VIOLATION = -20

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOGS_DIR = "logs"
EPISODE_LOG_SUFFIX = "_episode.jsonl"
SUMMARY_LOG_SUFFIX = "_summary.json"