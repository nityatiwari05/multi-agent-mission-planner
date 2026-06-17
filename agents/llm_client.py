# agents/llm_client.py
#
# Shared LLM client used by both Planner and Critic.
#
# Every agent calls call_structured(...) and never thinks about whether
# it's talking to a local Ollama model or OpenAI's API - swapping backends
# is a one-line config change (LLM_BACKEND in config/settings.py or the
# LLM_BACKEND env var), not a code change in planner.py / critic.py.
#
# A third backend, "mock", returns deterministic canned responses with no
# network call at all - this is what lets you validate the entire closed
# loop (replanning included) before Ollama is even running.

import json
from typing import Callable, Optional

import requests

from config.settings import (
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
)


class LLMCallError(RuntimeError):
    pass


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_openai(system_prompt: str, user_prompt: str) -> str:
    # Called via raw requests rather than the openai SDK, on purpose -
    # requirements.txt stays dependency-light (requests/dotenv/rich only).
    if not OPENAI_API_KEY:
        raise LLMCallError("LLM_BACKEND=openai but OPENAI_API_KEY is not set.")
    resp = requests.post(
        f"{OPENAI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        },
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def raw_complete(system_prompt: str, user_prompt: str) -> str:
    """One call to whichever backend is configured. No retry/validation here
    - that's layered on top by call_structured()."""
    if LLM_BACKEND == "ollama":
        return _call_ollama(system_prompt, user_prompt)
    if LLM_BACKEND == "openai":
        return _call_openai(system_prompt, user_prompt)
    if LLM_BACKEND == "mock":
        from agents.mock_responses import mock_complete  # test-only dependency, kept local
        return mock_complete(system_prompt, user_prompt)
    raise LLMCallError(f"Unknown LLM_BACKEND '{LLM_BACKEND}' (expected ollama/openai/mock)")


def call_structured(
    system_prompt: str,
    user_prompt: str,
    validator: Optional[Callable[[dict], None]] = None,
    max_retries: Optional[int] = None,
) -> dict:
    """
    Calls the configured LLM backend and parses the result as JSON, retrying
    with the error fed back to the model if the output is malformed or
    fails `validator(parsed_dict)` (validator should raise ValueError on
    invalid input). This is what actually enforces "structured output only"
    - a system-prompt instruction alone won't, especially with an 8B model.
    """
    max_retries = LLM_MAX_RETRIES if max_retries is None else max_retries
    prompt = user_prompt
    last_error: Optional[Exception] = None

    for _ in range(max_retries + 1):
        raw = raw_complete(system_prompt, prompt)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Top-level JSON must be an object")
            if validator:
                validator(parsed)
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            prompt = (
                f"{user_prompt}\n\n"
                f"Your previous response was: {raw}\n"
                f"That response was invalid: {e}\n"
                f"Return ONLY a corrected JSON object. No markdown fences, no commentary."
            )

    raise LLMCallError(f"Failed to get valid structured output after {max_retries + 1} attempts: {last_error}")