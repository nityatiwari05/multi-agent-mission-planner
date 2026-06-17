# memory/episode_logger.py
#
# Writes one JSONL file per mission run (one line per iteration) plus a
# summary JSON. This is the raw material a future RL training step would
# read (state -> action -> reward per iteration); for now it also doubles
# as your debugging trail and your "show your work" artifact for an
# interview.

import json
from pathlib import Path

from config.settings import LOGS_DIR, EPISODE_LOG_SUFFIX, SUMMARY_LOG_SUFFIX


class EpisodeLogger:
    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.logs_dir = Path(LOGS_DIR)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.episode_path = self.logs_dir / f"{episode_id}{EPISODE_LOG_SUFFIX}"
        self.summary_path = self.logs_dir / f"{episode_id}{SUMMARY_LOG_SUFFIX}"

    def log_iteration(self, record: dict) -> None:
        with self.episode_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_summary(self, summary: dict) -> None:
        self.summary_path.write_text(json.dumps(summary, indent=2, default=str))