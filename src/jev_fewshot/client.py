"""The only live boundary, plus JSONL response storage safe for public release."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .prompt import QUESTION


@dataclass(frozen=True)
class Answer:
    value: str
    probabilities: dict[str, float]
    model: str | None
    usage: dict | None
    latency_ms: float


def _dict(value: Any) -> dict:
    return value.model_dump() if hasattr(value, "model_dump") else dict(value)


def parse_topic_answer(response: Any, latency_ms: float) -> Answer:
    """Validate a response before it can enter a resumable result cache."""
    answers = getattr(response, "answers", None) or {}
    if "topic" not in answers:
        raise ValueError("Jev response did not contain the required topic answer")
    answer = _dict(answers["topic"])
    # SDK 0.7 represents the answer union as a Pydantic RootModel.
    answer = answer.get("root", answer)
    if answer.get("choice") is None:
        raise ValueError("Jev topic answer did not contain a choice")
    probabilities = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
    return Answer(str(answer["choice"]), probabilities, getattr(response, "model", None),
                  _dict(response.usage) if getattr(response, "usage", None) else None, latency_ms)


class JevClient:
    """Lazy SDK client: importing, preflighting, and testing never reaches the network."""
    def __init__(self) -> None:
        self._client = None

    async def classify(self, request_state: Mapping[str, Any]) -> Answer:
        if self._client is None:
            if not os.environ.get("TYPESAFE_API_KEY"):
                raise RuntimeError("TYPESAFE_API_KEY is required for live execution.")
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
            self._client = AsyncTypeSafeClient(retry=RetryPolicy(max_retries=6, backoff_max=30.0))
        started = time.perf_counter()
        response = await self._client.system_one(state=dict(request_state), questions={"topic": QUESTION})
        return parse_topic_answer(response, round((time.perf_counter() - started) * 1000, 2))


class ResponseCache:
    """Append-only JSONL cache. Rows intentionally contain no target or example text."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._rows: dict[str, dict] = {}
        if self.path.exists():
            self._rows = {row["fingerprint"]: row for row in self._read()}

    def _read(self):
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def get(self, request_fingerprint: str) -> dict | None:
        return self._rows.get(request_fingerprint)

    def append(self, row: dict) -> None:
        if "text" in json.dumps(row).lower() or "typesafe_api_key" in json.dumps(row).lower():
            raise ValueError("Response artifacts must not contain text or credentials")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._rows[row["fingerprint"]] = row
