#!/usr/bin/env python3
"""Frozen zero-/few-shot Jev study on dair-ai/emotion.

The held-out scoreboard is the dataset's complete 2,000-row test split.  The
six labels are naturally imbalanced, so accuracy is accompanied by macro-F1.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

ROOT = Path(__file__).resolve().parents[1]
REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")
SEEDS = range(5)
SHOTS = (0, 6, 12, 24)
QUESTION = {"type": "choice", "instructions": (
    "Classify only target.text into its primary emotion. labeled_examples are "
    "training examples of the intended categories; do not classify them."),
    "criteria": {label: None for label in LABELS}}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_rows():
    data = load_dataset("dair-ai/emotion", revision=REVISION, cache_dir=str(ROOT / "data"))
    def rows(split):
        return [{"id": f"{split}-{i}", "text": row["text"], "label": LABELS[int(row["label"])]}
                for i, row in enumerate(data[split])]
    return rows("train"), rows("test")


def draws(train):
    output = {}
    for seed in SEEDS:
        rng = random.Random(seed)
        draw = []
        for label in LABELS:
            draw.extend(rng.sample([row for row in train if row["label"] == label], 4))
        output[seed] = draw
    return output


def selected_examples(draw, shots):
    if shots == 0:
        return []
    n = shots // len(LABELS)
    return [row for label in LABELS for row in [r for r in draw if r["label"] == label][:n]]


def state(target, examples):
    return {"labeled_examples": [{"text": r["text"], "label": r["label"]} for r in examples],
            "target": {"text": target["text"]}}


def plan(test, all_draws):
    for target in test:
        yield 0, None, target, []
    for shots in SHOTS[1:]:
        for seed, draw in all_draws.items():
            demos = selected_examples(draw, shots)
            for target in test:
                yield shots, seed, target, demos


def cache_path(): return ROOT / "results" / "emotion_responses.jsonl"


def read_cache():
    path = cache_path()
    return {row["fingerprint"]: row for row in (json.loads(line) for line in path.read_text().splitlines())} if path.exists() else {}


async def run(max_requests, concurrency):
    train, test = load_rows(); all_draws = draws(train); requests = list(plan(test, all_draws)); cache = read_cache()
    pending = [(s, d, t, e) for s, d, t, e in requests if digest({"state": state(t, e), "question": QUESTION}) not in cache]
    if max_requests < len(pending): raise SystemExit(f"{len(pending)} requests remain; max is {max_requests}")
    print(f"{len(requests)} planned; {len(cache)} cached; {len(pending)} to send.")
    client = AsyncTypeSafeClient(retry=RetryPolicy(max_retries=6, backoff_max=30.0)); lock = asyncio.Lock(); done = 0
    cache_path().parent.mkdir(exist_ok=True)
    async def one(item):
        nonlocal done
        shots, seed, target, demos = item; request_state = state(target, demos); key = digest({"state": request_state, "question": QUESTION}); started = time.perf_counter()
        response = await client.system_one(state=request_state, questions={"emotion": QUESTION})
        raw = response.answers["emotion"].model_dump().get("root", {})
        row = {"fingerprint": key, "target_id": target["id"], "actual": target["label"], "shots": shots, "draw_seed": seed,
               "prediction": raw["choice"], "probabilities": raw["probabilities"], "model": response.model,
               "usage": response.usage.model_dump(), "latency_ms": round((time.perf_counter()-started)*1000,2)}
        async with lock:
            with cache_path().open("a") as handle: handle.write(json.dumps(row, sort_keys=True)+"\n")
            done += 1
            if done % 100 == 0 or done == len(pending): print(f"{done}/{len(pending)}", flush=True)
    for start in range(0, len(pending), concurrency): await asyncio.gather(*(one(x) for x in pending[start:start+concurrency]))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command", choices=("preflight", "run")); parser.add_argument("--approve", action="store_true"); parser.add_argument("--max-requests", type=int, default=32000); parser.add_argument("--concurrency", type=int, default=16); args=parser.parse_args()
    train, test = load_rows(); all_draws = draws(train)
    if args.command == "preflight": print(json.dumps({"dataset_revision": REVISION, "test_n": len(test), "labels": LABELS, "requests": len(list(plan(test, all_draws))), "network_calls": 0}, indent=2)); return
    if not args.approve: raise SystemExit("Live calls require --approve")
    load_dotenv(ROOT / ".env"); asyncio.run(run(args.max_requests, args.concurrency))

if __name__ == "__main__": main()
