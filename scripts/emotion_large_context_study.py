#!/usr/bin/env python3
"""Preregistered large-context Jev study on the untouched Emotion validation split.

This file deliberately does not reuse the completed test-split experiment.
Its cache and published aggregate are separate, and it never writes text.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

ROOT = Path(__file__).resolve().parents[1]
REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")
SEEDS = tuple(range(5))
PER_LABEL = (0, 1, 4, 16, 64)
QUESTION = {"type": "choice", "instructions": (
    "Classify only target.text into its primary emotion. labeled_examples are "
    "training examples of the intended categories; do not classify them."),
    "criteria": {label: None for label in LABELS}}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_rows():
    data = load_dataset("dair-ai/emotion", revision=REVISION, cache_dir=str(ROOT / "data"))
    def rows(split: str):
        return [{"id": f"{split}-{i}", "text": row["text"], "label": LABELS[int(row["label"])]}
                for i, row in enumerate(data[split])]
    return rows("train"), rows("validation")


def draws(train: list[dict]) -> dict[int, list[dict]]:
    """Draw 64 examples/class; every lower dose is its deterministic prefix."""
    by_label = {label: [row for row in train if row["label"] == label] for label in LABELS}
    return {seed: [row for label in LABELS for row in random.Random(seed).sample(by_label[label], 64)]
            for seed in SEEDS}


def selected_examples(draw: list[dict], per_label: int) -> list[dict]:
    return [row for label in LABELS for row in
            [candidate for candidate in draw if candidate["label"] == label][:per_label]]


def state(target: dict, examples: list[dict]) -> dict:
    return {"labeled_examples": [{"text": row["text"], "label": row["label"]} for row in examples],
            "target": {"text": target["text"]}}


def plan(scoreboard: list[dict], all_draws: dict[int, list[dict]]):
    for target in scoreboard:
        yield 0, None, target, []
    for per_label in PER_LABEL[1:]:
        for seed, draw in all_draws.items():
            examples = selected_examples(draw, per_label)
            for target in scoreboard:
                yield per_label * len(LABELS), seed, target, examples


def cache_path() -> Path:
    return ROOT / "results" / "emotion_large_context_responses.jsonl"


def summary_path() -> Path:
    return ROOT / "results" / "emotion_large_context_summary.json"


def read_cache() -> dict[str, dict]:
    path = cache_path()
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {row["fingerprint"]: row for row in rows if row.get("prediction") in LABELS}


def scored_rows() -> tuple[list[dict], int]:
    """Expand a cached decision to every scoreboard record with that exact state.

    A handful of validation texts are byte-for-byte identical.  They share a
    complete state/question fingerprint and therefore correctly share one
    model call, but each remains a separate, preselected scoreboard item.
    """
    train, scoreboard = load_rows()
    cache = read_cache()
    output, required = [], set()
    for shots, seed, target, examples in plan(scoreboard, draws(train)):
        key = digest({"state": state(target, examples), "question": QUESTION})
        required.add(key)
        if key not in cache:
            raise SystemExit(f"Missing cached response for {target['id']} ({shots} shots).")
        row = dict(cache[key])
        row.update({"target_id": target["id"], "actual": target["label"], "shots": shots, "draw_seed": seed})
        output.append(row)
    return output, len(required)


def reliability(rows: list[dict], bins: int = 10) -> list[dict]:
    output = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [row for row in rows if low <= max(row["probabilities"].get(x, 0.0) for x in LABELS) < high or
                  (index == bins - 1 and max(row["probabilities"].get(x, 0.0) for x in LABELS) == 1)]
        if bucket:
            conf = sum(max(row["probabilities"].get(x, 0.0) for x in LABELS) for row in bucket) / len(bucket)
            acc = sum(row["actual"] == row["prediction"] for row in bucket) / len(bucket)
            output.append({"low": low, "high": high, "n": len(bucket), "confidence": conf, "accuracy": acc})
    return output


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    actual, predicted = [r["actual"] for r in rows], [r["prediction"] for r in rows]
    recalls, f1s = {}, []
    for label in LABELS:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        denom = sum(a == label for a in actual)
        recalls[label] = tp / denom if denom else 0.0
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    brier = sum(sum((float(row["probabilities"].get(label, 0.0)) - float(row["actual"] == label)) ** 2
                    for label in LABELS) for row in rows) / len(rows)
    log_loss = -sum(math.log(max(float(row["probabilities"].get(row["actual"], 0.0)), 1e-12)) for row in rows) / len(rows)
    rel = reliability(rows)
    ece = sum(item["n"] / len(rows) * abs(item["confidence"] - item["accuracy"]) for item in rel)
    usage = [row.get("usage") or {} for row in rows]
    return {"n": len(rows), "accuracy": sum(a == p for a, p in zip(actual, predicted)) / len(rows),
            "macro_f1": sum(f1s) / len(f1s), "recall": recalls, "brier": brier, "log_loss": log_loss,
            "ece": ece, "reliability": rel,
            "mean_latency_ms": sum(row.get("latency_ms", 0.0) for row in rows) / len(rows),
            "input_tokens": sum(item.get("input_tokens") or 0 for item in usage),
            "output_tokens": sum(item.get("output_tokens") or 0 for item in usage)}


def bootstrap_macro_f1(zero_rows: list[dict], many_rows: list[dict], resamples: int = 2000, seed: int = 20260928) -> dict:
    zero = {row["target_id"]: row for row in zero_rows}
    by_draw: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in many_rows:
        by_draw[int(row["draw_seed"])][row["target_id"]] = row
    draw_ids, target_ids = sorted(by_draw), sorted(set(zero).intersection(*(set(x) for x in by_draw.values())))
    label_index = {label: index for index, label in enumerate(LABELS)}
    actual = [label_index[zero[target]["actual"]] for target in target_ids]
    zero_prediction = [label_index[zero[target]["prediction"]] for target in target_ids]
    few_prediction = {draw: [label_index[by_draw[draw][target]["prediction"]] for target in target_ids]
                      for draw in draw_ids}

    def macro_from_matrix(matrix: list[list[int]]) -> float:
        f1s = []
        for label in range(len(LABELS)):
            tp = matrix[label][label]
            fp = sum(matrix[row][label] for row in range(len(LABELS)) if row != label)
            fn = sum(matrix[label][column] for column in range(len(LABELS)) if column != label)
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        return sum(f1s) / len(f1s)

    def delta(draw_counts: Counter, target_counts: Counter) -> float:
        baseline = [[0] * len(LABELS) for _ in LABELS]
        few = [[0] * len(LABELS) for _ in LABELS]
        for target_index, repetitions in target_counts.items():
            baseline[actual[target_index]][zero_prediction[target_index]] += repetitions
        for draw, draw_repetitions in draw_counts.items():
            predictions = few_prediction[draw]
            for target_index, repetitions in target_counts.items():
                few[actual[target_index]][predictions[target_index]] += draw_repetitions * repetitions
        return macro_from_matrix(few) - macro_from_matrix(baseline)

    estimate = delta(Counter(draw_ids), Counter(range(len(target_ids))))
    rng = random.Random(seed)
    values = sorted(delta(Counter(rng.choice(draw_ids) for _ in draw_ids),
                          Counter(rng.randrange(len(target_ids)) for _ in target_ids))
                    for _ in range(resamples))
    return {"estimate": estimate, "low": values[int(.025 * resamples)], "high": values[int(.975 * resamples)],
            "resamples": resamples, "seed": seed}


def report() -> None:
    rows, unique_model_states = scored_rows()
    by_condition: dict[int, list[dict]] = defaultdict(list)
    for row in rows: by_condition[int(row["shots"])] .append(row)
    output = {"dataset": "dair-ai/emotion", "dataset_revision": REVISION, "scoreboard_split": "validation",
              "scoreboard_n": 2000, "labels": LABELS, "analysis_records": len(rows),
              "unique_model_states": unique_model_states, "conditions": {},
              "primary": {"baseline_shots": 0, "many_shots": 384, "metric": "macro_f1"}}
    for shots in sorted(by_condition):
        condition = by_condition[shots]
        if shots == 0:
            output["conditions"][str(shots)] = {"pooled": metrics(condition)}
        else:
            per_draw = {str(seed): metrics([row for row in condition if row["draw_seed"] == seed]) for seed in SEEDS}
            mean = {key: sum(item[key] for item in per_draw.values()) / len(per_draw)
                    for key in ("accuracy", "macro_f1", "brier", "log_loss", "ece", "mean_latency_ms")}
            output["conditions"][str(shots)] = {"per_draw": per_draw, "mean": mean, "pooled": metrics(condition)}
    if 0 in by_condition and 384 in by_condition:
        output["primary"]["hierarchical_bootstrap_macro_f1_delta"] = bootstrap_macro_f1(by_condition[0], by_condition[384])
    summary_path().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["primary"], indent=2))


async def run(max_requests: int, concurrency: int) -> None:
    train, scoreboard = load_rows(); all_draws = draws(train); requests = list(plan(scoreboard, all_draws)); cache = read_cache()
    pending = [(shots, seed, target, examples) for shots, seed, target, examples in requests
               if digest({"state": state(target, examples), "question": QUESTION}) not in cache]
    if max_requests < len(pending):
        raise SystemExit(f"{len(pending)} requests remain; max is {max_requests}")
    print(f"{len(requests)} planned; {len(cache)} cached; {len(pending)} to send.")
    client = AsyncTypeSafeClient(retry=RetryPolicy(max_retries=6, backoff_max=30.0)); lock = asyncio.Lock(); done = 0
    cache_path().parent.mkdir(exist_ok=True)
    async def one(item):
        nonlocal done
        shots, seed, target, examples = item; request_state = state(target, examples); key = digest({"state": request_state, "question": QUESTION}); started = time.perf_counter()
        response = await client.system_one(state=request_state, questions={"emotion": QUESTION})
        raw = response.answers["emotion"].model_dump(); raw = raw.get("root", raw)
        row = {"fingerprint": key, "target_id": target["id"], "actual": target["label"], "shots": shots, "draw_seed": seed,
               "prediction": raw["choice"], "probabilities": raw["probabilities"], "model": response.model,
               "usage": response.usage.model_dump(), "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        async with lock:
            with cache_path().open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")
            done += 1
            if done % 100 == 0 or done == len(pending): print(f"{done}/{len(pending)}", flush=True)
    for start in range(0, len(pending), concurrency):
        await asyncio.gather(*(one(item) for item in pending[start:start + concurrency]))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("preflight", "run", "report")); parser.add_argument("--approve", action="store_true"); parser.add_argument("--max-requests", type=int, default=42000); parser.add_argument("--concurrency", type=int, default=8); args = parser.parse_args()
    if args.command == "report": report(); return
    train, scoreboard = load_rows(); all_draws = draws(train)
    if args.command == "preflight":
        print(json.dumps({"dataset_revision": REVISION, "scoreboard_split": "validation", "scoreboard_n": len(scoreboard), "labels": LABELS, "requests": len(list(plan(scoreboard, all_draws))), "network_calls": 0}, indent=2)); return
    if not args.approve: raise SystemExit("Live calls require --approve")
    load_dotenv(ROOT / ".env"); asyncio.run(run(args.max_requests, args.concurrency))


if __name__ == "__main__": main()
