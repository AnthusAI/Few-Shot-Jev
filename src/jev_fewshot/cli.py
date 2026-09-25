"""CLI: prepare/download, cost-free preflight, explicitly approved run, report."""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from . import DATASET_REVISION, DRAW_SEEDS, SHOT_LEVELS, TEST_PER_CLASS
from .client import JevClient, ResponseCache
from .data import (balanced_scoreboard, demonstration_draws, examples_for_shots, load_ag_news,
                   manifest_rows, write_jsonl)
from .metrics import hierarchical_accuracy_delta, summarize
from .prompt import fingerprint, state

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests" / "scoreboard.jsonl"
RESPONSES = ROOT / "results" / "responses.jsonl"


def selection():
    train, test = load_ag_news(cache_dir=ROOT / "data")
    scoreboard = balanced_scoreboard(test)
    return scoreboard, demonstration_draws(train, scoreboard)


def prepare_manifest() -> None:
    scoreboard, _ = selection()
    write_jsonl(MANIFEST, manifest_rows(scoreboard))
    print(f"wrote {len(scoreboard)} text-free scoreboard rows to {MANIFEST.relative_to(ROOT)}")
    print(f"dataset revision: {DATASET_REVISION}")


def request_plan(scoreboard, draws):
    for target in scoreboard:
        yield 0, None, target, []
    for shots in SHOT_LEVELS[1:]:
        for draw_seed, draw in draws.items():
            examples = examples_for_shots(draw, shots)
            for target in scoreboard:
                yield shots, draw_seed, target, examples


def preflight() -> None:
    scoreboard, draws = selection()
    plan = list(request_plan(scoreboard, draws))
    print(json.dumps({"dataset_revision": DATASET_REVISION, "scoreboard_n": len(scoreboard),
                      "test_per_class": TEST_PER_CLASS, "draw_seeds": list(DRAW_SEEDS),
                      "conditions": list(SHOT_LEVELS), "requests": len(plan),
                      "zero_shot": len(scoreboard), "few_shot": len(plan) - len(scoreboard),
                      "network_calls": 0}, indent=2))
    print("No Jev request was made. Run `jev-agnews prepare-manifest` before committing a release manifest.")


async def live_run(max_requests: int, concurrency: int) -> None:
    scoreboard, draws = selection()
    plan = list(request_plan(scoreboard, draws))
    if max_requests < len(plan):
        raise SystemExit(f"The preregistered run needs {len(plan)} requests; max is {max_requests}.")
    cache = ResponseCache(RESPONSES)
    client = JevClient()
    pending = [entry for entry in plan if cache.get(fingerprint(entry[2], entry[3])) is None]
    if len(pending) > max_requests:
        raise SystemExit(f"{len(pending)} uncached requests exceed the approved maximum of {max_requests}.")
    print(f"{len(plan)} planned; {len(plan) - len(pending)} cached; {len(pending)} live requests to send.")
    completed = 0
    lock = asyncio.Lock()

    async def one(entry) -> None:
        nonlocal completed
        shots, draw_seed, target, examples = entry
        key = fingerprint(target, examples)
        answer = await client.classify(state(target, examples))
        cache.append({"fingerprint": key, "target_id": target.manifest_id, "actual": target.label,
                      "shots": shots, "draw_seed": draw_seed, "prediction": answer.value,
                      "probabilities": answer.probabilities, "model": answer.model,
                      "usage": answer.usage, "latency_ms": answer.latency_ms})
        async with lock:
            completed += 1
            if completed % 100 == 0 or completed == len(pending):
                print(f"{completed}/{len(pending)}", flush=True)

    for start in range(0, len(pending), concurrency):
        await asyncio.gather(*(one(entry) for entry in pending[start:start + concurrency]))


def report() -> None:
    if not RESPONSES.exists():
        raise SystemExit("No response cache exists. Run the approved live experiment first.")
    rows = [json.loads(line) for line in RESPONSES.read_text(encoding="utf-8").splitlines() if line]
    groups: dict[tuple[int, int | None], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["shots"], row.get("draw_seed"))].append(row)
    output = {"dataset_revision": DATASET_REVISION, "conditions": {}}
    for (shots, draw), group in sorted(groups.items()):
        output["conditions"][f"{shots}-shot" + ("" if draw is None else f"/draw-{draw}")] = summarize(group)
    zero = groups.get((0, None), [])
    four = [row for (shots, _), group in groups.items() if shots == 4 for row in group]
    if zero and four:
        interval = hierarchical_accuracy_delta(zero, four)
        output["primary_0_vs_4_accuracy_delta"] = interval.__dict__
    # This aggregate, text-free report is the published study artifact. The
    # per-request response cache remains ignored and local.
    destination = ROOT / "results" / "summary.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {destination.relative_to(ROOT)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-manifest", help="download data and write the text-free release manifest")
    sub.add_parser("preflight", help="enumerate the run without an API call")
    run = sub.add_parser("run", help="execute only with explicit approval")
    run.add_argument("--approve", action="store_true", help="acknowledge paid API use")
    run.add_argument("--max-requests", type=int, required=True)
    run.add_argument("--concurrency", type=int, default=16)
    sub.add_parser("report", help="summarize a completed response cache")
    args = parser.parse_args(argv)
    if args.command == "prepare-manifest":
        prepare_manifest()
    elif args.command == "preflight":
        preflight()
    elif args.command == "run":
        if not args.approve:
            raise SystemExit("Live calls require --approve after inspecting preflight output.")
        from dotenv import load_dotenv
        load_dotenv()
        if args.concurrency < 1:
            raise SystemExit("--concurrency must be positive")
        asyncio.run(live_run(args.max_requests, args.concurrency))
    else:
        report()
