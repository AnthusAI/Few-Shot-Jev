#!/usr/bin/env python3
"""Preregistered fixed-budget selector comparison for Jev on Emotion."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jev_fewshot.selection import (prototype_balanced, random_balanced, retrieved_balanced,
                                   split_candidate_development_scoreboard)

ROOT = Path(__file__).resolve().parents[1]
REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")
SCOREBOARD_COUNTS = {"sadness": 583, "joy": 670, "love": 163, "anger": 270, "fear": 242, "surprise": 72}
SEEDS = tuple(range(5)); PER_LABEL = 16
TOTAL_RECORDS = 19000
QUESTION = {"type": "choice", "instructions": "Classify only target.text into its primary emotion. labeled_examples are training examples of the intended categories; do not classify them.", "criteria": {label: None for label in LABELS}}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_splits():
    data = load_dataset("dair-ai/emotion", revision=REVISION, cache_dir=str(ROOT / "data"))
    rows = [{"id": f"train-{i}", "text": row["text"], "label": LABELS[int(row["label"])]}
            for i, row in enumerate(data["train"])]
    return split_candidate_development_scoreboard(rows, LABELS, seed=20260930, development_per_label=100,
                                                   scoreboard_per_label=SCOREBOARD_COUNTS)


def state(target, examples):
    return {"labeled_examples": [{"text": row["text"], "label": row["label"]} for row in examples], "target": {"text": target["text"]}}


def contexts(candidate):
    random_contexts = {f"random-{seed}": random_balanced(candidate, LABELS, per_label=PER_LABEL, seed=seed) for seed in SEEDS}
    return random_contexts | {"prototype": prototype_balanced(candidate, LABELS, per_label=PER_LABEL)}


def requests():
    candidate, development, scoreboard = load_splits(); static = contexts(candidate)
    for name in sorted(key for key in static if key.startswith("random-")):
        for target in development: yield "development", name, target, static[name]
    for target in scoreboard: yield "scoreboard", "zero", target, []
    for name, examples in static.items():
        for target in scoreboard: yield "scoreboard", name, target, examples
    for target in scoreboard: yield "scoreboard", "retrieval", target, retrieved_balanced(target, candidate, LABELS, per_label=PER_LABEL)


def cache_path(): return ROOT / "results" / "emotion_selection_responses.jsonl"
def summary_path(): return ROOT / "results" / "emotion_selection_summary.json"
def manifest_path(): return ROOT / "manifests" / "emotion_selection_scoreboard.jsonl"


def read_cache():
    if not cache_path().exists(): return {}
    return {row["fingerprint"]: row for row in (json.loads(line) for line in cache_path().read_text().splitlines() if line.strip()) if row.get("prediction") in LABELS}


def write_manifest():
    _, _, scoreboard = load_splits(); manifest_path().parent.mkdir(exist_ok=True)
    records = [{"id": row["id"], "label": row["label"], "normalized_text_sha256": hashlib.sha256(" ".join(row["text"].split()).lower().encode()).hexdigest()} for row in scoreboard]
    manifest_path().write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    print(f"wrote {len(records)} text-free scoreboard records")


def macro_f1(rows):
    values=[]
    for label in LABELS:
        tp=sum(r["actual"]==label and r["prediction"]==label for r in rows); fp=sum(r["actual"]!=label and r["prediction"]==label for r in rows); fn=sum(r["actual"]==label and r["prediction"]!=label for r in rows)
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(values)/len(values)


def metrics(rows):
    usage=[r.get("usage") or {} for r in rows]
    return {"n":len(rows), "accuracy":sum(r["actual"]==r["prediction"] for r in rows)/len(rows), "macro_f1":macro_f1(rows), "brier":sum(sum((float(r["probabilities"].get(label,0))-float(r["actual"]==label))**2 for label in LABELS) for r in rows)/len(rows), "recall":{label:sum(r["actual"]==label and r["prediction"]==label for r in rows)/sum(r["actual"]==label for r in rows) for label in LABELS}, "mean_latency_ms":sum(r["latency_ms"] for r in rows)/len(rows), "input_tokens":sum(x.get("input_tokens") or 0 for x in usage), "output_tokens":sum(x.get("output_tokens") or 0 for x in usage)}


def materialized_rows():
    cache=read_cache(); output=[]; keys=set()
    for split, method, target, examples in requests():
        key=digest({"state":state(target,examples),"question":QUESTION}); keys.add(key)
        if key not in cache: raise SystemExit(f"missing response for {split}/{method}/{target['id']}")
        row=dict(cache[key]); row.update({"split":split,"method":method,"target_id":target["id"],"actual":target["label"]}); output.append(row)
    return output, len(keys)


def report():
    rows, unique_states=materialized_rows(); by=defaultdict(list)
    for row in rows: by[(row["split"],row["method"])].append(row)
    dev={name:metrics(by[("development",name)]) for name in sorted(name for split,name in by if split=="development")}
    winner=max(dev, key=lambda name: (dev[name]["macro_f1"], name))
    score={name:metrics(by[("scoreboard",name)]) for name in sorted(name for split,name in by if split=="scoreboard")}
    random_mean=sum(score[f"random-{seed}"]["macro_f1"] for seed in SEEDS)/len(SEEDS)
    output={"dataset":"dair-ai/emotion", "dataset_revision":REVISION, "candidate_split":"train remainder", "scoreboard_n":2000, "analysis_records":len(rows), "unique_model_states":unique_states, "development":{"scores":dev,"selected_global_context":winner}, "scoreboard":score, "comparisons":{"random_macro_f1_mean":random_mean,"prototype_minus_random":score["prototype"]["macro_f1"]-random_mean,"retrieval_minus_random":score["retrieval"]["macro_f1"]-random_mean,"development_selected_minus_random":score[winner]["macro_f1"]-random_mean}}
    summary_path().write_text(json.dumps(output,indent=2,sort_keys=True)+"\n"); print(json.dumps(output["comparisons"],indent=2))


async def run(max_requests, concurrency):
    cache=read_cache()
    # Cache states can be shared by byte-identical targets.  This conservative
    # ceiling check may overestimate resumed work by only those duplicates.
    if TOTAL_RECORDS-len(cache)>max_requests: raise SystemExit(f"up to {TOTAL_RECORDS-len(cache)} requests remain; max is {max_requests}")
    print(f"{TOTAL_RECORDS} analysis records; {len(cache)} cached states; streaming remaining requests.")
    client=AsyncTypeSafeClient(retry=RetryPolicy(max_retries=6,backoff_max=30.)); lock=asyncio.Lock(); done=0; cache_path().parent.mkdir(exist_ok=True)
    async def one(item):
        nonlocal done
        split,method,target,examples=item; request_state=state(target,examples); key=digest({"state":request_state,"question":QUESTION}); started=time.perf_counter()
        response=await client.system_one(state=request_state,questions={"emotion":QUESTION}); raw=response.answers["emotion"].model_dump(); raw=raw.get("root",raw)
        row={"fingerprint":key,"split":split,"method":method,"target_id":target["id"],"actual":target["label"],"prediction":raw["choice"],"probabilities":raw["probabilities"],"model":response.model,"usage":response.usage.model_dump(),"latency_ms":round((time.perf_counter()-started)*1000,2)}
        async with lock:
            with cache_path().open("a") as out: out.write(json.dumps(row,sort_keys=True)+"\n")
            done+=1
            if done%100==0: print(f"{done} new responses",flush=True)
    batch=[]
    for item in requests():
        key=digest({"state":state(item[2],item[3]),"question":QUESTION})
        if key in cache: continue
        batch.append(item)
        if len(batch)==concurrency:
            await asyncio.gather(*(one(current) for current in batch)); batch=[]
    if batch: await asyncio.gather(*(one(current) for current in batch))
    print(f"completed {done} new responses",flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("manifest","preflight","run","report")); parser.add_argument("--approve",action="store_true"); parser.add_argument("--max-requests",type=int,default=19000); parser.add_argument("--concurrency",type=int,default=8); args=parser.parse_args()
    if args.command=="manifest": write_manifest(); return
    if args.command=="report": report(); return
    if args.command=="preflight": print(json.dumps({"dataset_revision":REVISION,"analysis_records":TOTAL_RECORDS,"expected_unique_request_ceiling":TOTAL_RECORDS,"network_calls":0},indent=2)); return
    if not args.approve: raise SystemExit("Live calls require --approve")
    load_dotenv(ROOT/".env"); asyncio.run(run(args.max_requests,args.concurrency))

if __name__=="__main__": main()
