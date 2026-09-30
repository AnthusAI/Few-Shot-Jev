"""Deterministic, label-preserving demonstration selectors."""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from functools import lru_cache
from heapq import nlargest
from typing import Iterable


TOKEN = re.compile(r"[a-z]+")
_GROUPS: dict[int, dict[str, list[dict]]] = {}
_RETRIEVAL_INDEX: dict[int, dict[str, tuple[list[dict], dict[str, list[int]], list[int]]]] = {}


@lru_cache(maxsize=None)
def tokens(text: str) -> frozenset[str]:
    """Tokenize each immutable dataset text once across all retrieval calls."""
    return frozenset(TOKEN.findall(text.lower()))


def split_candidate_development_scoreboard(rows: list[dict], labels: tuple[str, ...], *, seed: int,
                                           development_per_label: int, scoreboard_per_label: dict[str, int]):
    """Create disjoint deterministic splits without looking at article text or outcomes."""
    candidate, development, scoreboard = [], [], []
    for offset, label in enumerate(labels):
        group = [row for row in rows if row["label"] == label]
        shuffled = group.copy()
        random.Random(seed + offset).shuffle(shuffled)
        scoreboard_n = scoreboard_per_label[label]
        development.extend(shuffled[:development_per_label])
        scoreboard.extend(shuffled[development_per_label:development_per_label + scoreboard_n])
        candidate.extend(shuffled[development_per_label + scoreboard_n:])
    return candidate, development, scoreboard


def random_balanced(candidate: list[dict], labels: tuple[str, ...], *, per_label: int, seed: int) -> list[dict]:
    return [row for label in labels for row in random.Random(seed).sample(
        [row for row in candidate if row["label"] == label], per_label)]


def prototype_balanced(candidate: list[dict], labels: tuple[str, ...], *, per_label: int) -> list[dict]:
    """Select class-central records using a transparent lexical centroid proxy."""
    selected = []
    for label in labels:
        group = [row for row in candidate if row["label"] == label]
        frequency = Counter(token for row in group for token in tokens(row["text"]))
        def score(row: dict):
            row_tokens = tokens(row["text"])
            return (sum(frequency[token] for token in row_tokens) / len(row_tokens) if row_tokens else 0.0,
                    -len(row["text"]), row["id"])
        selected.extend(sorted(group, key=score, reverse=True)[:per_label])
    return selected


def retrieved_balanced(target: dict, candidate: list[dict], labels: tuple[str, ...], *, per_label: int) -> list[dict]:
    """Return the lexical nearest candidates per label, using no target label."""
    target_tokens = tokens(target["text"])
    candidate_id = id(candidate)
    groups = _GROUPS.setdefault(candidate_id, {label: [row for row in candidate if row["label"] == label]
                                               for label in labels})
    if candidate_id not in _RETRIEVAL_INDEX:
        indexed = {}
        for label, group in groups.items():
            inverted: dict[str, list[int]] = {}
            lengths = []
            for index, row in enumerate(group):
                row_tokens = tokens(row["text"])
                lengths.append(len(row_tokens))
                for token in row_tokens:
                    inverted.setdefault(token, []).append(index)
            indexed[label] = group, inverted, lengths
        _RETRIEVAL_INDEX[candidate_id] = indexed
    output = []
    for label in labels:
        group, inverted, lengths = _RETRIEVAL_INDEX[candidate_id][label]
        overlaps: Counter[int] = Counter(index for token in target_tokens for index in inverted.get(token, ()))
        # A positive overlap always outranks a zero overlap.  Include the
        # deterministic ID tie-break only when unusually short targets need
        # zero-overlap fillers.
        ranked = nlargest(per_label, overlaps,
                          key=lambda index: (overlaps[index] / math.sqrt(max(1, len(target_tokens)) * max(1, lengths[index])),
                                             group[index]["id"]))
        if len(ranked) < per_label:
            positive = set(overlaps)
            ranked.extend(nlargest(per_label - len(ranked),
                                   (index for index in range(len(group)) if index not in positive),
                                   key=lambda index: group[index]["id"]))
        output.extend(group[index] for index in ranked)
    return output
