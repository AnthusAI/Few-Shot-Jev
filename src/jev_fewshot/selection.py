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
    groups = _GROUPS.setdefault(id(candidate), {label: [row for row in candidate if row["label"] == label]
                                                 for label in labels})
    output = []
    for label in labels:
        def score(row: dict):
            row_tokens = tokens(row["text"])
            overlap = len(target_tokens.intersection(row_tokens))
            similarity = overlap / math.sqrt(max(1, len(target_tokens)) * max(1, len(row_tokens)))
            return similarity, row["id"]
        output.extend(nlargest(per_label, groups[label], key=score))
    return output
