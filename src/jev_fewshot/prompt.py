"""One deliberately invariant Jev request design for every experimental arm."""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

from . import LABELS
from .data import Article

QUESTION = {
    "type": "choice",
    "instructions": (
        "Classify only target.text, a news article, into its primary topic. "
        "The labeled_examples are training examples of the intended categories; do not "
        "classify them. Choose exactly one option for target.text."
    ),
    # Choice criteria are a label-keyed mapping in the SDK. `None` means the
    # label itself is the only supplied definition; examples carry the task
    # specific information being tested here.
    "criteria": {label: None for label in LABELS},
}


def state(target: Article, examples: Sequence[Article]) -> dict:
    """State is structurally identical in all conditions; only examples changes."""
    return {
        "labeled_examples": [{"text": example.text, "label": example.label} for example in examples],
        "target": {"text": target.text},
    }


def fingerprint(target: Article, examples: Sequence[Article]) -> str:
    payload = {"state": state(target, examples), "questions": {"topic": QUESTION}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
