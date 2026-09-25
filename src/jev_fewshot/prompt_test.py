from . import LABELS
from .data import Article
from .prompt import QUESTION, state


def test_a_zero_shot_state_keeps_an_empty_example_field():
    target = Article("test", 1, "target", "World")
    assert state(target, []) == {"labeled_examples": [], "target": {"text": "target"}}


def test_a_few_shot_state_separates_target_from_labeled_examples():
    target = Article("test", 1, "target", "World")
    example = Article("train", 2, "example", "Sports")
    request = state(target, [example])
    assert request["target"]["text"] == "target"
    assert request["labeled_examples"] == [{"text": "example", "label": "Sports"}]
    assert QUESTION["criteria"] == {label: None for label in LABELS}
