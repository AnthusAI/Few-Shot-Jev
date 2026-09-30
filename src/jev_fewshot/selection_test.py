from .selection import (prototype_balanced, random_balanced, retrieved_balanced,
                        split_candidate_development_scoreboard)


LABELS = ("a", "b")
ROWS = [{"id": f"{label}-{i}", "label": label, "text": f"{label} common token {i}"}
        for label in LABELS for i in range(20)]


def test_a_split_is_disjoint_and_repeatable():
    first = split_candidate_development_scoreboard(ROWS, LABELS, seed=5, development_per_label=3,
                                                    scoreboard_per_label={"a": 2, "b": 2})
    second = split_candidate_development_scoreboard(ROWS, LABELS, seed=5, development_per_label=3,
                                                     scoreboard_per_label={"a": 2, "b": 2})
    assert first == second
    assert len({row["id"] for split in first for row in split}) == len(ROWS)


def test_selectors_preserve_the_requested_label_coverage():
    candidate, _, _ = split_candidate_development_scoreboard(ROWS, LABELS, seed=5, development_per_label=3,
                                                               scoreboard_per_label={"a": 2, "b": 2})
    target = {"id": "target", "label": "a", "text": "a common token"}
    for selected in (random_balanced(candidate, LABELS, per_label=2, seed=0),
                     prototype_balanced(candidate, LABELS, per_label=2),
                     retrieved_balanced(target, candidate, LABELS, per_label=2)):
        assert [row["label"] for row in selected] == ["a", "a", "b", "b"]
        assert target["id"] not in {row["id"] for row in selected}
