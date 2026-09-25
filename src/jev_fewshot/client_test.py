from pathlib import Path
from types import SimpleNamespace

from .client import ResponseCache, parse_topic_answer


def test_a_response_cache_resumes_by_complete_request_fingerprint(tmp_path: Path):
    cache = ResponseCache(tmp_path / "responses.jsonl")
    row = {"fingerprint": "one", "target_id": "test-1", "actual": "World", "shots": 0,
           "draw_seed": None, "prediction": "World", "probabilities": {"World": 1.0},
           "model": "fake", "usage": None, "latency_ms": 1.0}
    cache.append(row)
    assert ResponseCache(cache.path).get("one") == row
    assert ResponseCache(cache.path).get("different-complete-state") is None


def test_a_response_cache_rejects_article_text_and_credentials(tmp_path: Path):
    cache = ResponseCache(tmp_path / "responses.jsonl")
    try:
        cache.append({"fingerprint": "one", "text": "article"})
    except ValueError as error:
        assert "text" in str(error)
    else:
        raise AssertionError("article text was accepted")


def test_a_malformed_live_response_is_rejected_before_caching():
    try:
        parse_topic_answer(SimpleNamespace(answers={}), 1.0)
    except ValueError as error:
        assert "topic" in str(error)
    else:
        raise AssertionError("malformed response was accepted")


def test_an_sdk_root_choice_answer_is_normalized():
    answer = SimpleNamespace(model_dump=lambda: {"root": {"choice": "World", "probabilities": {"World": 0.8}}})
    response = SimpleNamespace(answers={"topic": answer}, model="jev-latest", usage=None)
    parsed = parse_topic_answer(response, 1.0)
    assert parsed.value == "World"
    assert parsed.probabilities == {"World": 0.8}
