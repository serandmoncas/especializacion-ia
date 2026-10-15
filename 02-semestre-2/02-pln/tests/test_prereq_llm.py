"""Juez LLM con un cliente falso: herramienta forzada, caché, reintentos y respuestas raras."""

from types import SimpleNamespace

import pytest
from research_paths.http import MissingKeyError
from research_paths.prereq.llm import LLMJudge, build_prompt


class OverloadedError(Exception):
    status_code = 529


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)  # cada elemento: bool | "raro" | Exception
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if item == "raro":
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="no sé")])
        block = SimpleNamespace(
            type="tool_use", name="responder", input={"prerrequisito": item, "razon": "porque"}
        )
        return SimpleNamespace(content=[block])


def _judge(tmp_path, script, **kw):
    fake = SimpleNamespace(messages=FakeMessages(script))
    return LLMJudge(tmp_path, client=fake, max_workers=1, sleep=lambda s: None, **kw), fake


ITEM = ("Varianza", "La varianza mide la dispersión.", "Media", "La media es el promedio.")


def test_prompt_contains_both_concepts_and_truncates():
    p = build_prompt("A", "x" * 5000, "B", "texto B", max_chars=100)
    assert "A" in p and "B" in p and "texto B" in p and "x" * 101 not in p


def test_forces_the_tool_and_parses_answers(tmp_path):
    judge, fake = _judge(tmp_path, [True, False])
    other = ("Media", "La media es el promedio.", "Varianza", "La varianza mide la dispersión.")
    assert judge.judge([ITEM, other]) == [True, False]
    call = fake.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "responder"}
    assert call["model"] == "claude-haiku-5-5"


def test_cache_avoids_paying_twice(tmp_path):
    judge, _ = _judge(tmp_path, [True])
    judge.judge([ITEM])
    judge2, fake2 = _judge(tmp_path, [])
    assert judge2.judge([ITEM]) == [True]
    assert fake2.messages.calls == [] and judge2.stats["cache"] == 1


def test_unparseable_answer_is_none_and_counted(tmp_path):
    judge, _ = _judge(tmp_path, ["raro"])
    assert judge.judge([ITEM]) == [None]
    assert judge.stats["unparsed"] == 1


def test_retries_overload_then_succeeds(tmp_path):
    judge, fake = _judge(tmp_path, [OverloadedError(), OverloadedError(), True])
    assert judge.judge([ITEM]) == [True]
    assert len(fake.messages.calls) == 3


def test_non_retryable_error_propagates(tmp_path):
    judge, _ = _judge(tmp_path, [ValueError("mal")])
    with pytest.raises(ValueError):
        judge.judge([ITEM])


def test_missing_key_only_matters_when_the_api_is_needed(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge, _ = _judge(tmp_path, [True])
    judge.judge([ITEM])  # llena la caché con el cliente falso
    cached_only = LLMJudge(tmp_path, client=None, max_workers=1)
    assert cached_only.judge([ITEM]) == [True]  # sin clave, pero todo está en caché
    with pytest.raises(MissingKeyError, match="ANTHROPIC_API_KEY"):
        cached_only.judge([("X", "x", "Y", "y")])
