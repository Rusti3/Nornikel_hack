from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.mekg.llm_provider import OpenRouterChatAdapter

from .test_mekg import config


class Answer(BaseModel):
    value: int


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(
            choices=[SimpleNamespace(message=value)],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 12}),
        )


def fake_client(*responses):
    completions = FakeCompletions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def openrouter_config(tmp_path, *, keys=("test-key",)):
    return replace(
        config(tmp_path),
        agent_llm_provider="openrouter",
        openrouter_api_keys=keys,
        openrouter_min_interval=0,
        openrouter_model="nvidia/nemotron-3-ultra-550b-a55b:free",
    )


def test_structured_output_enables_reasoning_and_strict_schema(tmp_path):
    message = SimpleNamespace(content='{"value":3}', reasoning_details=[{"type": "reasoning.text", "text": "x"}])
    client, calls = fake_client(message)
    adapter = OpenRouterChatAdapter(openrouter_config(tmp_path), clients=[client])

    result = adapter.with_structured_output(Answer).invoke([
        SystemMessage(content="Return a number"), HumanMessage(content="three")
    ])

    assert result.value == 3
    assert calls.calls[0]["extra_body"]["reasoning"] == {"enabled": True}
    assert calls.calls[0]["extra_body"]["provider"] == {"require_parameters": True}
    assert calls.calls[0]["response_format"]["json_schema"]["strict"] is True


def test_reasoning_details_are_preserved_unmodified_for_one_repair(tmp_path):
    reasoning = [{"type": "reasoning.text", "text": "private"}]
    bad = SimpleNamespace(content="not-json", reasoning_details=reasoning)
    good = SimpleNamespace(content='{"value":4}', reasoning_details=None)
    client, calls = fake_client(bad, good)
    adapter = OpenRouterChatAdapter(openrouter_config(tmp_path), clients=[client])

    result = adapter.invoke_structured(Answer, [HumanMessage(content="four")])

    assert result.value == 4
    assert len(calls.calls) == 2
    assistant = calls.calls[1]["messages"][-2]
    assert assistant["reasoning_details"] is reasoning
    assert "reasoning_details" not in adapter.drain_diagnostics()[0]


class RateLimitError(RuntimeError):
    status_code = 429


def test_key_slot_rotates_on_quota_error_without_exposing_keys(tmp_path):
    first, first_calls = fake_client(RateLimitError("quota"))
    second, second_calls = fake_client(SimpleNamespace(content="ok", reasoning_details=None))
    cfg = openrouter_config(tmp_path, keys=("secret-one", "secret-two"))
    adapter = OpenRouterChatAdapter(cfg, clients=[first, second])

    assert adapter.invoke([HumanMessage(content="hello")]).content == "ok"
    assert len(first_calls.calls) == 1
    assert len(second_calls.calls) == 1
    diagnostics = adapter.drain_diagnostics()
    assert diagnostics[0]["key_slot"] == 1
    assert "secret" not in str(diagnostics)


def test_missing_key_fails_lazily_so_health_and_bm25_can_still_start(tmp_path):
    adapter = OpenRouterChatAdapter(openrouter_config(tmp_path, keys=()))
    try:
        adapter.invoke([HumanMessage(content="hello")])
    except ValueError as exc:
        assert "API keys" in str(exc)
    else:
        raise AssertionError("missing key must fail on first remote request")
