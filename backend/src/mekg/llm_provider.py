from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel

from .config import MEKGConfig


SchemaT = TypeVar("SchemaT", bound=BaseModel)
_RATE_LOCK = threading.Lock()
_LAST_CALL: dict[tuple[str, str], float] = {}


@dataclass(frozen=True)
class LLMCallDiagnostics:
    provider: str
    model: str
    elapsed_seconds: float
    key_slot: int
    repaired: bool = False
    usage: dict[str, Any] | None = None


def _message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, dict) else str(item)
            for item in value
        )
    return str(value or "")


def _to_openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        role = getattr(message, "type", "user")
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        result.append({"role": role, "content": _message_content(getattr(message, "content", message))})
    return result


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


class _StructuredInvoker(Generic[SchemaT]):
    def __init__(self, adapter: "OpenRouterChatAdapter", schema: type[SchemaT]) -> None:
        self.adapter = adapter
        self.schema = schema

    def invoke(self, messages: list[Any]) -> SchemaT:
        return self.adapter.invoke_structured(self.schema, messages)


class OpenRouterChatAdapter:
    """LangChain-compatible adapter with key rotation and private reasoning repair."""

    provider_name = "openrouter"

    def __init__(self, config: MEKGConfig, *, clients: list[Any] | None = None) -> None:
        self.config = config
        self.model = config.openrouter_model
        self._keys = config.openrouter_api_keys
        self._clients = clients or [
            OpenAI(
                api_key=key,
                base_url=config.openrouter_base_url,
                timeout=config.openrouter_timeout,
                max_retries=0,
            )
            for key in self._keys
        ]
        self._slot = 0
        self._slot_lock = threading.Lock()
        self._diagnostics: list[LLMCallDiagnostics] = []

    def with_structured_output(
        self, schema: type[SchemaT], *, method: str = "json_schema", **_kwargs: Any
    ) -> _StructuredInvoker[SchemaT]:
        if method != "json_schema":
            raise ValueError("OpenRouter adapter supports json_schema structured output only")
        return _StructuredInvoker(self, schema)

    def invoke(self, messages: list[Any]) -> AIMessage:
        message, diagnostics = self._request(_to_openai_messages(messages))
        self._diagnostics.append(diagnostics)
        additional = {}
        reasoning = getattr(message, "reasoning_details", None)
        if reasoning is not None:
            additional["reasoning_details"] = reasoning
        return AIMessage(content=_message_content(message.content), additional_kwargs=additional)

    def invoke_structured(self, schema: type[SchemaT], messages: list[Any]) -> SchemaT:
        openai_messages = _to_openai_messages(messages)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        }
        first_message = None
        first_error: Exception | None = None
        try:
            first_message, diagnostics = self._request(
                openai_messages, response_format=response_format
            )
            parsed = schema.model_validate_json(_message_content(first_message.content))
            self._diagnostics.append(diagnostics)
            return parsed
        except Exception as exc:
            if _status_code(exc) in {401, 402, 429}:
                raise
            first_error = exc

        repair_messages = list(openai_messages)
        if first_message is not None:
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": _message_content(first_message.content),
            }
            reasoning = getattr(first_message, "reasoning_details", None)
            if reasoning is not None:
                # OpenRouter requires these blocks to be passed back unmodified.
                assistant["reasoning_details"] = reasoning
            repair_messages.extend([
                assistant,
                {
                    "role": "user",
                    "content": "The previous output was invalid. Return only corrected JSON matching the schema.",
                },
            ])
        else:
            repair_messages.append({
                "role": "system",
                "content": (
                    "Return JSON only. It must validate against this JSON Schema: "
                    + json.dumps(schema.model_json_schema(), ensure_ascii=False)
                ),
            })
        try:
            repaired_message, diagnostics = self._request(repair_messages)
            result = schema.model_validate_json(_message_content(repaired_message.content))
            self._diagnostics.append(LLMCallDiagnostics(
                provider=diagnostics.provider,
                model=diagnostics.model,
                elapsed_seconds=diagnostics.elapsed_seconds,
                key_slot=diagnostics.key_slot,
                repaired=True,
                usage=diagnostics.usage,
            ))
            return result
        except Exception as repair_error:
            raise RuntimeError(
                f"OpenRouter structured output failed: {type(first_error).__name__}; "
                f"repair failed: {type(repair_error).__name__}"
            ) from repair_error

    def drain_diagnostics(self) -> list[dict[str, Any]]:
        values = [item.__dict__.copy() for item in self._diagnostics]
        self._diagnostics.clear()
        return values

    def _request(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[Any, LLMCallDiagnostics]:
        if not self._clients:
            raise ValueError("No OpenRouter API keys configured")
        attempted = 0
        last_error: Exception | None = None
        while attempted < len(self._clients):
            with self._slot_lock:
                slot = self._slot % len(self._clients)
                client = self._clients[slot]
            self._wait_for_rate_limit()
            started = time.monotonic()
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self.config.openrouter_max_output_tokens,
                "extra_body": {
                    "reasoning": {"enabled": self.config.openrouter_reasoning},
                    "provider": {"require_parameters": bool(response_format)},
                },
            }
            if response_format:
                kwargs["response_format"] = response_format
            try:
                response = client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                usage_value = getattr(response, "usage", None)
                usage = usage_value.model_dump() if hasattr(usage_value, "model_dump") else None
                return message, LLMCallDiagnostics(
                    provider=self.provider_name,
                    model=self.model,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    key_slot=slot,
                    usage=usage,
                )
            except Exception as exc:
                last_error = exc
                attempted += 1
                if _status_code(exc) not in {401, 402, 429} or attempted >= len(self._clients):
                    raise
                with self._slot_lock:
                    self._slot = (slot + 1) % len(self._clients)
        raise RuntimeError("All OpenRouter key slots are unavailable") from last_error

    def _wait_for_rate_limit(self) -> None:
        key = (self.config.openrouter_base_url, self.model)
        with _RATE_LOCK:
            now = time.monotonic()
            wait_for = self.config.openrouter_min_interval - (now - _LAST_CALL.get(key, 0.0))
            if wait_for > 0:
                time.sleep(wait_for)
            _LAST_CALL[key] = time.monotonic()


def build_agent_llm(config: MEKGConfig) -> Any:
    if config.agent_llm_provider == "openrouter":
        return OpenRouterChatAdapter(config)
    if config.agent_llm_provider != "yandex":
        raise ValueError("Unsupported AGENT_LLM_PROVIDER")
    return ChatOpenAI(
        api_key=config.yandex_api_key,
        base_url=config.yandex_base_url,
        model=config.llm_model,
        temperature=0,
        timeout=90,
        max_retries=1,
        default_headers={"OpenAI-Project": config.yandex_folder_id},
    )
