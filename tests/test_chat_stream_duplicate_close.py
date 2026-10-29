from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from omlx.api.openai_models import ChatCompletionRequest, Message
from omlx.server import stream_chat_completion
from omlx.api.thinking import ThinkingParser, extract_thinking


@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_duplicate_close_parser_chunk_boundaries(chunk_size):
    parser = ThinkingParser(start_in_thinking=True, defer_empty_initial_close=True)
    text = "</think>private planning</think>answer"
    parts = [parser.feed(text[i:i + chunk_size]) for i in range(0, len(text), chunk_size)]
    parts.append(parser.finish())
    assert "".join(t for t, _ in parts) == "private planning"
    assert "".join(c for _, c in parts) == "answer"


def test_nonempty_initial_thinking_is_not_deferred():
    parser = ThinkingParser(start_in_thinking=True, defer_empty_initial_close=True)
    assert parser.feed("reason</think>answer") == ("reason", "answer")
    assert parser.finish() == ("", "")


def test_single_empty_close_remains_visible():
    parser = ThinkingParser(start_in_thinking=True, defer_empty_initial_close=True)
    assert parser.feed("</think>answer") == ("", "")
    assert parser.finish() == ("", "answer")


def test_nonstream_duplicate_close_keeps_reasoning():
    assert extract_thinking("</think>private planning</think>answer", start_in_thinking=True) == ("private planning", "answer")



class _Tokenizer:
    think_start = "<think>"
    think_start_id = 1
    think_end = "</think>"
    think_end_id = 2
    unk_token_id = -1

    def encode(self, text, add_special_tokens=False):
        if text.rstrip().endswith("<think>"):
            return [1]
        if text == "</think>":
            return [2]
        return [99]

    def convert_tokens_to_ids(self, token):
        return {"<think>": 1, "</think>": 2}.get(token, -1)


class _DuplicateCloseToolEngine:
    tokenizer = _Tokenizer()

    def _apply_chat_template(
        self,
        messages,
        tools=None,
        chat_template_kwargs=None,
        is_partial=None,
    ):
        return "<think>"

    async def stream_chat(self, messages, **kwargs):
        yield SimpleNamespace(
            text="</think>private ",
            new_text="</think>private ",
            tool_calls=None,
            completion_tokens=3,
            prompt_tokens=10,
            cached_tokens=0,
            finished=False,
            finish_reason=None,
            prefill_duration=0.01,
            generation_duration=0.01,
        )
        yield SimpleNamespace(
            text="</think>private planning</think>",
            new_text="planning</think>",
            tool_calls=[
                {
                    "id": "call_test",
                    "name": "inspect_file",
                    "arguments": '{"path":"service.py"}',
                }
            ],
            completion_tokens=6,
            prompt_tokens=10,
            cached_tokens=0,
            finished=True,
            finish_reason="tool_calls",
            prefill_duration=0.01,
            generation_duration=0.02,
        )


@pytest.mark.asyncio
async def test_prompt_opened_tool_stream_defers_empty_initial_close():
    request = ChatCompletionRequest(
        model="test-deepseek-v4",
        messages=[Message(role="user", content="inspect")],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "inspect_file",
                    "description": "Inspect a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
    )

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_chunks = 0
    async for event in stream_chat_completion(
        _DuplicateCloseToolEngine(),
        [{"role": "user", "content": "inspect"}],
        request,
        tools=[tool.model_dump(exclude_none=True) for tool in request.tools or []],
        chat_template_kwargs={"enable_thinking": True},
    ):
        if not event.startswith("data: {"):
            continue
        payload = json.loads(event[6:])
        for choice in payload.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("tool_calls"):
                tool_chunks += 1

    assert "".join(reasoning_parts) == "private planning"
    assert "".join(content_parts) == ""
    assert tool_chunks == 1
