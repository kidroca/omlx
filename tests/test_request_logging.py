import ast
import logging
from pathlib import Path
from types import SimpleNamespace

from omlx import server
from omlx.api.openai_models import ChatCompletionRequest
from omlx.request_logging import format_llm_completion_metrics
from omlx.server import _format_chat_completion_metrics_for_log


def test_format_llm_completion_metrics_labels_timing_windows():
    assert format_llm_completion_metrics(
        prompt_tokens=147_186,
        cached_tokens=144_640,
        completion_tokens=80,
        time_to_first_token=14.51,
        generation_duration=5.30,
        total_duration=19.81,
    ) == (
        "prompt=147186 cached=144640 uncached=2546, "
        "ttft=14.51s, generation=80 tokens/5.30s (15.1 tok/s), "
        "total=19.81s (4.0 output tok/s e2e)"
    )


def test_format_llm_completion_metrics_handles_zero_durations():
    assert format_llm_completion_metrics(
        prompt_tokens=10,
        cached_tokens=0,
        completion_tokens=0,
        time_to_first_token=0.0,
        generation_duration=0.0,
        total_duration=0.0,
    ) == (
        "prompt=10 cached=0 uncached=10, "
        "ttft=0.00s, generation=0 tokens/0.00s (0.0 tok/s), "
        "total=0.00s (0.0 output tok/s e2e)"
    )


def test_format_llm_completion_metrics_clamps_cached_tokens_to_prompt_partition():
    over_prompt = format_llm_completion_metrics(
        prompt_tokens=100,
        cached_tokens=120,
        completion_tokens=5,
        time_to_first_token=1.0,
        generation_duration=1.0,
        total_duration=2.0,
    )
    negative = format_llm_completion_metrics(
        prompt_tokens=100,
        cached_tokens=-5,
        completion_tokens=5,
        time_to_first_token=1.0,
        generation_duration=1.0,
        total_duration=2.0,
    )

    assert "prompt=100 cached=100 uncached=0" in over_prompt
    assert "prompt=100 cached=0 uncached=100" in negative


def test_chat_completion_paths_use_semantic_metrics_formatter():
    source = (Path(__file__).resolve().parents[1] / "omlx" / "server.py").read_text()
    functions = {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for function_name in ("_build_chat_completion", "stream_chat_completion"):
        calls = {
            call.func.id
            for call in ast.walk(functions[function_name])
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        assert "_format_chat_completion_metrics_for_log" in calls


def test_diffusion_chat_completion_keeps_modality_specific_speed_fields():
    output = SimpleNamespace(
        prompt_tokens=20,
        completion_tokens=50,
        generation_tps=12.5,
        diffusion_canvas_tps=8.0,
        prompt_tps=40.0,
        diffusion_work_tps=6.0,
        diffusion_denoising_steps=4,
    )

    assert _format_chat_completion_metrics_for_log(
        output,
        total_duration=5.0,
        time_to_first_token=1.0,
        generation_duration=4.0,
        is_diffusion=True,
    ) == (
        "50 tokens in 5.00s (10.0 tok/s e2e, output=12.5 tok/s, "
        "canvas=8.0 tok/s, prompt=40.0 tok/s, work=6.0 tok/s, steps=4), "
        "prompt: 20"
    )


async def test_stream_chat_logging_uses_producer_timestamps_before_visible_text(
    caplog, monkeypatch
):
    class FakeEngine:
        tokenizer = None
        is_diffusion_model = False

        async def stream_chat(self, **_kwargs):
            yield SimpleNamespace(
                new_text="",
                prompt_tokens=147_186,
                cached_tokens=144_640,
                completion_tokens=1,
                generated_at=114.51,
                generated_until=114.51,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            )
            yield SimpleNamespace(
                new_text="answer",
                prompt_tokens=147_186,
                cached_tokens=144_640,
                completion_tokens=80,
                generated_at=119.81,
                generated_until=119.81,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        max_tokens=80,
    )
    perf_counter_values = iter([100.0, 999.0])
    monkeypatch.setattr(
        server.time,
        "perf_counter",
        lambda: next(perf_counter_values, 999.0),
    )

    with caplog.at_level(logging.INFO, logger="omlx.server"):
        chunks = [
            chunk
            async for chunk in server.stream_chat_completion(
                FakeEngine(),
                request.messages,
                request,
                max_tokens=80,
            )
        ]

    assert chunks[-1] == "data: [DONE]\n\n"
    assert (
        "Chat completion: prompt=147186 cached=144640 uncached=2546, "
        "ttft=14.51s, generation=80 tokens/5.30s (15.1 tok/s), "
        "total=19.81s (4.0 output tok/s e2e)"
    ) in caplog.text
