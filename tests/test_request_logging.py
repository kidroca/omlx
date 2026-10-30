# SPDX-License-Identifier: Apache-2.0
"""Offline regressions: python3 tests/test_request_logging.py -v.

Execute actual server AST statements without importing server or conftest:
those imports install MLX workarounds and import GPU/model dependencies.
This is timing/logging coverage, not an end-to-end server/engine test.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import unittest

ROOT = Path(__file__).resolve().parents[1]


def server_tree():
    return ast.parse((ROOT / "omlx/server.py").read_text())


def find_function(name):
    return next(
        node for node in ast.walk(server_tree())
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "server.py", "exec"), namespace)


def logging_namespace():
    namespace = {"Optional": Optional}
    helper = ROOT / "omlx/request_logging.py"
    exec(compile(helper.read_text(), str(helper), "exec"), namespace)
    for name in ("_format_generation_speed_for_log", "_format_chat_completion_metrics_for_log"):
        execute([find_function(name)], namespace)
    return namespace


def stream_timing(outputs, *, diffusion=False):
    """Run real timing blocks with producer receipts then a delayed consumer."""
    stream = find_function("stream_chat_completion")
    loop = next(node for node in ast.walk(stream) if isinstance(node, ast.AsyncFor))
    first = next(
        i for i, node in enumerate(loop.body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "first_token_time is None"
    )
    last = next(
        i for i, node in enumerate(loop.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "last_output"
    )
    completion = next(
        node for node in stream.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "last_output and last_output.finished"
    )
    end = next(
        i for i, node in enumerate(completion.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
        and [target.id for target in node.targets[0].elts] == ["metric_prefill_duration", "metric_gen_duration"]
    )
    clock = [114.51]
    namespace = dict(
        start_time=100.0, first_token_time=None, first_visible_time=120.0,
        last_token_time=None, last_completion_tokens=0,
        engine=SimpleNamespace(is_diffusion_model=diffusion),
        time=SimpleNamespace(perf_counter=lambda: clock[0]),
    )
    for index, output in enumerate(outputs):
        clock[0] = [114.51, 119.81, 700.0][index]
        namespace["output"] = output
        execute(loop.body[first:last + 1], namespace)
    clock[0] = 999.0
    execute(completion.body[:end], namespace)
    return namespace


def producer_outputs():
    return [
        SimpleNamespace(
            completion_tokens=count, new_text=text, generated_at=stamp,
            generated_until=stamp, first_token_at=114.51, finished=finished,
        )
        for count, text, stamp, finished in [(1, "", 114.51, False), (80, "answer", 119.81, True)]
    ]


class RequestLoggingTests(unittest.TestCase):
    def test_semantic_formatter_labels_distinct_windows(self):
        helper = ROOT / "omlx/request_logging.py"
        self.assertTrue(helper.exists(), "semantic metrics formatter is missing")
        formatter = logging_namespace()["format_llm_completion_metrics"]
        self.assertEqual(
            formatter(
                prompt_tokens=147186, cached_tokens=144640, completion_tokens=80,
                time_to_first_token=14.51, generation_duration=5.30, total_duration=19.81,
            ),
            "prompt=147186 cached=144640 uncached=2546, ttft=14.51s, "
            "generation=80 tokens/5.30s (15.1 tok/s), total=19.81s (4.0 output tok/s e2e)",
        )

    def test_formatter_handles_missing_ttft_zero_and_negative_values(self):
        formatter = logging_namespace()["format_llm_completion_metrics"]
        self.assertIn(
            "ttft=unavailable",
            formatter(
                prompt_tokens=10, cached_tokens=20, completion_tokens=0,
                time_to_first_token=None, generation_duration=0, total_duration=0,
            ),
        )
        for cached, expected in [(20, "cached=10 uncached=0"), (-1, "cached=0 uncached=10")]:
            with self.subTest(cached=cached):
                text = formatter(
                    prompt_tokens=10, cached_tokens=cached, completion_tokens=-1,
                    time_to_first_token=-1, generation_duration=-1, total_duration=-1,
                )
                self.assertIn(expected, text)
                self.assertIn("ttft=0.00s, generation=0 tokens/0.00s (0.0 tok/s)", text)
                self.assertIn("total=0.00s (0.0 output tok/s e2e)", text)

    def test_both_chat_paths_emit_semantic_metrics_and_keep_request_context(self):
        for name in ("_build_chat_completion", "stream_chat_completion"):
            with self.subTest(path=name):
                function = find_function(name)
                metric_assignment = next(
                    node for node in ast.walk(function)
                    if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "completion_metrics"
                )
                log = next(
                    node for node in ast.walk(function)
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "logger.info"
                    and "Chat completion:" in ast.unparse(node)
                )
                logs = []
                output = SimpleNamespace(prompt_tokens=100, cached_tokens=90, completion_tokens=10,
                                         finish_reason="stop")
                namespace = logging_namespace()
                namespace.update(
                    output=output, last_output=output, elapsed=5.0, total_duration=5.0,
                    ttft=1.0, model_ttft=1.0, metric_gen_duration=4.0, is_diffusion=False,
                    resolved_model="test-model", finish_reason="stop", max_tokens=20,
                    request=SimpleNamespace(max_tokens=20), kwargs={"max_tokens": 20},
                    model_ttft_text="1.00s", visible_ttft_text="2.00s",
                    logger=SimpleNamespace(info=logs.append),
                )
                execute([metric_assignment, log], namespace)
                self.assertEqual(len(logs), 1)
                self.assertIn("model=test-model, prompt=100 cached=90 uncached=10", logs[0])
                self.assertIn("ttft=1.00s, generation=10 tokens/4.00s (2.5 tok/s)", logs[0])
                self.assertIn("total=5.00s (2.0 output tok/s e2e)", logs[0])
                self.assertIn("finish_reason=stop, max_tokens=20, request_max_tokens=20", logs[0])
                if name == "stream_chat_completion":
                    self.assertIn("stream_model_ttft=1.00s, stream_visible_ttft=2.00s", logs[0])

    def test_diffusion_keeps_modality_specific_speeds(self):
        wrapper = logging_namespace()["_format_chat_completion_metrics_for_log"]
        output = SimpleNamespace(
            prompt_tokens=20, completion_tokens=50, generation_tps=12.5,
            diffusion_canvas_tps=8.0, prompt_tps=40.0, diffusion_work_tps=6.0,
            diffusion_denoising_steps=4,
        )
        self.assertEqual(
            wrapper(output, total_duration=5.0, time_to_first_token=1.0,
                    generation_duration=4.0, is_diffusion=True),
            "50 tokens in 5.00s (10.0 tok/s e2e, output=12.5 tok/s, "
            "canvas=8.0 tok/s, prompt=40.0 tok/s, work=6.0 tok/s, steps=4), prompt: 20",
        )
        self.assertEqual(stream_timing(producer_outputs(), diffusion=True)["total_duration"], 899.0)

    def test_stream_excludes_consumer_delay_preserving_model_and_visible_ttft(self):
        namespace = stream_timing(producer_outputs())
        self.assertAlmostEqual(namespace["total_duration"], 19.81)
        self.assertAlmostEqual(namespace["gen_duration"], 5.30)
        self.assertAlmostEqual(namespace["model_ttft"], 14.51)
        self.assertEqual(namespace["visible_ttft"], 20.0)

    def test_stream_prefers_first_token_at_and_generated_until(self):
        outputs = producer_outputs()
        outputs[0].generated_at = 115.0
        outputs[1].generated_at = 118.0
        namespace = stream_timing(outputs)
        self.assertAlmostEqual(namespace["model_ttft"], 14.51)
        self.assertAlmostEqual(namespace["gen_duration"], 5.30)

    def test_stream_falls_back_to_generated_at(self):
        outputs = producer_outputs()
        for output in outputs:
            del output.first_token_at
            del output.generated_until
        namespace = stream_timing(outputs)
        self.assertAlmostEqual(namespace["total_duration"], 19.81)
        self.assertAlmostEqual(namespace["model_ttft"], 14.51)

    def test_stream_receipt_fallback_ignores_empty_terminal_chunk(self):
        outputs = [
            SimpleNamespace(completion_tokens=count, new_text="", finished=finished)
            for count, finished in [(1, False), (80, False), (80, True)]
        ]
        namespace = stream_timing(outputs)
        self.assertAlmostEqual(namespace["total_duration"], 19.81)
        self.assertAlmostEqual(namespace["gen_duration"], 5.30)

    def test_stream_no_tokens_keeps_ttft_unavailable(self):
        namespace = stream_timing([SimpleNamespace(completion_tokens=0, new_text="", finished=True)])
        self.assertIsNone(namespace["model_ttft"])
        self.assertEqual(namespace["total_duration"], 899.0)


if __name__ == "__main__":
    unittest.main()
