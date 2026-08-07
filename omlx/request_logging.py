# SPDX-License-Identifier: Apache-2.0
"""Formatting helpers for request-completion telemetry."""


def format_llm_completion_metrics(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    time_to_first_token: float,
    generation_duration: float,
    total_duration: float,
) -> str:
    """Format unambiguous LLM timing windows for an info-level log record.

    ``time_to_first_token`` and ``total_duration`` are end-to-end request
    intervals. ``generation_duration`` is the decode window used for generation
    throughput. Cached and uncached token counts are informational only; they do
    not imply separate cache-restoration and prefill-compute durations.
    """
    ttft = max(float(time_to_first_token), 0.0)
    generation_seconds = max(float(generation_duration), 0.0)
    total_seconds = max(float(total_duration), 0.0)
    prompt_count = max(int(prompt_tokens), 0)
    cached_count = min(max(int(cached_tokens), 0), prompt_count)
    completion_count = max(int(completion_tokens), 0)
    uncached_tokens = prompt_count - cached_count

    generation_tps = (
        completion_count / generation_seconds if generation_seconds > 0 else 0.0
    )
    end_to_end_tps = completion_count / total_seconds if total_seconds > 0 else 0.0

    return (
        f"prompt={prompt_count} cached={cached_count} "
        f"uncached={uncached_tokens}, ttft={ttft:.2f}s, "
        f"generation={completion_count} tokens/{generation_seconds:.2f}s "
        f"({generation_tps:.1f} tok/s), total={total_seconds:.2f}s "
        f"({end_to_end_tps:.1f} output tok/s e2e)"
    )
