# SPDX-License-Identifier: Apache-2.0
"""Formatting helpers for request-completion telemetry."""
from __future__ import annotations


def format_llm_completion_metrics(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    time_to_first_token: float | None,
    generation_duration: float,
    total_duration: float,
) -> str:
    """Label decode and request windows without inventing cache/prefill timing.

    TTFT is producer/model TTFT, not first client-visible text. Missing producer
    timing stays unavailable, matching the server's usage timing contract.
    """
    ttft = (
        f"{max(float(time_to_first_token), 0.0):.2f}s"
        if time_to_first_token is not None
        else "unavailable"
    )
    generation_seconds = max(float(generation_duration), 0.0)
    total_seconds = max(float(total_duration), 0.0)
    prompt_count = max(int(prompt_tokens), 0)
    cached_count = min(max(int(cached_tokens), 0), prompt_count)
    completion_count = max(int(completion_tokens), 0)
    generation_tps = completion_count / generation_seconds if generation_seconds > 0 else 0.0
    end_to_end_tps = completion_count / total_seconds if total_seconds > 0 else 0.0
    return (
        f"prompt={prompt_count} cached={cached_count} uncached={prompt_count - cached_count}, "
        f"ttft={ttft}, generation={completion_count} tokens/{generation_seconds:.2f}s "
        f"({generation_tps:.1f} tok/s), total={total_seconds:.2f}s "
        f"({end_to_end_tps:.1f} output tok/s e2e)"
    )
