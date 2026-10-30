# LAD request-completion logging patch

Ports the behavior of historical LAD commit `a58d62a5` to the maintained source based on upstream `419f669d`, preserving newer upstream model/visible TTFT reporting.

## Behavior

- Non-diffusion chat completion logs explicitly label prompt, cached and uncached token counts, model TTFT, generation duration/rate, and total duration/end-to-end output rate.
- Streaming completion timing ends at the producer's `generated_until`, falling back to `generated_at`, then token-receipt time for engines without producer timestamps. Downstream SSE consumption delay is excluded where producer timing is available.
- Missing TTFT is reported as unavailable. Existing request/model context and streaming model/visible TTFT fields remain.
- Diffusion-specific speed formatting is retained.

The streaming completion window also feeds existing request metrics; this is not exclusively a cosmetic log-string change. End-to-end here means server request start through producer completion, not client-observed delivery completion. Receipt-time fallback cannot remove all upstream buffering or transport delay.

## Files

- `omlx/request_logging.py`: pure formatting helper.
- `omlx/server.py`: both chat completion log paths and streaming timing integration.
- `tests/test_request_logging.py`: dependency-light offline regression coverage.

## Verification

Run from the repository root:

```sh
python3 tests/test_request_logging.py -v
git diff --check
```

Nine offline tests cover both log paths, cache counts, distinct timing windows, absent TTFT, producer timestamp precedence, receipt fallback, empty terminal chunks, slow consumer exclusion and diffusion formatting. Tests execute the actual server AST blocks without importing the server/MLX fixtures. Failing regressions were observed before implementation.

This does not qualify full server integration or real-model behavior. No local model loading, inference, runtime build, installation or service restart was performed. A source commit/push does not deploy this patch; runtime promotion requires separate approval.
