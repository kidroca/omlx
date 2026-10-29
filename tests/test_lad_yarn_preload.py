"""No checkpoint weights or inference: real pre-load hooks and rotary kernels only."""
import dataclasses
import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.rope_utils import initialize_rope

from omlx.utils.model_loading import maybe_apply_pre_load_patches
from test_mlx_vlm_qwen4_exp_compat import _tiny_config


@pytest.mark.parametrize("factor", [1, 2])
def test_flashnext_preload_rotary_reference(tmp_path, factor):
    import mlx_vlm.utils as utils
    from omlx.patches.qwen35_yarn import set_context_window

    config = _tiny_config()
    raw = dataclasses.asdict(config)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    original = dict(raw["text_config"]["rope_parameters"])
    try:
        maybe_apply_pre_load_patches(
            str(tmp_path), SimpleNamespace(max_context_window=128 * factor), for_vlm=True
        )
        loaded = utils.load_config(tmp_path)
        rope = loaded["text_config"]["rope_parameters"]
        from mlx_vlm.models.qwen4_exp import TextConfig
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttention

        attn = Qwen4ExpAttention(TextConfig.from_dict(loaded["text_config"]))
        rotary = attn.rotary_emb
        assert attn.indexer.rotary_emb is rotary
        native_freq = 1.0 / (rotary.base ** (mx.arange(0, rotary.dim, 2) / rotary.dim))
        if factor == 1:
            assert rope == original
            assert rotary.attention_scaling == 1.0
            assert mx.allclose(rotary.inv_freq, native_freq).item()
        else:
            assert rope["factor"] == 2
            ref = initialize_rope(rotary.dim, rotary.base, False, rope, 128)
            assert mx.allclose(rotary.inv_freq, 1.0 / ref._freqs).item()
            assert not mx.allclose(rotary.inv_freq, native_freq).item()
            assert rotary.attention_scaling == ref.mscale > 1
            q = mx.ones((1, 4, 3, 8))
            k = mx.ones((1, 2, 3, 8))
            positions = [129, 200, 255]
            ids = mx.broadcast_to(mx.array([positions])[None], (3, 1, 3))
            with mx.stream(mx.cpu):
                got, _ = rotary.apply_rotary(q, k, ids, cast_output=False)
                expected = mx.concatenate([ref(q[:, :, i:i+1], offset=p) for i, p in enumerate(positions)], axis=2)
                assert mx.allclose(got, expected, atol=1e-5).item()
    finally:
        set_context_window(tmp_path, None)


def test_factor_two_fused_kernel_parity(monkeypatch):
    from test_qwen4_attn_fused import _YARN_CONFIG, test_fp32_prep_matches_norms_and_mrope
    config = dataclasses.replace(_YARN_CONFIG, rope_parameters={**_YARN_CONFIG.rope_parameters, "factor": 2.0})
    test_fp32_prep_matches_norms_and_mrope(1, 3, config)


def test_final_fused_prologue_passes_yarn_scaling(monkeypatch):
    from mlx_vlm.models.qwen4_exp import language
    from mlx_vlm.models.qwen4_exp import attn_fused

    config = _tiny_config().text_config
    config.rope_parameters = {**config.rope_parameters, "type": "yarn", "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 128}
    attn = language.Qwen4ExpAttention(config)
    seen = []

    def prep(*args, **kwargs):
        seen.append(kwargs["rope_scale"])
        return "q", "k"

    monkeypatch.setattr(attn_fused, "prep_qk", prep)
    monkeypatch.setattr(language.row_exact_qmv, "quantized_linears_tiled", lambda modules, x, *args: tuple(m(x) for m in modules))
    # Only the projection/prologue is exercised, not attention or generation.
    positions = mx.zeros((3, 1, 1), dtype=mx.int32)
    attn._fused_prologue(mx.ones((1, 1, 32)), 0, positions)
    assert seen == [attn.rotary_emb.attention_scaling]
    assert seen[0] > 1
