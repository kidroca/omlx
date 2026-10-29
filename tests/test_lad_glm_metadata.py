import pytest

from omlx.patches.mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch


@pytest.fixture(autouse=True)
def _preload_glm_compat():
    apply_mlx_vlm_glm5_next_compat_patch()


def test_glm5_next_transformers_accepts_exact_mtp_mlp_suffix():
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextTextConfig,
    )

    config = Glm5NextTextConfig(
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse", "sparse"],
    )

    assert config.mlp_layer_types == ["dense", "sparse", "sparse"]

def test_glm5_next_transformers_rejects_unrelated_mlp_length_mismatch():
    from huggingface_hub.errors import StrictDataclassClassValidationError
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextTextConfig,
    )

    with pytest.raises(StrictDataclassClassValidationError):
        Glm5NextTextConfig(
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            layer_types=["linear_attention", "deepseek_sparse_attention"],
            mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        )
