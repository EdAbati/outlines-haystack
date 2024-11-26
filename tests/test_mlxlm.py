from outlines_haystack.generators.mlxlm import MLXLMTextGenerator


def test_init_default() -> None:
    component = MLXLMTextGenerator(model_name="mlx-community/Phi-3-mini-4k-instruct-8bit")
    assert component.model_name == "mlx-community/Phi-3-mini-4k-instruct-8bit"
    assert component.tokenizer_config == {}
    assert component.model_config == {}
    assert component.adapter_path is None
    assert not component.lazy
