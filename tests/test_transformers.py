from outlines_haystack.generators.transformers import TransformersTextGenerator


def test_init_default() -> None:
    component = TransformersTextGenerator(model_name="microsoft/Phi-3-mini-4k-instruct", device="cpu")
    assert component.model_name == "microsoft/Phi-3-mini-4k-instruct"
    assert component.device == "cpu"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}
    assert component.model is None
