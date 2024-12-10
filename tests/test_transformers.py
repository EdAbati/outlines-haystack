from unittest import mock

import pytest
from haystack import Pipeline

from outlines_haystack.generators.transformers import TransformersTextGenerator
from tests.utils import mock_text_func


def test_init_default() -> None:
    component = TransformersTextGenerator(model_name="microsoft/Phi-3-mini-4k-instruct", device="cpu")
    assert component.model_name == "microsoft/Phi-3-mini-4k-instruct"
    assert component.device == "cpu"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}
    assert component.model is None


@mock.patch("outlines_haystack.generators.transformers.models.transformers", return_value="mock_model")
def test_warm_up(mock_mlxlm: mock.Mock) -> None:
    component = TransformersTextGenerator(model_name="hf_org/fake_model", device="cpu")
    assert component.model is None
    assert not component._warmed_up
    component.warm_up()
    assert component.model == "mock_model"
    assert component._warmed_up
    mock_mlxlm.assert_called_once_with(
        model_name="hf_org/fake_model",
        device="cpu",
        model_kwargs={},
        tokenizer_kwargs={},
    )


def test_run_not_warm() -> None:
    component = TransformersTextGenerator(model_name="microsoft/Phi-3-mini-4k-instruct", device="cpu")
    with pytest.raises(
        RuntimeError,
        match="The component TransformersTextGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


def test_to_dict_error() -> None:
    component = TransformersTextGenerator(model_name="microsoft/Phi-3-mini-4k-instruct", device="cpu")
    with pytest.raises(AttributeError, match="'TransformersTextGenerator' object has no attribute 'to_dict'"):
        component.to_dict()


def test_from_dict_error() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.transformers.TransformersTextGenerator",
        "init_parameters": {
            "model_name": "microsoft/Phi-3-mini-4k-instruct",
        },
    }
    with pytest.raises(AttributeError, match="type object 'TransformersTextGenerator' has no attribute 'from_dict'"):
        TransformersTextGenerator.from_dict(component_dict)


def test_pipeline() -> None:
    component = TransformersTextGenerator(model_name="microsoft/Phi-3-mini-4k-instruct", device="cpu")
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


def test_run() -> None:
    component = TransformersTextGenerator(model_name="microsoft/Phi-3-mini-4k-instruct", device="cpu")

    with (
        mock.patch("outlines_haystack.generators.transformers.generate.text") as mock_generate_text,
        mock.patch("outlines_haystack.generators.transformers.models.transformers") as mock_model_mlxlm,
    ):
        mock_model_mlxlm.return_value = "MockModel"
        mock_generate_text.return_value = mock_text_func
        component.warm_up()
        response = component.run("How are you?")

    # check that the component returns the correct ChatMessage response
    assert isinstance(response, dict)
    assert "replies" in response
    assert isinstance(response["replies"], list)
    assert len(response["replies"]) == 1
    assert [isinstance(reply, str) for reply in response["replies"]]
