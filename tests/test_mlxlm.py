from unittest import mock

import pytest
from haystack import Pipeline

from outlines_haystack.generators.mlxlm import MLXLMTextGenerator
from tests.utils import mock_text_func


def test_init_default() -> None:
    component = MLXLMTextGenerator(model_name="mlx-community/Phi-3-mini-4k-instruct-8bit")
    assert component.model_name == "mlx-community/Phi-3-mini-4k-instruct-8bit"
    assert component.tokenizer_config == {}
    assert component.model_config == {}
    assert component.adapter_path is None
    assert not component.lazy
    assert not component._warmed_up


@mock.patch("outlines_haystack.generators.mlxlm.models.mlxlm", return_value="mock_model")
def test_warm_up(mock_mlxlm: mock.Mock) -> None:
    component = MLXLMTextGenerator(model_name="mlx-community/fake_model")
    assert component.model is None
    assert not component._warmed_up
    component.warm_up()
    assert component.model == "mock_model"
    assert component._warmed_up
    mock_mlxlm.assert_called_once_with(
        model_name="mlx-community/fake_model",
        tokenizer_config={},
        model_config={},
        adapter_path=None,
        lazy=False,
    )


def test_run_not_warm() -> None:
    component = MLXLMTextGenerator(model_name="mlx-community/fake_model")
    with pytest.raises(
        RuntimeError,
        match="The component MLXLMTextGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


def test_to_dict_error() -> None:
    component = MLXLMTextGenerator(
        model_name="mlx-community/some-model",
        tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
    )
    with pytest.raises(AttributeError, match="'MLXLMTextGenerator' object has no attribute 'to_dict'"):
        component.to_dict()


def test_from_dict_error() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.mlxlm.MLXLMTextGenerator",
        "init_parameters": {
            "model_name": "mlx-community/some-model",
            "tokenizer_config": {"eos_token": "<|endoftext|>", "trust_remote_code": True},
        },
    }
    with pytest.raises(AttributeError, match="type object 'MLXLMTextGenerator' has no attribute 'from_dict'"):
        MLXLMTextGenerator.from_dict(component_dict)


def test_pipeline() -> None:
    component = MLXLMTextGenerator(model_name="mlx-community/Phi-3-mini-4k-instruct-8bit")
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


def test_run() -> None:
    component = MLXLMTextGenerator(model_name="mlx-community/Phi-3-mini-4k-instruct-8bit")

    with (
        mock.patch("outlines_haystack.generators.mlxlm.generate.text") as mock_generate_text,
        mock.patch("outlines_haystack.generators.mlxlm.models.mlxlm") as mock_model_mlxlm,
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
