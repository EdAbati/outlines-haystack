# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline

from outlines_haystack.generators.mlxlm import MLXLMTextGenerator

if TYPE_CHECKING:
    from collections.abc import Generator

MODEL_NAME = "mlx-community/some_model"


@pytest.fixture
def mock_mlxlm_generator_text(mock_mlxlm_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    mock_generator_obj.return_value = "Hello world."
    mock_mlxlm_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_init_default() -> None:
    component = MLXLMTextGenerator(model_name=MODEL_NAME)
    assert component.model_name == MODEL_NAME
    assert component.tokenizer_config == {}
    assert component.model_config == {}
    assert component.adapter_path is None
    assert not component.lazy


def test_warm_up(
    mock_from_mlxlm: mock.MagicMock,
    mock_mlxlm_generator: mock.MagicMock,
    mock_mlx_lm_load: mock.MagicMock,
) -> None:
    component = MLXLMTextGenerator(model_name=MODEL_NAME)
    assert component.model is None
    assert component._generator is None
    assert not component._warmed_up

    # Setup mocks
    mock_mlx_model = mock.MagicMock()
    mock_tokenizer = mock.MagicMock()
    mock_mlx_lm_load.return_value = (mock_mlx_model, mock_tokenizer)
    mock_outlines_model = mock.MagicMock()
    mock_from_mlxlm.return_value = mock_outlines_model

    component.warm_up()

    assert component._warmed_up
    mock_mlx_lm_load.assert_called_once_with(
        MODEL_NAME,
        tokenizer_config={},
        model_config={},
        adapter_path=None,
        lazy=False,
    )
    mock_from_mlxlm.assert_called_once_with(mock_mlx_model, mock_tokenizer)
    assert component.model == mock_outlines_model
    mock_mlxlm_generator.assert_called_once_with(mock_outlines_model)


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_run_not_warm() -> None:
    component = MLXLMTextGenerator(model_name=MODEL_NAME)
    with pytest.raises(
        RuntimeError,
        match="The component MLXLMTextGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_to_dict() -> None:
    component = MLXLMTextGenerator(
        model_name=MODEL_NAME,
        tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
    )
    expected_dict = {
        "type": "outlines_haystack.generators.mlxlm.MLXLMTextGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "tokenizer_config": {"eos_token": "<|endoftext|>", "trust_remote_code": True},
            "model_config": {},
            "adapter_path": None,
            "lazy": False,
        },
    }
    assert component.to_dict() == expected_dict


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.mlxlm.MLXLMTextGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "tokenizer_config": {"eos_token": "<|endoftext|>", "trust_remote_code": True},
        },
    }
    component = MLXLMTextGenerator.from_dict(component_dict)
    assert component.model_name == MODEL_NAME
    assert component.tokenizer_config == {"eos_token": "<|endoftext|>", "trust_remote_code": True}
    assert component.model_config == {}
    assert component.adapter_path is None
    assert not component.lazy


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_pipeline() -> None:
    component = MLXLMTextGenerator(model_name=MODEL_NAME)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator_text", "mock_mlx_lm_load")
def test_run() -> None:
    component = MLXLMTextGenerator(model_name=MODEL_NAME)
    component.warm_up()
    response = component.run("How are you?")

    assert isinstance(response, dict)
    assert "replies" in response
    assert isinstance(response["replies"], list)
    assert len(response["replies"]) == 1
    assert all(isinstance(reply, str) for reply in response["replies"])
    assert response["replies"][0] == "Hello world."


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_run_empty_prompt() -> None:
    component = MLXLMTextGenerator(model_name=MODEL_NAME)
    component.warm_up()
    response = component.run("")

    assert response == {"replies": []}


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlx_lm_load")
def test_run_with_generation_kwargs(mock_mlxlm_generator_text: mock.MagicMock) -> None:
    component = MLXLMTextGenerator(model_name=MODEL_NAME)
    component.warm_up()
    generation_kwargs = {"max_tokens": 200, "seed": 999}
    response = component.run("How are you?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    mock_mlxlm_generator_text.assert_called_once_with("How are you?", **generation_kwargs)
    assert response["replies"][0] == "Hello world."
