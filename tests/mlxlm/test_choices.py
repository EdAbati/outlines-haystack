# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline
from outlines.types import Choice

from outlines_haystack.generators.mlxlm import MLXLMChoiceGenerator
from tests.utils import CHOICES

if TYPE_CHECKING:
    from collections.abc import Generator

MODEL_NAME = "mlx-community/some_model"


@pytest.fixture
def mock_mlxlm_generator_choices(mock_mlxlm_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    mock_generator_obj.return_value = "yes"
    mock_mlxlm_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_init_default() -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    assert component.model_name == MODEL_NAME
    assert component.tokenizer_config == {}
    assert component.model_config == {}
    assert component.adapter_path is None
    assert not component.lazy
    assert component.choices == CHOICES


def test_init_invalid_choices() -> None:
    with pytest.raises(ValueError, match="Choices must be a list of strings"):
        MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=[1, 2, 3])


def test_warm_up(
    mock_from_mlxlm: mock.MagicMock,
    mock_mlxlm_generator: mock.MagicMock,
    mock_mlx_lm_load: mock.MagicMock,
) -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
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
    # Verify Generator was called with the model and Choice type
    called_args = mock_mlxlm_generator.call_args
    assert called_args[0][0] == mock_outlines_model
    assert isinstance(called_args[0][1], Choice)


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_run_not_warm() -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    with pytest.raises(
        RuntimeError,
        match="The component MLXLMChoiceGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_to_dict() -> None:
    component = MLXLMChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
        tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
    )
    expected_dict = {
        "type": "outlines_haystack.generators.mlxlm.MLXLMChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
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
        "type": "outlines_haystack.generators.mlxlm.MLXLMChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
            "tokenizer_config": {"eos_token": "<|endoftext|>", "trust_remote_code": True},
        },
    }
    component = MLXLMChoiceGenerator.from_dict(component_dict)
    assert component.model_name == MODEL_NAME
    assert component.choices == CHOICES
    assert component.tokenizer_config == {"eos_token": "<|endoftext|>", "trust_remote_code": True}
    assert component.model_config == {}
    assert component.adapter_path is None
    assert not component.lazy


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_pipeline() -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator_choices", "mock_mlx_lm_load")
def test_run() -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    component.warm_up()
    response = component.run("What do you think?")

    assert isinstance(response, dict)
    assert "choice" in response
    assert response["choice"] == "yes"


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_run_empty_prompt() -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    component.warm_up()
    response = component.run("")

    assert response == {"choice": ""}


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlx_lm_load")
def test_run_with_generation_kwargs(mock_mlxlm_generator_choices: mock.MagicMock) -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    component.warm_up()
    generation_kwargs = {"max_tokens": 50, "seed": 123}
    response = component.run("What do you think?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    mock_mlxlm_generator_choices.assert_called_once_with("What do you think?", **generation_kwargs)
    assert response["choice"] == "yes"
