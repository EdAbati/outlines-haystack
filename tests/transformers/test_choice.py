# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline
from outlines.types import Choice

from outlines_haystack.generators.transformers import TransformersChoiceGenerator
from tests.utils import CHOICES

if TYPE_CHECKING:
    from collections.abc import Generator

MODEL_NAME = "hf_org/some_model"


@pytest.fixture
def mock_transformers_generator_choices(mock_transformers_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    mock_generator_obj.return_value = "yes"
    mock_transformers_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_init_default() -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES, device="cpu")
    assert component.model_name == MODEL_NAME
    assert component.choices == CHOICES
    assert component.device == "cpu"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_init_with_kwargs() -> None:
    component = TransformersChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
        device="cpu",
        model_kwargs={"torch_dtype": "float16"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    assert component.model_kwargs == {"torch_dtype": "float16"}
    assert component.tokenizer_kwargs == {"padding_side": "left"}


def test_init_invalid_choices() -> None:
    with pytest.raises(ValueError, match="Choices must be a list of strings"):
        TransformersChoiceGenerator(model_name=MODEL_NAME, choices=[1, 2, 3])


def test_warm_up(
    mock_from_transformers: mock.MagicMock,
    mock_transformers_generator: mock.MagicMock,
    mock_auto_model: mock.MagicMock,
    mock_auto_tokenizer: mock.MagicMock,
) -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES, device="cpu")
    assert component.model is None
    assert component._generator is None
    assert not component._warmed_up

    # Setup mocks
    mock_hf_model = mock.MagicMock()
    mock_hf_tokenizer = mock.MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_hf_model
    mock_auto_tokenizer.from_pretrained.return_value = mock_hf_tokenizer
    mock_outlines_model = mock.MagicMock()
    mock_from_transformers.return_value = mock_outlines_model

    component.warm_up()

    assert component._warmed_up
    mock_auto_model.from_pretrained.assert_called_once_with(MODEL_NAME, device_map="cpu")
    mock_auto_tokenizer.from_pretrained.assert_called_once_with(MODEL_NAME)
    mock_from_transformers.assert_called_once_with(model=mock_hf_model, tokenizer_or_processor=mock_hf_tokenizer)
    assert component.model == mock_outlines_model
    # Verify Generator was called with the model and Choice type
    called_args = mock_transformers_generator.call_args
    assert called_args[0][0] == mock_outlines_model
    assert isinstance(called_args[0][1], Choice)


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_run_not_warm() -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    with pytest.raises(
        RuntimeError,
        match="The component TransformersChoiceGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_to_dict() -> None:
    component = TransformersChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
        device="cuda",
    )
    expected_dict = {
        "type": "outlines_haystack.generators.transformers.TransformersChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
            "device": "cuda",
            "model_kwargs": {},
            "tokenizer_kwargs": {},
        },
    }
    assert component.to_dict() == expected_dict


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.transformers.TransformersChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
            "device": "cuda",
        },
    }
    component = TransformersChoiceGenerator.from_dict(component_dict)
    assert component.model_name == MODEL_NAME
    assert component.choices == CHOICES
    assert component.device == "cuda"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_pipeline() -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator_choices", "mock_auto_model", "mock_auto_tokenizer"
)
def test_run() -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    component.warm_up()
    response = component.run("What do you think?")

    assert isinstance(response, dict)
    assert "choice" in response
    assert response["choice"] == "yes"


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_run_empty_prompt() -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    component.warm_up()
    response = component.run("")

    assert response == {"choice": ""}


@pytest.mark.usefixtures("mock_from_transformers", "mock_auto_model", "mock_auto_tokenizer")
def test_run_with_generation_kwargs(mock_transformers_generator_choices: mock.MagicMock) -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    component.warm_up()
    generation_kwargs = {"max_tokens": 50, "seed": 123}
    response = component.run("What do you think?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    mock_transformers_generator_choices.assert_called_once_with("What do you think?", **generation_kwargs)
    assert response["choice"] == "yes"
