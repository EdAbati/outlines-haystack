# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline

from outlines_haystack.generators.transformers import TransformersTextGenerator

if TYPE_CHECKING:
    from collections.abc import Generator

MODEL_NAME = "hf_org/some_model"


@pytest.fixture
def mock_transformers_generator_text(mock_transformers_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    mock_generator_obj.return_value = "Hello world."
    mock_transformers_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_init_default() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    assert component.model_name == MODEL_NAME
    assert component.device == "cpu"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_init_with_kwargs() -> None:
    component = TransformersTextGenerator(
        model_name=MODEL_NAME,
        device="cpu",
        model_kwargs={"torch_dtype": "float16"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    assert component.model_name == MODEL_NAME
    assert component.device == "cpu"
    assert component.model_kwargs == {"torch_dtype": "float16"}
    assert component.tokenizer_kwargs == {"padding_side": "left"}


def test_warm_up(
    mock_from_transformers: mock.MagicMock,
    mock_transformers_generator: mock.MagicMock,
    mock_auto_model: mock.MagicMock,
    mock_auto_tokenizer: mock.MagicMock,
) -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
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
    mock_transformers_generator.assert_called_once_with(mock_outlines_model)


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_run_not_warm() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    with pytest.raises(
        RuntimeError,
        match="The component TransformersTextGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_to_dict() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    expected_dict = {
        "type": "outlines_haystack.generators.transformers.TransformersTextGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "device": "cpu",
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
        "type": "outlines_haystack.generators.transformers.TransformersTextGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "device": "cpu",
        },
    }
    component = TransformersTextGenerator.from_dict(component_dict)
    assert component.model_name == MODEL_NAME
    assert component.device == "cpu"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_pipeline() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator_text", "mock_auto_model", "mock_auto_tokenizer"
)
def test_run() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    component.warm_up()
    response = component.run("How are you?")

    # check that the component returns the correct response
    assert isinstance(response, dict)
    assert "replies" in response
    assert isinstance(response["replies"], list)
    assert len(response["replies"]) == 1
    assert all(isinstance(reply, str) for reply in response["replies"])
    assert response["replies"][0] == "Hello world."


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_run_empty_prompt() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    component.warm_up()
    response = component.run("")

    assert response == {"replies": []}


@pytest.mark.usefixtures("mock_from_transformers", "mock_auto_model", "mock_auto_tokenizer")
def test_run_with_generation_kwargs(mock_transformers_generator_text: mock.MagicMock) -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    component.warm_up()
    generation_kwargs = {"max_tokens": 200, "seed": 999}
    response = component.run("How are you?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    mock_transformers_generator_text.assert_called_once_with("How are you?", **generation_kwargs)
    assert response["replies"][0] == "Hello world."
