# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline
from outlines.types import Choice

from outlines_haystack.generators.llama_cpp import LlamaCppChoiceGenerator
from tests.utils import CHOICES

if TYPE_CHECKING:
    from collections.abc import Generator

REPO_ID = "TheBloke/Llama-2-7B-GGUF"
FILE_NAME = "llama-2-7b.Q4_K_M.gguf"


@pytest.fixture
def mock_llamacpp_generator_choices(mock_llamacpp_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    mock_generator_obj.return_value = "yes"
    mock_llamacpp_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_init_default() -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.choices == CHOICES
    assert component.model_kwargs == {}


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_init_with_kwargs() -> None:
    component = LlamaCppChoiceGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        choices=CHOICES,
        model_kwargs={"n_gpu_layers": 4},
    )
    assert component.model_kwargs == {"n_gpu_layers": 4}


def test_init_invalid_choices() -> None:
    with pytest.raises(ValueError, match="Choices must be a list of strings"):
        LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=[1, 2, 3])


def test_warm_up(
    mock_from_llamacpp: mock.MagicMock,
    mock_llamacpp_generator: mock.MagicMock,
    mock_llama: mock.MagicMock,
) -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    assert component.model is None
    assert component._generator is None
    assert not component._warmed_up

    # Setup mocks
    mock_llamacpp_model = mock.MagicMock()
    mock_llama.from_pretrained.return_value = mock_llamacpp_model
    mock_outlines_model = mock.MagicMock()
    mock_from_llamacpp.return_value = mock_outlines_model

    component.warm_up()

    assert component._warmed_up
    mock_llama.from_pretrained.assert_called_once_with(
        repo_id=REPO_ID,
        filename=FILE_NAME,
    )
    mock_from_llamacpp.assert_called_once_with(mock_llamacpp_model)
    assert component.model == mock_outlines_model
    # Verify Generator was called with the model and Choice type
    called_args = mock_llamacpp_generator.call_args
    assert called_args[0][0] == mock_outlines_model
    assert isinstance(called_args[0][1], Choice)


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_run_not_warm() -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    with pytest.raises(
        RuntimeError,
        match="The component LlamaCppChoiceGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_to_dict() -> None:
    component = LlamaCppChoiceGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        choices=CHOICES,
        model_kwargs={"n_gpu_layers": 4},
    )
    expected_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppChoiceGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "choices": CHOICES,
            "model_kwargs": {"n_gpu_layers": 4},
        },
    }
    assert component.to_dict() == expected_dict


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppChoiceGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "choices": CHOICES,
            "model_kwargs": {"n_gpu_layers": 4},
        },
    }
    component = LlamaCppChoiceGenerator.from_dict(component_dict)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.choices == CHOICES
    assert component.model_kwargs == {"n_gpu_layers": 4}


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_pipeline() -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator_choices", "mock_llama")
def test_run() -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    component.warm_up()
    response = component.run("What do you think?")

    assert isinstance(response, dict)
    assert "choice" in response
    assert response["choice"] == "yes"


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_run_empty_prompt() -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    component.warm_up()
    response = component.run("")

    assert response == {"choice": ""}


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llama")
def test_run_with_generation_kwargs(mock_llamacpp_generator_choices: mock.MagicMock) -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    component.warm_up()
    generation_kwargs = {"max_tokens": 50, "seed": 123}
    response = component.run("What do you think?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    mock_llamacpp_generator_choices.assert_called_once_with("What do you think?", **generation_kwargs)
    assert response["choice"] == "yes"
