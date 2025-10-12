# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from unittest import mock

import pytest
from haystack import Pipeline
from outlines import samplers

from outlines_haystack.generators.llama_cpp import LlamaCppChoiceGenerator
from tests.utils import CHOICES, mock_choice_func

REPO_ID = "TheBloke/Llama-2-7B-GGUF"
FILE_NAME = "llama-2-7b.Q4_K_M.gguf"


def test_init_default() -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.choices == CHOICES
    assert component.model_kwargs == {}
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {}
    assert component.generation_kwargs == {}


def test_init_invalid_choices() -> None:
    with pytest.raises(ValueError, match="Choices must be a list of strings"):
        LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=[1, 2, 3])


def test_init_different_sampler() -> None:
    component = LlamaCppChoiceGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        choices=CHOICES,
        sampling_algorithm="multinomial",
        sampling_algorithm_kwargs={"temperature": 0.5},
        generation_kwargs={"max_tokens": 100},
    )
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}
    assert component.generation_kwargs == {"max_tokens": 100}


@mock.patch("outlines_haystack.generators.llama_cpp.generate.choice", return_value="mock_generator")
@mock.patch("outlines_haystack.generators.llama_cpp.models.llamacpp", return_value="mock_model")
def test_warm_up(mock_model: mock.Mock, mock_generator: mock.Mock) -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    assert component.model is None
    assert component.sampler is None
    assert not component._warmed_up
    component.warm_up()
    assert component.model == "mock_model"
    assert component._warmed_up
    mock_model.assert_called_once_with(
        repo_id=REPO_ID,
        filename=FILE_NAME,
    )
    mock_generator.assert_called_once_with(
        "mock_model",
        choices=CHOICES,
        sampler=mock.ANY,
    )
    assert isinstance(component.sampler, samplers.MultinomialSampler)


def test_run_not_warm() -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    with pytest.raises(
        RuntimeError,
        match="The component LlamaCppChoiceGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


def test_to_dict() -> None:
    component = LlamaCppChoiceGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        choices=CHOICES,
        model_kwargs={"n_gpu_layers": 4},
        sampling_algorithm_kwargs={"temperature": 0.5},
    )
    expected_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppChoiceGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "choices": CHOICES,
            "model_kwargs": {"n_gpu_layers": 4},
            "sampling_algorithm": "multinomial",
            "sampling_algorithm_kwargs": {"temperature": 0.5},
            "generation_kwargs": {},
        },
    }
    assert component.to_dict() == expected_dict


def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppChoiceGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "choices": CHOICES,
            "model_kwargs": {"n_gpu_layers": 4},
            "sampling_algorithm": "multinomial",
            "sampling_algorithm_kwargs": {"temperature": 0.5},
        },
    }
    component = LlamaCppChoiceGenerator.from_dict(component_dict)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.choices == CHOICES
    assert component.model_kwargs == {"n_gpu_layers": 4}
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}


def test_pipeline() -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


def test_run() -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)

    with (
        mock.patch("outlines_haystack.generators.llama_cpp.generate.choice") as mock_generate_choice,
        mock.patch("outlines_haystack.generators.llama_cpp.models.llamacpp") as mock_model_llamacpp,
    ):
        mock_model_llamacpp.return_value = "MockModel"
        mock_generate_choice.return_value = mock_choice_func
        component.warm_up()
        response = component.run("What do you think?")

    # check that the component returns the correct response
    assert isinstance(response, dict)
    assert "choice" in response
    assert response["choice"] == "yes"


def test_run_empty_prompt() -> None:
    component = LlamaCppChoiceGenerator(repo_id=REPO_ID, file_name=FILE_NAME, choices=CHOICES)

    with (
        mock.patch("outlines_haystack.generators.llama_cpp.generate.choice") as mock_generate_choice,
        mock.patch("outlines_haystack.generators.llama_cpp.models.llamacpp") as mock_model_llamacpp,
    ):
        mock_model_llamacpp.return_value = "MockModel"
        mock_generate_choice.return_value = mock_choice_func
        component.warm_up()
        response = component.run("")

    assert isinstance(response, dict)
    assert "choice" in response
    assert response["choice"] == ""
