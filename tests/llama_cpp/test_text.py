# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from unittest import mock

import pytest
from haystack import Pipeline
from outlines import samplers

from outlines_haystack.generators.llama_cpp import LlamaCppTextGenerator
from tests.utils import mock_text_func

REPO_ID = "hf_org/model-GGUF"
FILE_NAME = "model.Q4_K_M.gguf"


def test_init_default() -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.model_kwargs == {}
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {}
    assert component.generation_kwargs == {}


def test_init_different_sampler() -> None:
    component = LlamaCppTextGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        sampling_algorithm="multinomial",
        sampling_algorithm_kwargs={"temperature": 0.5},
        generation_kwargs={"max_tokens": 100},
    )
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}
    assert component.generation_kwargs == {"max_tokens": 100}


@mock.patch("outlines_haystack.generators.llama_cpp.models.llamacpp", return_value="mock_model")
def test_warm_up(mock_model: mock.Mock) -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
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
    assert isinstance(component.sampler, samplers.MultinomialSampler)


def test_run_not_warm() -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
    with pytest.raises(
        RuntimeError,
        match="The component LlamaCppTextGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


def test_to_dict() -> None:
    component = LlamaCppTextGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        model_kwargs={"n_gpu_layers": 4},
        sampling_algorithm_kwargs={"temperature": 0.5},
    )
    expected_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppTextGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "model_kwargs": {"n_gpu_layers": 4},
            "sampling_algorithm": "multinomial",
            "sampling_algorithm_kwargs": {"temperature": 0.5},
        },
    }
    assert component.to_dict() == expected_dict


def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppTextGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "model_kwargs": {"n_gpu_layers": 4},
            "sampling_algorithm_kwargs": {"temperature": 0.5},
        },
    }
    component = LlamaCppTextGenerator.from_dict(component_dict)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.model_kwargs == {"n_gpu_layers": 4}
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}


def test_pipeline() -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


def test_run() -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)

    with (
        mock.patch("outlines_haystack.generators.llama_cpp.generate.text") as mock_generate_text,
        mock.patch("outlines_haystack.generators.llama_cpp.models.llamacpp") as mock_model_llamacpp,
    ):
        mock_model_llamacpp.return_value = "MockModel"
        mock_generate_text.return_value = mock_text_func
        component.warm_up()
        response = component.run("How are you?")

    # check that the component returns the correct response
    assert isinstance(response, dict)
    assert "replies" in response
    assert isinstance(response["replies"], list)
    assert len(response["replies"]) == 1
    assert all(isinstance(reply, str) for reply in response["replies"])
    assert response["replies"][0] == "Hello world."


def test_run_empty_prompt() -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)

    with (
        mock.patch("outlines_haystack.generators.llama_cpp.generate.text") as mock_generate_text,
        mock.patch("outlines_haystack.generators.llama_cpp.models.llamacpp") as mock_model_llamacpp,
    ):
        mock_model_llamacpp.return_value = "MockModel"
        mock_generate_text.return_value = mock_text_func
        component.warm_up()
        response = component.run("")

    assert isinstance(response, dict)
    assert "replies" in response
    assert isinstance(response["replies"], list)
    assert len(response["replies"]) == 0
