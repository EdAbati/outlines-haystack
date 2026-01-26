# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline

from outlines_haystack.generators.llama_cpp import LlamaCppTextGenerator

if TYPE_CHECKING:
    from collections.abc import Generator

REPO_ID = "hf_org/model-GGUF"
FILE_NAME = "model.Q4_K_M.gguf"


@pytest.fixture
def mock_llamacpp_generator_text(mock_llamacpp_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    mock_generator_obj.return_value = "Hello world."
    mock_llamacpp_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_init_default() -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.model_kwargs == {}


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_init_with_kwargs() -> None:
    component = LlamaCppTextGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        model_kwargs={"n_gpu_layers": 4},
    )
    assert component.model_kwargs == {"n_gpu_layers": 4}


def test_warm_up(
    mock_from_llamacpp: mock.MagicMock,
    mock_llamacpp_generator: mock.MagicMock,
    mock_llama: mock.MagicMock,
) -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
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
    mock_llamacpp_generator.assert_called_once_with(mock_outlines_model)


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_run_not_warm() -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
    with pytest.raises(
        RuntimeError,
        match="The component LlamaCppTextGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_to_dict() -> None:
    component = LlamaCppTextGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        model_kwargs={"n_gpu_layers": 4},
    )
    expected_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppTextGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "model_kwargs": {"n_gpu_layers": 4},
        },
    }
    assert component.to_dict() == expected_dict


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppTextGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "model_kwargs": {"n_gpu_layers": 4},
        },
    }
    component = LlamaCppTextGenerator.from_dict(component_dict)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.model_kwargs == {"n_gpu_layers": 4}


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_pipeline() -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator_text", "mock_llama")
def test_run() -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
    component.warm_up()
    response = component.run("How are you?")

    # check that the component returns the correct response
    assert isinstance(response, dict)
    assert "replies" in response
    assert isinstance(response["replies"], list)
    assert len(response["replies"]) == 1
    assert all(isinstance(reply, str) for reply in response["replies"])
    assert response["replies"][0] == "Hello world."


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_run_empty_prompt() -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
    component.warm_up()
    response = component.run("")

    assert response == {"replies": []}


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llama")
def test_run_with_generation_kwargs(mock_llamacpp_generator_text: mock.MagicMock) -> None:
    component = LlamaCppTextGenerator(repo_id=REPO_ID, file_name=FILE_NAME)
    component.warm_up()
    generation_kwargs = {"max_tokens": 200, "seed": 999}
    response = component.run("How are you?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    mock_llamacpp_generator_text.assert_called_once_with("How are you?", **generation_kwargs)
    assert response["replies"][0] == "Hello world."
