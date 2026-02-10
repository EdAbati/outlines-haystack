# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline

from outlines_haystack.generators.llama_cpp import LlamaCppJSONGenerator
from tests.utils import User, user_schema_str

if TYPE_CHECKING:
    from collections.abc import Generator

REPO_ID = "TheBloke/Llama-2-7B-GGUF"
FILE_NAME = "llama-2-7b.Q4_K_M.gguf"


@pytest.fixture
def mock_llamacpp_generator_json(mock_llamacpp_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    json_string = json.dumps({"name": "John"})
    mock_generator_obj.return_value = json_string
    mock_llamacpp_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.fixture
def mock_llamacpp_generator_json_jane(mock_llamacpp_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    json_string = json.dumps({"name": "Jane"})
    mock_generator_obj.return_value = json_string
    mock_llamacpp_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_init_default() -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.schema_object == user_schema_str
    assert component.model_kwargs == {}


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_init_with_string_schema() -> None:
    schema_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=schema_str)
    assert component.schema_object == schema_str


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_init_with_kwargs() -> None:
    component = LlamaCppJSONGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        schema_object=User,
        model_kwargs={"n_gpu_layers": 4},
    )
    assert component.model_kwargs == {"n_gpu_layers": 4}


def test_warm_up(
    mock_from_llamacpp: mock.MagicMock,
    mock_llamacpp_generator: mock.MagicMock,
    mock_llama: mock.MagicMock,
) -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
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
    # Verify Generator was called with the model and the output type (Pydantic model)
    mock_llamacpp_generator.assert_called_once_with(mock_outlines_model, User)


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_run_not_warm() -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    with pytest.raises(
        RuntimeError,
        match="The component LlamaCppJSONGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_to_dict() -> None:
    component = LlamaCppJSONGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        schema_object=User,
        model_kwargs={"n_gpu_layers": 4},
    )
    expected_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppJSONGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "schema_object": user_schema_str,
            "model_kwargs": {"n_gpu_layers": 4},
        },
    }
    assert component.to_dict() == expected_dict


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppJSONGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "schema_object": user_schema_str,
            "model_kwargs": {"n_gpu_layers": 4},
        },
    }
    component = LlamaCppJSONGenerator.from_dict(component_dict)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.schema_object == user_schema_str
    assert component.model_kwargs == {"n_gpu_layers": 4}


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_pipeline() -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator_json", "mock_llama")
def test_run() -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    component.warm_up()
    response = component.run("Create a user profile for John")

    # check that the component returns the correct response
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"], list)
    assert len(response["structured_replies"]) == 1
    assert all(isinstance(reply, str) for reply in response["structured_replies"])
    json_string = json.dumps({"name": "John"})
    assert response["structured_replies"][0] == json_string


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator_json_jane", "mock_llama")
def test_run_with_string_return() -> None:
    # Test case where Generator returns a JSON string
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    component.warm_up()
    response = component.run("Create a user profile for Jane")

    json_string = json.dumps({"name": "Jane"})
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"][0], str)
    assert response["structured_replies"][0] == json_string


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llamacpp_generator", "mock_llama")
def test_run_empty_prompt() -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    component.warm_up()
    response = component.run("")

    assert response == {"structured_replies": []}


@pytest.mark.usefixtures("mock_from_llamacpp", "mock_llama")
def test_run_with_generation_kwargs(mock_llamacpp_generator_json: mock.MagicMock) -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    component.warm_up()
    generation_kwargs = {"max_tokens": 100, "seed": 42}
    response = component.run("Create a user profile for John", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    json_string = json.dumps({"name": "John"})
    mock_llamacpp_generator_json.assert_called_once_with("Create a user profile for John", **generation_kwargs)
    assert response["structured_replies"][0] == json_string
