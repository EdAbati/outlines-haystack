# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline

from outlines_haystack.generators.mlxlm import MLXLMJSONGenerator
from tests.utils import User, user_schema_str

if TYPE_CHECKING:
    from collections.abc import Generator

MODEL_NAME = "mlx-community/some_model"


@pytest.fixture
def mock_mlxlm_generator_json(mock_mlxlm_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    json_string = json.dumps({"name": "John"})
    mock_generator_obj.return_value = json_string
    mock_mlxlm_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.fixture
def mock_mlxlm_generator_json_jane(mock_mlxlm_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    json_string = json.dumps({"name": "Jane"})
    mock_generator_obj.return_value = json_string
    mock_mlxlm_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_init_default() -> None:
    component = MLXLMJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    assert component.model_name == MODEL_NAME
    assert component.tokenizer_config == {}
    assert component.model_config == {}
    assert component.adapter_path is None
    assert not component.lazy
    assert component.schema_object == user_schema_str


def test_warm_up(
    mock_from_mlxlm: mock.MagicMock,
    mock_mlxlm_generator: mock.MagicMock,
    mock_mlx_lm_load: mock.MagicMock,
) -> None:
    component = MLXLMJSONGenerator(model_name=MODEL_NAME, schema_object=User)
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
    # Verify Generator was called with the model and the output type (Pydantic model)
    mock_mlxlm_generator.assert_called_once_with(mock_outlines_model, User)


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_run_not_warm() -> None:
    component = MLXLMJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    with pytest.raises(
        RuntimeError,
        match="The component MLXLMJSONGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_to_dict() -> None:
    component = MLXLMJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
    )
    expected_dict = {
        "type": "outlines_haystack.generators.mlxlm.MLXLMJSONGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "schema_object": user_schema_str,
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
        "type": "outlines_haystack.generators.mlxlm.MLXLMJSONGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "schema_object": user_schema_str,
            "tokenizer_config": {"eos_token": "<|endoftext|>", "trust_remote_code": True},
        },
    }
    component = MLXLMJSONGenerator.from_dict(component_dict)
    assert component.model_name == MODEL_NAME
    assert component.schema_object == user_schema_str
    assert component.tokenizer_config == {"eos_token": "<|endoftext|>", "trust_remote_code": True}
    assert component.model_config == {}
    assert component.adapter_path is None
    assert not component.lazy


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_pipeline() -> None:
    component = MLXLMJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator_json", "mock_mlx_lm_load")
def test_run() -> None:
    component = MLXLMJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    component.warm_up()
    response = component.run("How are you?")

    # check that the component returns the correct response
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"], list)
    assert len(response["structured_replies"]) == 1
    assert all(isinstance(reply, str) for reply in response["structured_replies"])
    json_string = json.dumps({"name": "John"})
    assert response["structured_replies"][0] == json_string


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator_json_jane", "mock_mlx_lm_load")
def test_run_with_string_return() -> None:
    # Test case where Generator returns a JSON string
    component = MLXLMJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    component.warm_up()
    response = component.run("How are you?")

    json_string = json.dumps({"name": "Jane"})
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"][0], str)
    assert response["structured_replies"][0] == json_string


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlxlm_generator", "mock_mlx_lm_load")
def test_run_empty_prompt() -> None:
    component = MLXLMJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    component.warm_up()
    response = component.run("")

    assert response == {"structured_replies": []}


@pytest.mark.usefixtures("mock_from_mlxlm", "mock_mlx_lm_load")
def test_run_with_generation_kwargs(mock_mlxlm_generator_json: mock.MagicMock) -> None:
    component = MLXLMJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    component.warm_up()
    generation_kwargs = {"max_tokens": 100, "seed": 42}
    response = component.run("How are you?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    json_string = json.dumps({"name": "John"})
    mock_mlxlm_generator_json.assert_called_once_with("How are you?", **generation_kwargs)
    assert response["structured_replies"][0] == json_string
