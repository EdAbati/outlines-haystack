# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

import json
import os
from contextlib import nullcontext
from unittest import mock

import pytest
from haystack import Pipeline
from haystack.utils import Secret

from outlines_haystack.generators.openai import OpenAIJSONGenerator
from tests.utils import User, user_schema_str

MODEL_NAME = "gpt-4o-mini"


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
@mock.patch("outlines_haystack.generators.openai.from_openai")
@mock.patch("outlines_haystack.generators.openai.Generator")
def test_init_default(mock_generator: mock.MagicMock, mock_from_openai: mock.MagicMock) -> None:
    component = OpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    assert component.model_name == MODEL_NAME
    assert component.api_key.resolve_value() == "test-api-key"
    assert component.organization is None
    assert component.project is None
    assert component.base_url is None
    assert component.timeout == 30
    assert component.max_retries == 5
    assert component.default_headers is None
    assert component.default_query is None
    assert component.schema_object == user_schema_str
    mock_from_openai.assert_called_once()
    mock_generator.assert_called_once()


@mock.patch("outlines_haystack.generators.openai.from_openai")
@mock.patch("outlines_haystack.generators.openai.Generator")
def test_init_params(_mock_generator: mock.MagicMock, _mock_from_openai: mock.MagicMock) -> None:
    component = OpenAIJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        api_key=Secret.from_token("test-api-key"),
        timeout=60,
        max_retries=10,
        default_headers={"test-header": "test-value"},
    )
    assert component.model_name == MODEL_NAME
    assert component.api_key.resolve_value() == "test-api-key"
    assert component.organization is None
    assert component.project is None
    assert component.timeout == 60
    assert component.max_retries == 10
    assert component.default_headers == {"test-header": "test-value"}
    assert component.default_query is None
    assert component.schema_object == user_schema_str


def test_init_value_error() -> None:
    with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
        OpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
@mock.patch("outlines_haystack.generators.openai.from_openai")
@mock.patch("outlines_haystack.generators.openai.Generator")
def test_to_dict(_mock_generator: mock.MagicMock, _mock_from_openai: mock.MagicMock) -> None:
    component = OpenAIJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        timeout=60,
        max_retries=10,
        default_headers={"test-header": "test-value"},
    )
    assert component.to_dict() == {
        "type": "outlines_haystack.generators.openai.OpenAIJSONGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "schema_object": user_schema_str,
            "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
            "organization": None,
            "project": None,
            "base_url": None,
            "timeout": 60,
            "max_retries": 10,
            "default_headers": {"test-header": "test-value"},
            "default_query": None,
        },
    }


@pytest.mark.parametrize(
    "mock_os_environ",
    [
        {},
        {
            "OPENAI_API_KEY": "test-api-key",
        },
    ],
)
def test_from_dict(mock_os_environ: dict[str, str]) -> None:
    component_dict = {
        "type": "outlines_haystack.generators.openai.OpenAIJSONGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "schema_object": user_schema_str,
            "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
            "organization": None,
            "project": None,
            "base_url": None,
            "timeout": 60,
            "max_retries": 10,
            "default_headers": {"test-header": "test-value"},
            "default_query": None,
        },
    }
    error_context = (
        pytest.raises(ValueError, match="None of the following authentication environment variables are set")
        if not mock_os_environ
        else nullcontext()
    )

    with (
        mock.patch.dict(os.environ, mock_os_environ),
        mock.patch("outlines_haystack.generators.openai.from_openai"),
        mock.patch("outlines_haystack.generators.openai.Generator"),
        error_context,
    ):
        component = OpenAIJSONGenerator.from_dict(component_dict)

        if mock_os_environ:
            assert component.model_name == MODEL_NAME
            assert component.api_key.resolve_value() == "test-api-key"
            assert component.timeout == 60
            assert component.max_retries == 10
            assert component.default_headers == {"test-header": "test-value"}
            assert component.schema_object == user_schema_str


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
@mock.patch("outlines_haystack.generators.openai.from_openai")
@mock.patch("outlines_haystack.generators.openai.Generator")
def test_pipeline(_mock_generator: mock.MagicMock, _mock_from_openai: mock.MagicMock) -> None:
    component = OpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    with (
        mock.patch("outlines_haystack.generators.openai.from_openai"),
        mock.patch("outlines_haystack.generators.openai.Generator"),
    ):
        q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
@mock.patch("outlines_haystack.generators.openai.from_openai")
@mock.patch("outlines_haystack.generators.openai.Generator")
def test_run(mock_generator_class: mock.MagicMock, _mock_from_openai: mock.MagicMock) -> None:
    # Generator returns a JSON string in Outlines 1.x
    mock_generator = mock.MagicMock()
    json_string = json.dumps({"name": "John"})
    mock_generator.return_value = json_string
    mock_generator_class.return_value = mock_generator

    component = OpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    response = component.run("How are you?")

    # check that the component returns the correct response
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"], list)
    assert len(response["structured_replies"]) == 1
    assert all(isinstance(reply, str) for reply in response["structured_replies"])
    assert response["structured_replies"][0] == json_string


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
@mock.patch("outlines_haystack.generators.openai.from_openai")
@mock.patch("outlines_haystack.generators.openai.Generator")
def test_run_with_string_return(mock_generator_class: mock.MagicMock, _mock_from_openai: mock.MagicMock) -> None:
    # Test case where Generator returns a JSON string
    mock_generator = mock.MagicMock()
    json_string = json.dumps({"name": "Jane"})
    mock_generator.return_value = json_string
    mock_generator_class.return_value = mock_generator

    component = OpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    response = component.run("How are you?")

    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"][0], str)
    assert response["structured_replies"][0] == json_string


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
@mock.patch("outlines_haystack.generators.openai.from_openai")
@mock.patch("outlines_haystack.generators.openai.Generator")
def test_run_empty_prompt(_mock_generator_class: mock.MagicMock, _mock_from_openai: mock.MagicMock) -> None:
    component = OpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    response = component.run("")

    assert response == {"structured_replies": []}


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
@mock.patch("outlines_haystack.generators.openai.from_openai")
@mock.patch("outlines_haystack.generators.openai.Generator")
def test_run_with_generation_kwargs(mock_generator_class: mock.MagicMock, _mock_from_openai: mock.MagicMock) -> None:
    mock_generator = mock.MagicMock()
    json_string = json.dumps({"name": "John"})
    mock_generator.return_value = json_string
    mock_generator_class.return_value = mock_generator

    component = OpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    generation_kwargs = {"temperature": 0.7, "max_tokens": 100, "seed": 42}
    response = component.run("How are you?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    mock_generator.assert_called_once_with("How are you?", **generation_kwargs)
    assert response["structured_replies"][0] == json_string
