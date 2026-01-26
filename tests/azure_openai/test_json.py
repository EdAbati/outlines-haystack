# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline
from haystack.utils import Secret

from outlines_haystack.generators.azure_openai import AzureOpenAIJSONGenerator
from tests.utils import User, user_schema_str

if TYPE_CHECKING:
    from collections.abc import Generator

MODEL_NAME = "gpt-4o-mini"


@pytest.fixture
def mock_azure_openai_generator_json(mock_azure_openai_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    json_string = json.dumps({"name": "John"})
    mock_generator_obj.return_value = json_string
    mock_azure_openai_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.fixture
def mock_azure_openai_generator_json_jane(mock_azure_openai_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    json_string = json.dumps({"name": "Jane"})
    mock_generator_obj.return_value = json_string
    mock_azure_openai_generator.return_value = mock_generator_obj
    return mock_generator_obj


@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
        "OPENAI_API_VERSION": "test-api-version",
    },
)
def test_init_default(mock_from_openai: mock.MagicMock, mock_azure_openai_generator: mock.MagicMock) -> None:
    component = AzureOpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    assert component.model_name == MODEL_NAME
    assert component.azure_endpoint == "test-endpoint"
    assert component.azure_deployment is None
    assert component.api_version is None
    assert component.api_key.resolve_value() == "test-api-key"
    assert component.azure_ad_token.resolve_value() is None
    assert component.organization is None
    assert component.timeout == 30
    assert component.max_retries == 5
    assert component.default_headers is None
    assert component.default_query is None
    assert component.schema_object == user_schema_str
    mock_from_openai.assert_called_once()
    mock_azure_openai_generator.assert_called_once()


@pytest.mark.usefixtures("mock_from_openai", "mock_azure_openai_generator")
def test_init_params() -> None:
    component = AzureOpenAIJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        azure_endpoint="test-endpoint",
        azure_deployment="test-deployment",
        api_version="test-api-version",
        api_key=Secret.from_token("test-api-key"),
        timeout=60,
        max_retries=10,
        default_headers={"test-header": "test-value"},
    )
    assert component.model_name == MODEL_NAME
    assert component.azure_endpoint == "test-endpoint"
    assert component.azure_deployment == "test-deployment"
    assert component.api_version == "test-api-version"
    assert component.api_key.resolve_value() == "test-api-key"
    assert component.azure_ad_token.resolve_value() is None
    assert component.organization is None
    assert component.timeout == 60
    assert component.max_retries == 10
    assert component.default_headers == {"test-header": "test-value"}
    assert component.default_query is None
    assert component.schema_object == user_schema_str


@pytest.mark.parametrize(
    ("mock_os_environ", "expected_error"),
    [
        ({}, "Please provide an Azure endpoint"),
        ({"AZURE_OPENAI_ENDPOINT": "test-endpoint"}, "Please provide an API key or an Azure Active Directory token"),
    ],
)
def test_init_value_error(mock_os_environ: dict[str, str], expected_error: str) -> None:
    with mock.patch.dict(os.environ, mock_os_environ), pytest.raises(ValueError, match=expected_error):
        AzureOpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)


@pytest.mark.usefixtures("mock_from_openai", "mock_azure_openai_generator")
@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
    },
)
def test_to_dict() -> None:
    component = AzureOpenAIJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        azure_deployment="test-deployment",
        api_version="test-api-version",
        timeout=60,
        max_retries=10,
        default_headers={"test-header": "test-value"},
    )
    assert component.to_dict() == {
        "type": "outlines_haystack.generators.azure_openai.AzureOpenAIJSONGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "schema_object": user_schema_str,
            "azure_endpoint": "test-endpoint",
            "azure_deployment": "test-deployment",
            "api_version": "test-api-version",
            "api_key": {"type": "env_var", "env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False},
            "azure_ad_token": {"type": "env_var", "env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False},
            "organization": None,
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
            "AZURE_OPENAI_API_KEY": "test-api-key",
            "AZURE_OPENAI_ENDPOINT": "test-endpoint",
            "OPENAI_API_VERSION": "test-api-version",
        },
    ],
)
def test_from_dict(mock_os_environ: dict[str, str]) -> None:
    component_dict = {
        "type": "outlines_haystack.generators.azure_openai.AzureOpenAIJSONGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "schema_object": user_schema_str,
            "azure_endpoint": "test-endpoint",
            "azure_deployment": "test-deployment",
            "api_version": "test-api-version",
            "api_key": {"type": "env_var", "env_vars": ["AZURE_OPENAI_API_KEY"], "strict": False},
            "azure_ad_token": {"type": "env_var", "env_vars": ["AZURE_OPENAI_AD_TOKEN"], "strict": False},
            "timeout": 60,
            "max_retries": 10,
            "default_headers": {"test-header": "test-value"},
        },
    }
    error_context = (
        pytest.raises(ValueError, match="Please provide an API key") if not mock_os_environ else nullcontext()
    )

    with (
        mock.patch.dict(os.environ, mock_os_environ),
        mock.patch("outlines_haystack.generators.azure_openai.from_openai"),
        mock.patch("outlines_haystack.generators.azure_openai.Generator"),
        error_context,
    ):
        component = AzureOpenAIJSONGenerator.from_dict(component_dict)

        if mock_os_environ:
            assert component.model_name == MODEL_NAME
            assert component.azure_endpoint == "test-endpoint"
            assert component.azure_deployment == "test-deployment"
            assert component.api_version == "test-api-version"
            assert component.api_key.resolve_value() == "test-api-key"
            assert component.azure_ad_token.resolve_value() is None
            assert component.timeout == 60
            assert component.max_retries == 10
            assert component.default_headers == {"test-header": "test-value"}
            assert component.schema_object == user_schema_str


@pytest.mark.usefixtures("mock_from_openai", "mock_azure_openai_generator")
@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
        "OPENAI_API_VERSION": "test-api-version",
    },
)
def test_pipeline() -> None:
    component = AzureOpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures("mock_from_openai", "mock_azure_openai_generator_json")
@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
        "OPENAI_API_VERSION": "test-api-version",
    },
)
def test_run() -> None:
    component = AzureOpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    response = component.run("How are you?")

    # check that the component returns the correct response
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"], list)
    assert len(response["structured_replies"]) == 1
    assert all(isinstance(reply, str) for reply in response["structured_replies"])
    json_string = json.dumps({"name": "John"})
    assert response["structured_replies"][0] == json_string


@pytest.mark.usefixtures("mock_from_openai", "mock_azure_openai_generator_json_jane")
@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
        "OPENAI_API_VERSION": "test-api-version",
    },
)
def test_run_with_string_return() -> None:
    # Test case where Generator returns a JSON string
    component = AzureOpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    response = component.run("How are you?")

    json_string = json.dumps({"name": "Jane"})
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"][0], str)
    assert response["structured_replies"][0] == json_string


@pytest.mark.usefixtures("mock_from_openai", "mock_azure_openai_generator")
@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
        "OPENAI_API_VERSION": "test-api-version",
    },
)
def test_run_empty_prompt() -> None:
    component = AzureOpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    response = component.run("")

    assert response == {"structured_replies": []}


@pytest.mark.usefixtures("mock_from_openai")
@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
        "OPENAI_API_VERSION": "test-api-version",
    },
)
def test_run_with_generation_kwargs(mock_azure_openai_generator_json: mock.MagicMock) -> None:
    component = AzureOpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    generation_kwargs = {"temperature": 0.7, "max_tokens": 100, "seed": 42}
    response = component.run("How are you?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    json_string = json.dumps({"name": "John"})
    mock_azure_openai_generator_json.assert_called_once_with("How are you?", **generation_kwargs)
    assert response["structured_replies"][0] == json_string
