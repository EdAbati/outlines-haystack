# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

import os
from contextlib import nullcontext
from unittest import mock

import pytest
from haystack import Pipeline
from haystack.utils import Secret

from outlines_haystack.generators.azure_openai import AzureOpenAIJSONGenerator
from tests.utils import User, mock_json_func, user_schema_str

MODEL_NAME = "gpt-4o-mini"


@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
        "OPENAI_API_VERSION": "test-api-version",
    },
)
def test_init_default() -> None:
    component = AzureOpenAIJSONGenerator(model_name=MODEL_NAME, schema_object=User)
    assert component.model_name == MODEL_NAME
    assert component.azure_endpoint == "test-endpoint"
    assert component.azure_deployment is None
    assert component.api_version is None
    assert component.api_key.resolve_value() == "test-api-key"
    assert component.azure_ad_token.resolve_value() is None
    assert component.organization is None
    assert component.project is None
    assert component.timeout == 30
    assert component.max_retries == 5
    assert component.default_headers is None
    assert component.default_query is None
    assert component.generation_kwargs == {}
    assert component.openai_config is None
    assert component.schema_object == user_schema_str


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
        generation_kwargs={"temperature": 0.5},
    )
    assert component.model_name == MODEL_NAME
    assert component.azure_endpoint == "test-endpoint"
    assert component.azure_deployment == "test-deployment"
    assert component.api_version == "test-api-version"
    assert component.api_key.resolve_value() == "test-api-key"
    assert component.azure_ad_token.resolve_value() is None
    assert component.organization is None
    assert component.project is None
    assert component.timeout == 60
    assert component.max_retries == 10
    assert component.default_headers == {"test-header": "test-value"}
    assert component.default_query is None
    assert component.generation_kwargs == {"temperature": 0.5}
    assert component.openai_config.temperature == 0.5
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
        generation_kwargs={"temperature": 0.5},
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
            "project": None,
            "timeout": 60,
            "max_retries": 10,
            "default_headers": {"test-header": "test-value"},
            "default_query": None,
            "generation_kwargs": {"temperature": 0.5},
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
            "generation_kwargs": {"temperature": 0.5},
        },
    }
    error_context = (
        pytest.raises(ValueError, match="Please provide an API key") if not mock_os_environ else nullcontext()
    )

    with mock.patch.dict(os.environ, mock_os_environ), error_context:
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
            assert component.generation_kwargs == {"temperature": 0.5}
            assert component.openai_config.temperature == 0.5
            assert component.schema_object == user_schema_str


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

    with mock.patch("outlines_haystack.generators.azure_openai.generate.json") as mock_generate_text:
        mock_generate_text.return_value = mock_json_func
        response = component.run("How are you?")

    # check that the component returns the correct ChatMessage response
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"], list)
    assert len(response["structured_replies"]) == 1
    assert [isinstance(reply, str) for reply in response["structured_replies"]]
