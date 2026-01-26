# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline
from haystack.utils import Secret

from outlines_haystack.generators.azure_openai import AzureOpenAIChoiceGenerator
from tests.utils import CHOICES

if TYPE_CHECKING:
    from collections.abc import Generator

MODEL_NAME = "gpt-4"


@pytest.fixture
def mock_azure_openai_generator_choices(mock_azure_openai_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    mock_generator_obj.return_value = "yes"
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
    component = AzureOpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    assert component.model_name == MODEL_NAME
    assert component.choices == CHOICES
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
    assert component.choices == CHOICES
    mock_from_openai.assert_called_once()
    mock_azure_openai_generator.assert_called_once()


def test_init_params(mock_from_openai: mock.MagicMock, mock_azure_openai_generator: mock.MagicMock) -> None:
    component = AzureOpenAIChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
        azure_endpoint="test-endpoint",
        azure_deployment="test-deployment",
        api_version="test-api-version",
        api_key=Secret.from_token("test-api-key"),
        timeout=60,
        max_retries=10,
        default_headers={"test-header": "test-value"},
    )
    assert component.model_name == MODEL_NAME
    assert component.choices == CHOICES
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
    assert component.choices == CHOICES
    mock_from_openai.assert_called_once()
    mock_azure_openai_generator.assert_called_once()


@pytest.mark.parametrize(
    ("mock_os_environ", "expected_error"),
    [
        ({}, "Please provide an Azure endpoint"),
        ({"AZURE_OPENAI_ENDPOINT": "test-endpoint"}, "Please provide an API key or an Azure Active Directory token"),
    ],
)
def test_init_value_error(mock_os_environ: dict[str, str], expected_error: str) -> None:
    with mock.patch.dict(os.environ, mock_os_environ), pytest.raises(ValueError, match=expected_error):
        AzureOpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)


@pytest.mark.usefixtures("mock_from_openai", "mock_azure_openai_generator")
@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
    },
)
def test_to_dict() -> None:
    component = AzureOpenAIChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
        azure_deployment="test-deployment",
        api_version="test-api-version",
        timeout=60,
        max_retries=10,
        default_headers={"test-header": "test-value"},
    )
    assert component.to_dict() == {
        "type": "outlines_haystack.generators.azure_openai.AzureOpenAIChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
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


@pytest.mark.usefixtures("mock_from_openai", "mock_azure_openai_generator")
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
        "type": "outlines_haystack.generators.azure_openai.AzureOpenAIChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
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
        error_context,
    ):
        component = AzureOpenAIChoiceGenerator.from_dict(component_dict)

        if mock_os_environ:
            assert component.model_name == MODEL_NAME
            assert component.choices == CHOICES
            assert component.azure_endpoint == "test-endpoint"
            assert component.azure_deployment == "test-deployment"
            assert component.api_version == "test-api-version"
            assert component.api_key.resolve_value() == "test-api-key"
            assert component.azure_ad_token.resolve_value() is None
            assert component.timeout == 60
            assert component.max_retries == 10
            assert component.default_headers == {"test-header": "test-value"}
            assert component.choices == CHOICES


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
    component = AzureOpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures("mock_from_openai", "mock_azure_openai_generator_choices")
@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
        "OPENAI_API_VERSION": "test-api-version",
    },
)
def test_run() -> None:
    component = AzureOpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    response = component.run("Which option should I choose?")

    assert isinstance(response, dict)
    assert "choice" in response
    assert response["choice"] == "yes"


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
    component = AzureOpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    response = component.run("")

    assert response == {"choice": ""}


@pytest.mark.usefixtures("mock_from_openai")
@mock.patch.dict(
    os.environ,
    {
        "AZURE_OPENAI_API_KEY": "test-api-key",
        "AZURE_OPENAI_ENDPOINT": "test-endpoint",
        "OPENAI_API_VERSION": "test-api-version",
    },
)
def test_run_with_generation_kwargs(mock_azure_openai_generator_choices: mock.MagicMock) -> None:
    component = AzureOpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    generation_kwargs = {"temperature": 0.5, "max_tokens": 50, "seed": 123}
    response = component.run("Which option should I choose?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    mock_azure_openai_generator_choices.assert_called_once_with("Which option should I choose?", **generation_kwargs)
    assert response["choice"] == "yes"
