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

from outlines_haystack.generators.openai import OpenAIChoiceGenerator
from tests.utils import CHOICES

if TYPE_CHECKING:
    from collections.abc import Generator

MODEL_NAME = "gpt-4o-mini"


@pytest.fixture
def mock_openai_generator_choices(mock_openai_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    mock_generator_obj.return_value = "yes"
    mock_openai_generator.return_value = mock_generator_obj
    return mock_generator_obj


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
def test_init_default(mock_from_openai: mock.MagicMock, mock_openai_generator: mock.MagicMock) -> None:
    component = OpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    assert component.model_name == MODEL_NAME
    assert component.api_key.resolve_value() == "test-api-key"
    assert component.organization is None
    assert component.project is None
    assert component.base_url is None
    assert component.timeout == 30
    assert component.max_retries == 5
    assert component.default_headers is None
    assert component.default_query is None
    assert component.choices == CHOICES
    mock_from_openai.assert_called_once()
    mock_openai_generator.assert_called_once()


def test_init_params(mock_from_openai: mock.MagicMock, mock_openai_generator: mock.MagicMock) -> None:
    component = OpenAIChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
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
    assert component.choices == CHOICES
    mock_from_openai.assert_called_once()
    mock_openai_generator.assert_called_once()


def test_init_value_error() -> None:
    with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
        OpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)


@pytest.mark.usefixtures("mock_from_openai", "mock_openai_generator")
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
def test_to_dict() -> None:
    component = OpenAIChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
        timeout=60,
        max_retries=10,
        default_headers={"test-header": "test-value"},
    )
    assert component.to_dict() == {
        "type": "outlines_haystack.generators.openai.OpenAIChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
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


@pytest.mark.usefixtures("mock_from_openai", "mock_openai_generator")
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
        "type": "outlines_haystack.generators.openai.OpenAIChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
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
        error_context,
    ):
        component = OpenAIChoiceGenerator.from_dict(component_dict)

        if mock_os_environ:
            assert component.model_name == MODEL_NAME
            assert component.api_key.resolve_value() == "test-api-key"
            assert component.timeout == 60
            assert component.max_retries == 10
            assert component.default_headers == {"test-header": "test-value"}
            assert component.choices == CHOICES


@pytest.mark.usefixtures("mock_from_openai", "mock_openai_generator")
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
def test_pipeline() -> None:
    component = OpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures("mock_from_openai", "mock_openai_generator_choices")
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
def test_run() -> None:
    component = OpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    response = component.run("Which option should I choose?")

    assert isinstance(response, dict)
    assert "choice" in response
    assert response["choice"] == "yes"


@pytest.mark.usefixtures("mock_from_openai", "mock_openai_generator")
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
def test_run_empty_prompt() -> None:
    component = OpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    response = component.run("")

    assert response == {"choice": ""}


@pytest.mark.usefixtures("mock_from_openai")
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
def test_run_with_generation_kwargs(mock_openai_generator_choices: mock.MagicMock) -> None:
    component = OpenAIChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    generation_kwargs = {"temperature": 0.5, "max_tokens": 50, "seed": 123}
    response = component.run("Which option should I choose?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    mock_openai_generator_choices.assert_called_once_with("Which option should I choose?", **generation_kwargs)
    assert response["choice"] == "yes"
