# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Union

import openai
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.http_client import init_http_client
from outlines import Generator, from_openai
from outlines.types import Choice, JsonSchema
from pydantic import BaseModel
from typing_extensions import Self

from outlines_haystack.generators.utils import validate_choices

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from outlines.models import OpenAI as outlines_OpenAI


class _BaseOpenAIGenerator:
    def __init__(  # noqa: PLR0913
            self,
            model_name: str,
            api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),  # noqa: B008
            organization: Union[str, None] = None,
            project: Union[str, None] = None,
            base_url: Union[str, None] = None,
            timeout: Union[int, None] = None,
            max_retries: Union[int, None] = None,
            default_headers: Union[Mapping[str, str], None] = None,
            default_query: Union[Mapping[str, str], None] = None,
            http_client_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the OpenAI generator.

        Args:
            model_name: The name of the OpenAI model to use.
            api_key: The OpenAI API key. If not provided, uses the `OPENAI_API_KEY` environment variable.
            organization: The organization ID to use for the OpenAI API. If not provided, uses `OPENAI_ORG_ID`
            environment variable.
            project: The project ID to use for the OpenAI API. If not provided, uses `OPENAI_PROJECT_ID`
            environment variable.
            base_url: The base URL to use for the OpenAI API. If not provided, uses `OPENAI_BASE_URL` environment
            variable.
            timeout: The timeout to use for the OpenAI API. If not provided, uses `OPENAI_TIMEOUT` environment variable.
            Defaults to 30.0.
            max_retries: The maximum number of retries to use for the OpenAI API. If not provided, uses
            `OPENAI_MAX_RETRIES` environment variable. Defaults to 5.
            default_headers: The default headers to use in the OpenAI API client.
            default_query: The default query parameters to use in the OpenAI API client.
            http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client`or
            `httpx.AsyncClient`. For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

        """
        self.model_name = model_name
        self.api_key = api_key
        self.organization = organization
        self.project = project
        self.base_url = base_url

        # Same defaults as in Haystack
        # https://github.com/deepset-ai/haystack/blob/6e5c8cdbc04fc06640af433b6bef5201cd0b93bb/haystack/components/generators/openai.py#L130-L133
        self.timeout = timeout or float(os.environ.get("OPENAI_TIMEOUT", "30"))
        self.max_retries = max_retries or int(os.environ.get("OPENAI_MAX_RETRIES", "5"))

        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client_kwargs = http_client_kwargs

        self.client = openai.OpenAI(
            api_key=api_key.resolve_value(),
            organization=self.organization,
            project=self.project,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
            http_client=init_http_client(self.http_client_kwargs, async_client=False),
        )

        self.model: outlines_OpenAI = from_openai(self.client, model_name=self.model_name)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            model_name=self.model_name,
            api_key=self.api_key.to_dict(),
            organization=self.organization,
            project=self.project,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)


@component
class OpenAITextGenerator(_BaseOpenAIGenerator):
    """A component that generates text using the OpenAI API."""

    def __init__(  # noqa: PLR0913
            self,
            model_name: str,
            api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),  # noqa: B008
            organization: Union[str, None] = None,
            project: Union[str, None] = None,
            base_url: Union[str, None] = None,
            timeout: Union[int, None] = None,
            max_retries: Union[int, None] = None,
            default_headers: Union[Mapping[str, str], None] = None,
            default_query: Union[Mapping[str, str], None] = None,
            http_client_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the OpenAI text generator.

        Args:
            model_name: The name of the OpenAI model to use.
            api_key: The OpenAI API key. If not provided, uses the `OPENAI_API_KEY` environment variable.
            organization: The organization ID to use for the OpenAI API. If not provided, uses `OPENAI_ORG_ID`
            environment variable.
            project: The project ID to use for the OpenAI API. If not provided, uses `OPENAI_PROJECT_ID`
            environment variable.
            base_url: The base URL to use for the OpenAI API. If not provided, uses `OPENAI_BASE_URL` environment
            variable.
            timeout: The timeout to use for the OpenAI API. If not provided, uses `OPENAI_TIMEOUT` environment variable.
            Defaults to 30.0.
            max_retries: The maximum number of retries to use for the OpenAI API. If not provided, uses
            `OPENAI_MAX_RETRIES` environment variable. Defaults to 5.
            default_headers: The default headers to use in the OpenAI API client.
            default_query: The default query parameters to use in the OpenAI API client.
            http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`.

        """
        super(OpenAITextGenerator, self).__init__(  # noqa: UP008
            model_name=model_name,
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client_kwargs=http_client_kwargs,
        )
        self.generator = Generator(self.model)

    @component.output_types(replies=list[str])
    def run(self, prompt: str, generation_kwargs: dict[str, Any] | None = None) -> dict[str, list[str]]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            generation_kwargs: The arguments to use for generation. See [outlines docs](https://dottxt-ai.github.io/outlines/latest/features/models/openai/#inference-arguments)
                for more information.
        """
        if not prompt:
            return {"replies": []}

        generation_kwargs = generation_kwargs or {}
        answer = self.generator(prompt, **generation_kwargs)
        return {"replies": [answer]}


@component
class OpenAIJSONGenerator(_BaseOpenAIGenerator):
    """A component that generates structured data using the OpenAI API."""

    def __init__(  # noqa: PLR0913
            self,
            model_name: str,
            schema_object: Union[str, type[BaseModel], Callable],
            api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),  # noqa: B008
            organization: Union[str, None] = None,
            project: Union[str, None] = None,
            base_url: Union[str, None] = None,
            timeout: Union[int, None] = None,
            max_retries: Union[int, None] = None,
            default_headers: Union[Mapping[str, str], None] = None,
            default_query: Union[Mapping[str, str], None] = None,
            http_client_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the OpenAI JSON generator.

        Args:
            model_name: The name of the OpenAI model to use.
            schema_object: The schema object to use for structured generation. Can be a Pydantic model class,
                a JSON schema string, or a callable (function signature will be used as schema).
            api_key: The OpenAI API key. If not provided, uses the `OPENAI_API_KEY` environment variable.
            organization: The organization ID to use for the OpenAI API. If not provided, uses `OPENAI_ORG_ID`
            environment variable.
            project: The project ID to use for the OpenAI API. If not provided, uses `OPENAI_PROJECT_ID`
            environment variable.
            base_url: The base URL to use for the OpenAI API. If not provided, uses `OPENAI_BASE_URL` environment
            variable.
            timeout: The timeout to use for the OpenAI API. If not provided, uses `OPENAI_TIMEOUT` environment variable.
            Defaults to 30.0.
            max_retries: The maximum number of retries to use for the OpenAI API. If not provided, uses
            `OPENAI_MAX_RETRIES` environment variable. Defaults to 5.
            default_headers: The default headers to use in the OpenAI API client.
            default_query: The default query parameters to use in the OpenAI API client.
            http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`.

        """
        super(OpenAIJSONGenerator, self).__init__(  # noqa: UP008
            model_name=model_name,
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client_kwargs=http_client_kwargs,
        )

        # Determine the output type for the Generator
        if isinstance(schema_object, str):
            # Raw JSON schema string - wrap with JsonSchema
            self.schema_object = schema_object
            output_type = JsonSchema(schema_object)
        elif isinstance(schema_object, type) and issubclass(schema_object, BaseModel):
            # Pydantic model - use directly
            self.schema_object = json.dumps(schema_object.model_json_schema())
            output_type = schema_object
        else:
            # Callable - use directly, Outlines handles it internally
            self.schema_object = str(schema_object)
            output_type = schema_object

        self.generator = Generator(self.model, output_type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            schema_object=self.schema_object,
            api_key=self.api_key.to_dict(),
            organization=self.organization,
            project=self.project,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
        )

    @component.output_types(structured_replies=list[str])
    def run(self, prompt: str, generation_kwargs: dict[str, Any] | None = None) -> dict[str, list[str]]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            generation_kwargs: The arguments to use for generation. See [outlines docs](https://dottxt-ai.github.io/outlines/latest/features/models/openai/#inference-arguments)
                for more information.
        """
        if not prompt:
            return {"structured_replies": []}

        generation_kwargs = generation_kwargs or {}
        answer = self.generator(prompt, **generation_kwargs)
        return {"structured_replies": [answer]}


@component
class OpenAIChoiceGenerator(_BaseOpenAIGenerator):
    """A component that generates a choice between different options using the OpenAI API."""

    def __init__(  # noqa: PLR0913
            self,
            model_name: str,
            choices: list[str],
            api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),  # noqa: B008
            organization: Union[str, None] = None,
            project: Union[str, None] = None,
            base_url: Union[str, None] = None,
            timeout: Union[int, None] = None,
            max_retries: Union[int, None] = None,
            default_headers: Union[Mapping[str, str], None] = None,
            default_query: Union[Mapping[str, str], None] = None,
            http_client_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the OpenAI choice generator.

        Args:
            model_name: The name of the OpenAI model to use.
            choices: The list of choices to choose from.
            api_key: The OpenAI API key. If not provided, uses the `OPENAI_API_KEY` environment variable.
            organization: The organization ID to use for the OpenAI API. If not provided, uses `OPENAI_ORG_ID`
            environment variable.
            project: The project ID to use for the OpenAI API. If not provided, uses `OPENAI_PROJECT_ID`
            environment variable.
            base_url: The base URL to use for the OpenAI API. If not provided, uses `OPENAI_BASE_URL` environment
            variable.
            timeout: The timeout to use for the OpenAI API. If not provided, uses `OPENAI_TIMEOUT` environment variable.
            Defaults to 30.0.
            max_retries: The maximum number of retries to use for the OpenAI API. If not provided, uses
            `OPENAI_MAX_RETRIES` environment variable. Defaults to 5.
            default_headers: The default headers to use in the OpenAI API client.
            default_query: The default query parameters to use in the OpenAI API client.
            http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`.

        """
        super(OpenAIChoiceGenerator, self).__init__(  # noqa: UP008
            model_name=model_name,
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client_kwargs=http_client_kwargs,
        )
        validate_choices(choices)
        self.choices = choices
        self.generator = Generator(self.model, Choice(choices))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            choices=self.choices,
            api_key=self.api_key.to_dict(),
            organization=self.organization,
            project=self.project,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
        )

    @component.output_types(choice=str)
    def run(self, prompt: str, generation_kwargs: dict[str, Any] | None = None) -> dict[str, str]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            generation_kwargs: The arguments to use for generation. See [outlines docs](https://dottxt-ai.github.io/outlines/latest/features/models/openai/#inference-arguments)
                for more information.
        """
        if not prompt:
            return {"choice": ""}

        generation_kwargs = generation_kwargs or {}
        answer = self.generator(prompt, **generation_kwargs)
        return {"choice": answer}
