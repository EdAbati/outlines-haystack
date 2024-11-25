# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

import os
from collections.abc import Mapping
from typing import Any, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from outlines import generate, models
from typing_extensions import Self


class _BaseOpenAIGenerator:
    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),  # noqa: B008
        organization: Optional[str] = None,
        project: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        default_headers: Union[Mapping[str, str], None] = None,
        default_query: Union[Mapping[str, str], None] = None,
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
        """
        self.model_name = model_name
        self.api_key = api_key
        self.organization = organization
        self.project = project
        self.base_url = base_url

        # Same defaults as in Haystack
        # https://github.com/deepset-ai/haystack/blob/3ef8c081be460a91f3c5c29899a6ee6bbc429caa/haystack/components/generators/openai.py#L114-L117
        self.timeout = timeout or float(os.environ.get("OPENAI_TIMEOUT", 30.0))
        self.max_retries = max_retries or int(os.environ.get("OPENAI_MAX_RETRIES", 5))

        self.default_headers = default_headers
        self.default_query = default_query

        self.model = models.openai(
            self.model_name,
            api_key=self.api_key.resolve_value(),
            organization=self.organization,
            project=self.project,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
        )

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

    @component.output_types(replies=list[str])
    def run(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        stop_at: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = None,
    ) -> dict[str, list[str]]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            max_tokens: The maximum number of tokens to generate.
            stop_at: A string or list of strings after which to stop generation.
            seed: The seed to use for generation.
        """
        if not prompt:
            return {"replies": []}

        if seed is not None:
            self.model.config.seed = seed

        generate_text_func = generate.text(self.model)
        answer = generate_text_func(prompts=prompt, max_tokens=max_tokens, stop_at=stop_at)
        return {"replies": [answer]}
