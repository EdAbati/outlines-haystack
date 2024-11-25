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


class _BaseAzureOpenAIGenerator:
    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Secret = Secret.from_env_var("AZURE_OPENAI_API_KEY", strict=False),  # noqa: B008
        azure_ad_token: Secret = Secret.from_env_var("AZURE_OPENAI_AD_TOKEN", strict=False),  # noqa: B008
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        default_headers: Union[Mapping[str, str], None] = None,
        default_query: Union[Mapping[str, str], None] = None,
    ) -> None:
        """Initialize the Azure OpenAI generator.

        Args:
            model_name: The name of the OpenAI model to use. The model name is needed to load the correct tokenizer for
            the model. The tokenizer is necessary for structured generation. See https://dottxt-ai.github.io/outlines/latest/reference/models/openai/#azure-openai-models
            azure_endpoint: The endpoint of the deployed model, for example `https://example-resource.azure.openai.com/`.
            azure_deployment:  A model deployment, if given sets the base client URL to include
            `/deployments/{azure_deployment}`.
            api_version: The API version to use for the Azure OpenAI API.
            api_key: The Azure OpenAI API key. If not provided, uses the `OPENAI_API_KEY` environment variable.
            azure_ad_token: Your Azure Active Directory token, https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id
            organization: The organization ID to use for the Azure OpenAI API. If not provided, uses `OPENAI_ORG_ID`
            environment variable.
            project: The project ID to use for the Azure OpenAI API. If not provided, uses `OPENAI_PROJECT_ID`
            environment variable.
            timeout: The timeout to use for the Azure OpenAI API. If not provided, uses `OPENAI_TIMEOUT` environment
            variable. Defaults to 30.0.
            max_retries: The maximum number of retries to use for the Azure OpenAI API. If not provided, uses
            `OPENAI_MAX_RETRIES` environment variable. Defaults to 5.
            default_headers: The default headers to use in the Azure OpenAI API client.
            default_query: The default query parameters to use in the Azure OpenAI API client.
        """
        # Same defaults as in Haystack
        # https://github.com/deepset-ai/haystack/blob/97126eb544be5bb7d1c5273e85597db6011b017c/haystack/components/generators/azure.py#L116-L125
        azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            msg = "Please provide an Azure endpoint or set the environment variable AZURE_OPENAI_ENDPOINT."
            raise ValueError(msg)

        if api_key is None and azure_ad_token is None:
            msg = "Please provide an API key or an Azure Active Directory token."
            raise ValueError(msg)

        self.model_name = model_name
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.api_version = api_version
        self.api_key = api_key
        self.azure_ad_token = azure_ad_token
        self.organization = organization
        self.project = project

        # https://github.com/deepset-ai/haystack/blob/97126eb544be5bb7d1c5273e85597db6011b017c/haystack/components/generators/azure.py#L139-L140
        self.timeout = timeout or float(os.environ.get("OPENAI_TIMEOUT", 30.0))
        self.max_retries = max_retries or int(os.environ.get("OPENAI_MAX_RETRIES", 5))

        self.default_headers = default_headers
        self.default_query = default_query

        self.model = models.azure_openai(
            deployment_name=self.azure_deployment,
            model_name=self.model_name,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            api_key=self.api_key,
            azure_ad_token=self.azure_ad_token,
            organization=self.organization,
            project=self.project,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
        )

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            model_name=self.model_name,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_version=self.api_version,
            api_key=self.api_key.to_dict(),
            azure_ad_token=self.azure_ad_token.to_dict(),
            organization=self.organization,
            project=self.project,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key", "azure_ad_token"])
        return default_from_dict(cls, data)


@component
class AzureOpenAITextGenerator(_BaseAzureOpenAIGenerator):
    """A component that generates text using the Azure OpenAI API."""

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
