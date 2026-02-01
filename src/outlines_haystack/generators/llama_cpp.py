# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import outlines
from haystack import component, default_to_dict
from llama_cpp import Llama
from outlines.types import Choice
from pydantic import BaseModel

from outlines_haystack.generators._base import _BaseLocalGenerator
from outlines_haystack.generators.utils import process_schema_object, validate_choices

if TYPE_CHECKING:
    from outlines.generator import SteerableGenerator


class _BaseLlamaCppGenerator(_BaseLocalGenerator):
    def __init__(
        self,
        repo_id: str,
        file_name: str,
        model_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the LlamaCpp generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/llamacpp/#load-the-model

        Args:
            repo_id: The repository name in the Hugging Face Hub.
            file_name: The name of the GGUF model file.
            model_kwargs: A dictionary that contains the keyword arguments to pass when loading the model. For more info
            see the [Llama-cpp docs](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__)
        """
        super().__init__()
        self.repo_id = repo_id
        self.file_name = file_name
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

    def _load_model(self) -> None:
        """Load the LlamaCpp model."""
        llamacpp_model = Llama.from_pretrained(
            repo_id=self.repo_id,
            filename=self.file_name,
            **self.model_kwargs,
        )
        self.model = outlines.from_llamacpp(llamacpp_model)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            repo_id=self.repo_id,
            file_name=self.file_name,
            model_kwargs=self.model_kwargs,
        )


@component
class LlamaCppTextGenerator(_BaseLlamaCppGenerator):
    """A component for generating text using a LlamaCpp model."""

    def _warm_up_generate_func(self) -> None:
        self.generator: SteerableGenerator = outlines.Generator(self.model)

    @component.output_types(replies=list[str])
    def run(self, prompt: str, generation_kwargs: dict[str, Any] | None = None) -> dict[str, list[str]]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            generation_kwargs: The arguments to use for generation. See [outlines docs](https://dottxt-ai.github.io/outlines/latest/reference/generation/generator/)
                for more information.
        """
        self._check_component_warmed_up()

        if not prompt:
            return {"replies": []}

        generation_kwargs = generation_kwargs or {}
        answer = self.generator(prompt, **generation_kwargs)
        return {"replies": [answer]}


@component
class LlamaCppJSONGenerator(_BaseLlamaCppGenerator):
    """A component for generating structured data using a LlamaCpp model."""

    def __init__(
        self,
        repo_id: str,
        file_name: str,
        schema_object: Union[str, type[BaseModel], Callable],
        model_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the LlamaCpp JSON generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/llamacpp/#load-the-model

        Args:
            repo_id: The repository name in the Hugging Face Hub.
            file_name: The name of the GGUF model file.
            schema_object: The schema object to use for structured generation. Can be a Pydantic model class,
                a JSON schema string, or a callable (function signature will be used as schema).
            model_kwargs: A dictionary that contains the keyword arguments to pass when loading the model. For more info
            see the [Llama-cpp docs](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__)
        """
        _BaseLlamaCppGenerator.__init__(
            self,
            repo_id=repo_id,
            file_name=file_name,
            model_kwargs=model_kwargs,
        )

        self.schema_object, self.output_type = process_schema_object(schema_object)

    def _warm_up_generate_func(self) -> None:
        self.generator: SteerableGenerator = outlines.Generator(self.model, self.output_type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            repo_id=self.repo_id,
            file_name=self.file_name,
            schema_object=self.schema_object,
            model_kwargs=self.model_kwargs,
        )

    @component.output_types(structured_replies=list[str])
    def run(self, prompt: str, generation_kwargs: dict[str, Any] | None = None) -> dict[str, list[str]]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            generation_kwargs: The arguments to use for generation. See [outlines docs](https://dottxt-ai.github.io/outlines/latest/reference/generation/generator/)
                for more information.
        """
        self._check_component_warmed_up()

        if not prompt:
            return {"structured_replies": []}

        generation_kwargs = generation_kwargs or {}
        answer = self.generator(prompt, **generation_kwargs)
        return {"structured_replies": [answer]}


@component
class LlamaCppChoiceGenerator(_BaseLlamaCppGenerator):
    """A component that generates a choice between different options using a LlamaCpp model."""

    def __init__(
        self,
        repo_id: str,
        file_name: str,
        choices: list[str],
        model_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the LlamaCpp Choice generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/llamacpp/#load-the-model

        Args:
            repo_id: The repository name in the Hugging Face Hub.
            file_name: The name of the GGUF model file.
            choices: The list of choices to choose from.
            model_kwargs: A dictionary that contains the keyword arguments to pass when loading the model. For more info
            see the [Llama-cpp docs](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__).
        """
        _BaseLlamaCppGenerator.__init__(
            self,
            repo_id=repo_id,
            file_name=file_name,
            model_kwargs=model_kwargs,
        )
        validate_choices(choices)
        self.choices = choices

    def _warm_up_generate_func(self) -> None:
        self.generator: SteerableGenerator = outlines.Generator(self.model, Choice(self.choices))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            repo_id=self.repo_id,
            file_name=self.file_name,
            choices=self.choices,
            model_kwargs=self.model_kwargs,
        )

    @component.output_types(choice=str)
    def run(self, prompt: str, generation_kwargs: dict[str, Any] | None = None) -> dict[str, str]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            generation_kwargs: The arguments to use for generation. See [outlines docs](https://dottxt-ai.github.io/outlines/latest/reference/generation/generator/)
                for more information.
        """
        self._check_component_warmed_up()

        if not prompt:
            return {"choice": ""}

        generation_kwargs = generation_kwargs or {}
        choice = self.generator(prompt, **generation_kwargs)
        return {"choice": choice}
