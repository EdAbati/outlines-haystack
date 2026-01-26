# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

from haystack import component, default_to_dict
from outlines import Generator, from_transformers
from outlines.types import Choice
from transformers import AutoModelForCausalLM, AutoTokenizer

from outlines_haystack.generators.base import (
    _BaseLocalGenerator,
)
from outlines_haystack.generators.utils import process_schema_object, validate_choices

if TYPE_CHECKING:
    from collections.abc import Callable

    from outlines.generator import SteerableGenerator
    from pydantic import BaseModel


class _BaseTransformersGenerator(_BaseLocalGenerator):
    def __init__(
        self,
        model_name: str,
        device: Union[str, None] = None,
        model_kwargs: Union[dict[str, Any], None] = None,
        tokenizer_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the Transformers generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/transformers/

        Args:
            model_name: The name of the model as listed on Hugging Face's model page.
            device: The device(s) on which the model should be loaded. This overrides the `device_map` entry in
            `model_kwargs` when provided.
            model_kwargs: A dictionary that contains the keyword arguments to pass to the `from_pretrained` method
            when loading the model.
            tokenizer_kwargs: A dictionary that contains the keyword arguments to pass to the `from_pretrained` method
            when loading the tokenizer.
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}

    def _load_model(self) -> None:
        """Load the Transformers model and tokenizer."""
        hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.tokenizer_kwargs)

        if self.device is not None:
            hf_model = hf_model.to(self.device)

        self.model = from_transformers(hf_model, tokenizer)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            model_name=self.model_name,
            device=self.device,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
        )


@component
class TransformersTextGenerator(_BaseTransformersGenerator):
    """A component for generating text using a Transformers model."""

    def _warm_up_generate_func(self) -> None:
        self.generator: SteerableGenerator = Generator(self.model)

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
class TransformersJSONGenerator(_BaseTransformersGenerator):
    """A component for generating structured data using an Transformers model."""

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        schema_object: Union[str, type[BaseModel], Callable],
        device: Union[str, None] = None,
        model_kwargs: Union[dict[str, Any], None] = None,
        tokenizer_kwargs: Union[dict[str, Any], None] = None,
        whitespace_pattern: Union[str, None] = None,
    ) -> None:
        """Initialize the Transformers JSON generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/transformers/

        Args:
            model_name: The path or the huggingface repository to load the model from.
            schema_object: The schema object to use for structured generation. Can be a Pydantic model class,
                a JSON schema string, or a callable (function signature will be used as schema).
            device: The device(s) on which the model should be loaded. This overrides the `device_map` entry in
            `model_kwargs` when provided.
            model_kwargs: A dictionary that contains the keyword arguments to pass to the `from_pretrained` method
            when loading the model.
            tokenizer_kwargs: A dictionary that contains the keyword arguments to pass to the `from_pretrained` method
            when loading the tokenizer.
            whitespace_pattern: Pattern to use for JSON syntactic whitespace (doesn't impact string literals).
            See https://dottxt-ai.github.io/outlines/latest/reference/generation/json/ for more information.
        """
        _BaseTransformersGenerator.__init__(
            self,
            model_name=model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        self.schema_object, self.output_type = process_schema_object(schema_object, whitespace_pattern)
        self.whitespace_pattern = whitespace_pattern

    def _warm_up_generate_func(self) -> None:
        self.generator: SteerableGenerator = Generator(self.model, self.output_type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            schema_object=self.schema_object,
            device=self.device,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            whitespace_pattern=self.whitespace_pattern,
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
class TransformersChoiceGenerator(_BaseTransformersGenerator):
    """A component that generates a choice between different options using a Transformers model."""

    def __init__(
        self,
        model_name: str,
        choices: list[str],
        device: Union[str, None] = None,
        model_kwargs: Union[dict[str, Any], None] = None,
        tokenizer_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the Transformers Choice generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/transformers/

        Args:
            model_name: The name of the model as listed on Hugging Face's model page.
            choices: The list of choices to choose from.
            device: The device(s) on which the model should be loaded. This overrides the `device_map` entry in
            `model_kwargs` when provided.
            model_kwargs: A dictionary that contains the keyword arguments to pass to the `from_pretrained` method
            when loading the model.
            tokenizer_kwargs: A dictionary that contains the keyword arguments to pass to the `from_pretrained` method
            when loading the tokenizer.
        """
        _BaseTransformersGenerator.__init__(
            self,
            model_name=model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        validate_choices(choices)
        self.choices = choices

    def _warm_up_generate_func(self) -> None:
        self.generator: SteerableGenerator = Generator(self.model, Choice(self.choices))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            choices=self.choices,
            device=self.device,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
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
        answer = self.generator(prompt, **generation_kwargs)
        return {"choice": answer}
