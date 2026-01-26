# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Union

from haystack import component, default_from_dict, default_to_dict
from outlines import Generator, from_transformers
from outlines.types import Choice, JsonSchema
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import Self

from outlines_haystack.generators.utils import validate_choices

if TYPE_CHECKING:
    from collections.abc import Callable

    import outlines.models as outlines_models
    from outlines.generator import SteerableGenerator


class _BaseTransformersGenerator:
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
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}
        self.model: outlines_models.Transformers | None = None
        self._generator: SteerableGenerator | None = None

    @property
    def generator(self) -> SteerableGenerator:
        """Get the generator instance.

        Returns:
            The generator instance (callable that generates text/structured data).

        Raises:
            RuntimeError: If the generator is not initialized (warm_up not called).
        """
        if self._generator is None:
            msg = "Generator not initialized. Call warm_up() before accessing the generator."
            raise RuntimeError(msg)
        return self._generator

    @generator.setter
    def generator(self, value: SteerableGenerator) -> None:
        """Set the generator instance.

        Args:
            value: The generator instance to set.
        """
        self._generator = value

    @property
    def _warmed_up(self) -> bool:
        return self.model is not None and self._generator is not None

    def _warm_up_generate_func(self) -> None:
        """For performance reasons, we should create the generator once."""
        raise NotImplementedError

    def warm_up(self) -> None:
        """Initializes the component."""
        if self._warmed_up:
            return

        hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.tokenizer_kwargs)

        if self.device is not None:
            hf_model = hf_model.to(self.device)

        self.model = from_transformers(hf_model, tokenizer)
        self._warm_up_generate_func()

    def _check_component_warmed_up(self) -> None:
        if not self._warmed_up:
            msg = f"The component {self.__class__.__name__} was not warmed up. Please call warm_up() before running."
            raise RuntimeError(msg)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            model_name=self.model_name,
            device=self.device,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return default_from_dict(cls, data)


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
        super(TransformersJSONGenerator, self).__init__(  # noqa: UP008
            model_name=model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        # Determine the output type for the Generator
        if isinstance(schema_object, str):
            # Raw JSON schema string - wrap with JsonSchema and use whitespace_pattern
            self.schema_object = schema_object
            self.output_type = JsonSchema(schema_object, whitespace_pattern=whitespace_pattern)
        elif isinstance(schema_object, type) and issubclass(schema_object, BaseModel):
            # Pydantic model - use directly
            self.schema_object = json.dumps(schema_object.model_json_schema())
            self.output_type = schema_object
        else:
            # Callable - use directly, Outlines handles it internally
            self.schema_object = str(schema_object)
            self.output_type = schema_object

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
        super(TransformersChoiceGenerator, self).__init__(  # noqa: UP008
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
        choice = self.generator(prompt, **generation_kwargs)
        return {"choice": choice}
