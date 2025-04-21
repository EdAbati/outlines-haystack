# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import Any, Union

from haystack import component, default_from_dict, default_to_dict
from outlines import generate, models
from pydantic import BaseModel
from typing_extensions import Self

from outlines_haystack.generators.utils import (
    SamplingAlgorithm,
    get_sampler,
    get_sampling_algorithm,
    schema_object_to_json_str,
    validate_choices,
)


class _BaseTransformersGenerator:
    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        device: Union[str, None] = None,
        model_kwargs: Union[dict[str, Any], None] = None,
        tokenizer_kwargs: Union[dict[str, Any], None] = None,
        sampling_algorithm: SamplingAlgorithm = SamplingAlgorithm.MULTINOMIAL,
        sampling_algorithm_kwargs: Union[dict[str, Any], None] = None,
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
            sampling_algorithm: The sampling algorithm to use. Default: SamplingAlgorithm.MULTINOMIAL
            sampling_algorithm_kwargs: Additional keyword arguments for the sampling algorithm.
            See https://dottxt-ai.github.io/outlines/latest/reference/samplers/ for the available parameters.
            If None, defaults to an empty dictionary.
        """
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}
        self.sampling_algorithm = get_sampling_algorithm(sampling_algorithm)
        self.sampling_algorithm_kwargs = sampling_algorithm_kwargs if sampling_algorithm_kwargs is not None else {}
        self.model = None
        self.sampler = None
        self.generate_func = None

    @property
    def _warmed_up(self) -> bool:
        return self.model is not None or self.sampler is not None or self.generate_func is not None

    def _warm_up_generate_func(self) -> None:
        """For performance reasons, we should create the generate function once."""
        raise NotImplementedError

    def warm_up(self) -> None:
        """Initializes the component."""
        if self._warmed_up:
            return
        self.model = models.transformers(
            model_name=self.model_name,
            device=self.device,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
        )
        self.sampler = get_sampler(self.sampling_algorithm, **self.sampling_algorithm_kwargs)
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
            sampling_algorithm=self.sampling_algorithm.value,
            sampling_algorithm_kwargs=self.sampling_algorithm_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return default_from_dict(cls, data)


@component
class TransformersTextGenerator(_BaseTransformersGenerator):
    """A component for generating text using a Transformers model."""

    def _warm_up_generate_func(self) -> None:
        self.generate_func = generate.text(self.model, self.sampler)

    @component.output_types(replies=list[str])
    def run(
        self,
        prompt: str,
        max_tokens: Union[int, None] = None,
        stop_at: Union[str, list[str], None] = None,
        seed: Union[int, None] = None,
    ) -> dict[str, list[str]]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            max_tokens: The maximum number of tokens to generate.
            stop_at: A string or list of strings after which to stop generation.
            seed: The seed to use for generation.
        """
        self._check_component_warmed_up()

        if not prompt:
            return {"replies": []}

        answer = self.generate_func(prompts=prompt, max_tokens=max_tokens, stop_at=stop_at, seed=seed)
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
        sampling_algorithm: SamplingAlgorithm = SamplingAlgorithm.MULTINOMIAL,
        sampling_algorithm_kwargs: Union[dict[str, Any], None] = None,
        whitespace_pattern: Union[str, None] = None,
    ) -> None:
        """Initialize the Transformers JSON generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/transformers/

        Args:
            model_name: The path or the huggingface repository to load the model from.
            schema_object: The JSON Schema to generate data for. Can be a JSON string, a Pydantic model, or a callable.
            device: The device(s) on which the model should be loaded. This overrides the `device_map` entry in
            `model_kwargs` when provided.
            model_kwargs: A dictionary that contains the keyword arguments to pass to the `from_pretrained` method
            when loading the model.
            tokenizer_kwargs: A dictionary that contains the keyword arguments to pass to the `from_pretrained` method
            when loading the tokenizer.
            sampling_algorithm: The sampling algorithm to use. Default: SamplingAlgorithm.MULTINOMIAL
            sampling_algorithm_kwargs: Additional keyword arguments for the sampling algorithm.
            See https://dottxt-ai.github.io/outlines/latest/reference/samplers/ for the available parameters.
            If None, defaults to an empty dictionary.
            whitespace_pattern: Pattern to use for JSON syntactic whitespace (doesn't impact string literals).
            See https://dottxt-ai.github.io/outlines/latest/reference/generation/json/ for more information.
        """
        super(TransformersJSONGenerator, self).__init__(  # noqa: UP008
            model_name=model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            sampling_algorithm=sampling_algorithm,
            sampling_algorithm_kwargs=sampling_algorithm_kwargs,
        )
        self.schema_object = schema_object_to_json_str(schema_object)
        self.whitespace_pattern = whitespace_pattern

    def _warm_up_generate_func(self) -> None:
        self.generate_func = generate.json(
            self.model,
            schema_object=self.schema_object,
            sampler=self.sampler,
            whitespace_pattern=self.whitespace_pattern,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            schema_object=self.schema_object,
            device=self.device,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            sampling_algorithm=self.sampling_algorithm.value,
            sampling_algorithm_kwargs=self.sampling_algorithm_kwargs,
            whitespace_pattern=self.whitespace_pattern,
        )

    @component.output_types(structured_replies=list[dict[str, Any]])
    def run(
        self,
        prompt: str,
        max_tokens: Union[int, None] = None,
        stop_at: Union[str, list[str], None] = None,
        seed: Union[int, None] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            max_tokens: The maximum number of tokens to generate.
            stop_at: A string or list of strings after which to stop generation.
            seed: The seed to use for generation.
        """
        self._check_component_warmed_up()

        if not prompt:
            return {"structured_replies": []}

        answer = self.generate_func(prompts=prompt, max_tokens=max_tokens, stop_at=stop_at, seed=seed)
        return {"structured_replies": [answer]}


@component
class TransformersChoiceGenerator(_BaseTransformersGenerator):
    """A component that generates a choice between different options using a Transformers model."""

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        choices: list[str],
        device: Union[str, None] = None,
        model_kwargs: Union[dict[str, Any], None] = None,
        tokenizer_kwargs: Union[dict[str, Any], None] = None,
        sampling_algorithm: SamplingAlgorithm = SamplingAlgorithm.MULTINOMIAL,
        sampling_algorithm_kwargs: Union[dict[str, Any], None] = None,
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
            sampling_algorithm: The sampling algorithm to use. Default: SamplingAlgorithm.MULTINOMIAL
            sampling_algorithm_kwargs: Additional keyword arguments for the sampling algorithm.
            See https://dottxt-ai.github.io/outlines/latest/reference/samplers/ for the available parameters.
            If None, defaults to an empty dictionary.
        """
        super(TransformersChoiceGenerator, self).__init__(  # noqa: UP008
            model_name=model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            sampling_algorithm=sampling_algorithm,
            sampling_algorithm_kwargs=sampling_algorithm_kwargs,
        )
        validate_choices(choices)
        self.choices = choices

    def _warm_up_generate_func(self) -> None:
        self.generate_func = generate.choice(self.model, choices=self.choices, sampler=self.sampler)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            choices=self.choices,
            device=self.device,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            sampling_algorithm=self.sampling_algorithm.value,
            sampling_algorithm_kwargs=self.sampling_algorithm_kwargs,
        )

    @component.output_types(choice=str)
    def run(
        self,
        prompt: str,
        max_tokens: Union[int, None] = None,
        stop_at: Union[str, list[str], None] = None,
        seed: Union[int, None] = None,
    ) -> dict[str, str]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            max_tokens: The maximum number of tokens to generate.
            stop_at: A string or list of strings after which to stop generation.
            seed: The seed to use for generation.
        """
        self._check_component_warmed_up()

        if not prompt:
            return {"choice": ""}

        choice = self.generate_func(prompts=prompt, max_tokens=max_tokens, stop_at=stop_at, seed=seed)
        return {"choice": choice}
