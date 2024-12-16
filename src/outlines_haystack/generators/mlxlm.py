# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from typing import Any, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from outlines import generate, models
from typing_extensions import Self

from outlines_haystack.generators.utils import SamplingAlgorithm, get_sampler, get_sampling_algorithm


class _BaseMLXLMGenerator:
    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        tokenizer_config: Union[dict[str, Any], None] = None,
        model_config: Union[dict[str, Any], None] = None,
        adapter_path: Union[str, None] = None,
        lazy: bool = False,  # noqa: FBT001, FBT002
        sampling_algorithm: SamplingAlgorithm = SamplingAlgorithm.MULTINOMIAL,
        sampling_algorithm_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the MLXLM generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/mlxlm/#load-the-model

        Args:
            model_name: The path or the huggingface repository to load the model from.
            tokenizer_config: Configuration parameters specifically for the tokenizer.
            If None, defaults to an empty dictionary.
            model_config: Configuration parameters specifically for the model. If None, defaults to an empty dictionary.
            adapter_path: Path to the LoRA adapters. If provided, applies LoRA layers to the model. Default: None.
            lazy: If False eval the model parameters to make sure they are loaded in memory before returning,
            otherwise they will be loaded when needed. Default: False
            sampling_algorithm: The sampling algorithm to use. Default: SamplingAlgorithm.MULTINOMIAL
            sampling_algorithm_kwargs: Additional keyword arguments for the sampling algorithm.
            See https://dottxt-ai.github.io/outlines/latest/reference/samplers/ for the available parameters.
            If None, defaults to an empty dictionary.
        """
        self.model_name = model_name
        self.tokenizer_config = tokenizer_config if tokenizer_config is not None else {}
        self.model_config = model_config if model_config is not None else {}
        self.adapter_path = adapter_path
        self.lazy = lazy
        self.sampling_algorithm = get_sampling_algorithm(sampling_algorithm)
        self.sampling_algorithm_kwargs = sampling_algorithm_kwargs if sampling_algorithm_kwargs is not None else {}
        self.model = None
        self.sampler = None

    @property
    def _warmed_up(self) -> bool:
        return self.model is not None or self.sampler is not None

    def warm_up(self) -> None:
        """Initializes the component."""
        if self._warmed_up:
            return
        self.model = models.mlxlm(
            model_name=self.model_name,
            tokenizer_config=self.tokenizer_config,
            model_config=self.model_config,
            adapter_path=self.adapter_path,
            lazy=self.lazy,
        )
        self.sampler = get_sampler(self.sampling_algorithm, **self.sampling_algorithm_kwargs)

    def _check_component_warmed_up(self) -> None:
        if not self._warmed_up:
            msg = f"The component {self.__class__.__name__} was not warmed up. Please call warm_up() before running."
            raise RuntimeError(msg)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            model_name=self.model_name,
            tokenizer_config=self.tokenizer_config,
            model_config=self.model_config,
            adapter_path=self.adapter_path,
            lazy=self.lazy,
            sampling_algorithm=self.sampling_algorithm.value,
            sampling_algorithm_kwargs=self.sampling_algorithm_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return default_from_dict(cls, data)


@component
class MLXLMTextGenerator(_BaseMLXLMGenerator):
    """A component for generating text using an MLXLM model."""

    @component.output_types(replies=list[str])
    def run(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> dict[str, list[str]]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            max_tokens: The maximum number of tokens to generate.
        """
        self._check_component_warmed_up()

        if not prompt:
            return {"replies": []}

        generate_text_func = generate.text(self.model, self.sampler)
        answer = generate_text_func(prompts=prompt, max_tokens=max_tokens)
        return {"replies": [answer]}
