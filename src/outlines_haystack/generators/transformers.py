# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from typing import Any, Optional, Union

from haystack import component
from outlines import generate, models


class _BaseTransformersGenerator:
    def __init__(
        self,
        model_name: str,
        device: Union[str, None] = None,
        model_kwargs: Union[dict[str, Any], None] = None,
        tokenizer_kwargs: Union[dict[str, Any], None] = None,
    ) -> None:
        """Initialize the MLXLM generator component.

                For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/mlxlm/#load-the-model

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
        self.model = None

    @property
    def _warmed_up(self) -> bool:
        return self.model is not None

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

    def _check_component_warmed_up(self) -> None:
        if not self._warmed_up:
            msg = f"The component {self.__class__.__name__} was not warmed up. Please call warm_up() before running."
            raise RuntimeError(msg)


@component
class TransformersTextGenerator(_BaseTransformersGenerator):
    """A component for generating text using a Transformers model."""

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
        self._check_component_warmed_up()

        if not prompt:
            return {"replies": []}

        generate_text_func = generate.text(self.model)
        answer = generate_text_func(prompts=prompt, max_tokens=max_tokens, stop_at=stop_at, seed=seed)
        return {"replies": [answer]}
