# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from typing import Any, Optional, Union

from haystack import component
from outlines import generate, models


class _BaseMLXLMGenerator:
    def __init__(
        self,
        model_name: str,
        tokenizer_config: Union[dict[str, Any], None] = None,
        model_config: Union[dict[str, Any], None] = None,
        adapter_path: Optional[str] = None,
        lazy: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the MLXLM generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/mlxlm/#load-the-model

        Args:
            model_name: The path or the huggingface repository to load the model from.
            tokenizer_config: Configuration parameters specifically for the tokenizer. Defaults to an empty dictionary.
            model_config: Configuration parameters specifically for the model. Defaults to an empty dictionary.
            adapter_path: Path to the LoRA adapters. If provided, applies LoRA layers to the model. Default: None.
            lazy: If False eval the model parameters to make sure they are loaded in memory before returning,
            otherwise they will be loaded when needed. Default: False
        """
        self.model_name = model_name
        self.tokenizer_config = tokenizer_config if tokenizer_config is not None else {}
        self.model_config = model_config if model_config is not None else {}
        self.adapter_path = adapter_path
        self.lazy = lazy
        self.model = None

    @property
    def _warmed_up(self) -> bool:
        return self.model is not None

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

    def _check_component_warmed_up(self) -> None:
        if not self._warmed_up:
            msg = f"The component {self.__class__.__name__} was not warmed up. Please call warm_up() before running."
            raise RuntimeError(msg)


@component
class MLXLMTextGenerator(_BaseMLXLMGenerator):
    """A component for generating text using an MLXLM model."""

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
