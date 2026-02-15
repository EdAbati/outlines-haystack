# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import outlines
from haystack import component, default_to_dict
from haystack.lazy_imports import LazyImport
from outlines.types import Choice
from pydantic import BaseModel

from outlines_haystack.generators._base import _BaseLocalGenerator
from outlines_haystack.generators.utils import process_schema_object, validate_choices

with LazyImport("Install outlines-haystack[mlxlm]") as mlx_lm_import:
    import mlx_lm

if TYPE_CHECKING:
    from outlines.generator import SteerableGenerator


class _BaseMLXLMGenerator(_BaseLocalGenerator):
    def __init__(
        self,
        model_name: str,
        tokenizer_config: Union[dict[str, Any], None] = None,
        model_config: Union[dict[str, Any], None] = None,
        adapter_path: Union[str, None] = None,
        lazy: bool = False,  # noqa: FBT001, FBT002
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
        """
        super().__init__()
        self.model_name = model_name
        self.tokenizer_config = tokenizer_config if tokenizer_config is not None else {}
        self.model_config = model_config if model_config is not None else {}
        self.adapter_path = adapter_path
        self.lazy = lazy

    def _load_model(self) -> None:
        """Load the MLXLM model."""
        mlx_lm_import.check()
        mlx_model, tokenizer = mlx_lm.load(
            self.model_name,
            tokenizer_config=self.tokenizer_config,
            model_config=self.model_config,
            adapter_path=self.adapter_path,
            lazy=self.lazy,
        )
        self.model = outlines.from_mlxlm(mlx_model, tokenizer)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            tokenizer_config=self.tokenizer_config,
            model_config=self.model_config,
            adapter_path=self.adapter_path,
            lazy=self.lazy,
        )


@component
class MLXLMTextGenerator(_BaseMLXLMGenerator):
    """A component for generating text using an MLXLM model."""

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
class MLXLMJSONGenerator(_BaseMLXLMGenerator):
    """A component for generating structured data using an MLXLM model."""

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        schema_object: Union[str, type[BaseModel], Callable],
        tokenizer_config: Union[dict[str, Any], None] = None,
        model_config: Union[dict[str, Any], None] = None,
        adapter_path: Union[str, None] = None,
        lazy: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the MLXLM JSON generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/mlxlm/#load-the-model

        Args:
            model_name: The path or the huggingface repository to load the model from.
            schema_object: The schema object to use for structured generation. Can be a Pydantic model class,
                a JSON schema string, or a callable (function signature will be used as schema).
            tokenizer_config: Configuration parameters specifically for the tokenizer.
            If None, defaults to an empty dictionary.
            model_config: Configuration parameters specifically for the model. If None, defaults to an empty dictionary.
            adapter_path: Path to the LoRA adapters. If provided, applies LoRA layers to the model. Default: None.
            lazy: If False eval the model parameters to make sure they are loaded in memory before returning,
            otherwise they will be loaded when needed. Default: False
        """
        _BaseMLXLMGenerator.__init__(
            self,
            model_name=model_name,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            lazy=lazy,
        )

        self.schema_object, self.output_type = process_schema_object(schema_object)

    def _warm_up_generate_func(self) -> None:
        self.generator: SteerableGenerator = outlines.Generator(self.model, self.output_type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            schema_object=self.schema_object,
            tokenizer_config=self.tokenizer_config,
            model_config=self.model_config,
            adapter_path=self.adapter_path,
            lazy=self.lazy,
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
class MLXLMChoiceGenerator(_BaseMLXLMGenerator):
    """A component that generates a choice between different options using an MLXLM model."""

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        choices: list[str],
        tokenizer_config: Union[dict[str, Any], None] = None,
        model_config: Union[dict[str, Any], None] = None,
        adapter_path: Union[str, None] = None,
        lazy: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the MLXLM Choice generator component.

        For more info, see https://dottxt-ai.github.io/outlines/latest/reference/models/mlxlm/#load-the-model

        Args:
            model_name: The path or the huggingface repository to load the model from.
            choices: The list of choices to choose from.
            tokenizer_config: Configuration parameters specifically for the tokenizer.
            If None, defaults to an empty dictionary.
            model_config: Configuration parameters specifically for the model. If None, defaults to an empty dictionary.
            adapter_path: Path to the LoRA adapters. If provided, applies LoRA layers to the model. Default: None.
            lazy: If False eval the model parameters to make sure they are loaded in memory before returning,
            otherwise they will be loaded when needed. Default: False
        """
        _BaseMLXLMGenerator.__init__(
            self,
            model_name=model_name,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            lazy=lazy,
        )
        validate_choices(choices)
        self.choices = choices

    def _warm_up_generate_func(self) -> None:
        self.generator: SteerableGenerator = outlines.Generator(self.model, Choice(self.choices))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            choices=self.choices,
            tokenizer_config=self.tokenizer_config,
            model_config=self.model_config,
            adapter_path=self.adapter_path,
            lazy=self.lazy,
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
