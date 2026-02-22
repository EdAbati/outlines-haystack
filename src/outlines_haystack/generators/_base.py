# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
"""Base classes and mixins for Outlines-Haystack generators."""

from abc import abstractmethod
from typing import Any

from haystack import component, default_from_dict
from outlines.generator import SteerableGenerator
from typing_extensions import Self


class _BaseLocalGenerator:
    """Base class for local model generators (transformers, llama_cpp, mlxlm).

    Provides shared functionality for generators that need to load models locally
    and warm up before use.
    """

    def __init__(self) -> None:
        """Initialize the base local generator."""
        self.model: Any = None
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
        """Check if the component has been warmed up.

        Returns:
            True if both model and generator are initialized, False otherwise.
        """
        return self.model is not None and self._generator is not None

    def _check_component_warmed_up(self) -> None:
        """Check if the component has been warmed up.

        Raises:
            RuntimeError: If the component has not been warmed up.
        """
        if not self._warmed_up:
            msg = f"The component {self.__class__.__name__} was not warmed up. Please call warm_up() before running."
            raise RuntimeError(msg)

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def _warm_up_generate_func(self) -> None:
        """Create the generator function. Must be implemented by subclasses."""
        ...

    def warm_up(self) -> None:
        """Initialize the component by loading the model and creating the generator."""
        if self._warmed_up:
            return

        self._load_model()
        self._warm_up_generate_func()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize this component from a dictionary.

        Args:
            data: The dictionary to deserialize from.

        Returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)


class _BaseAPIGenerator:
    """Base class for API-based generators (openai, azure_openai).

    API generators don't need warm_up as they're always ready to use.
    """

    def _pre_run_check(self) -> None:
        """Pre-run check for API generators.

        API generators are always ready, so this is a no-op.
        """


class TextGeneratorMixin:
    """Mixin that provides the run() method for text generation."""

    @component.output_types(replies=list[str])
    def run(self, prompt: str, generation_kwargs: dict[str, Any] | None = None) -> dict[str, list[str]]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            generation_kwargs: The arguments to use for generation.

        Returns:
            A dictionary with the key "replies" containing a list of generated strings.
        """
        if hasattr(self, "_check_component_warmed_up"):
            self._check_component_warmed_up()

        if not prompt:
            return {"replies": []}

        generation_kwargs = generation_kwargs or {}
        answer = self.generator(prompt, **generation_kwargs)
        return {"replies": [answer]}


class JSONGeneratorMixin:
    """Mixin that provides the run() method for JSON/structured generation."""

    @component.output_types(structured_replies=list[str])
    def run(self, prompt: str, generation_kwargs: dict[str, Any] | None = None) -> dict[str, list[str]]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            generation_kwargs: The arguments to use for generation.

        Returns:
            A dictionary with the key "structured_replies" containing a list of structured outputs.
        """
        if hasattr(self, "_check_component_warmed_up"):
            self._check_component_warmed_up()

        if not prompt:
            return {"structured_replies": []}

        generation_kwargs = generation_kwargs or {}
        answer = self.generator(prompt, **generation_kwargs)
        return {"structured_replies": [answer]}


class ChoiceGeneratorMixin:
    """Mixin that provides the run() method for choice generation."""

    @component.output_types(choice=str)
    def run(self, prompt: str, generation_kwargs: dict[str, Any] | None = None) -> dict[str, str]:
        """Run the generation component based on a prompt.

        Args:
            prompt: The prompt to use for generation.
            generation_kwargs: The arguments to use for generation.

        Returns:
            A dictionary with the key "choice" containing the selected choice string.
        """
        if hasattr(self, "_check_component_warmed_up"):
            self._check_component_warmed_up()

        if not prompt:
            return {"choice": ""}

        generation_kwargs = generation_kwargs or {}
        answer = self.generator(prompt, **generation_kwargs)
        return {"choice": answer}
