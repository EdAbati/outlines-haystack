from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_from_transformers() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.transformers.outlines.from_transformers") as from_transformers:
        yield from_transformers


@pytest.fixture
def mock_transformers_generator() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.transformers.outlines.Generator") as transformers_generator:
        yield transformers_generator


@pytest.fixture
def mock_auto_model() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.transformers.AutoModelForCausalLM") as auto_model:
        yield auto_model


@pytest.fixture
def mock_auto_tokenizer() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.transformers.AutoTokenizer") as auto_tokenizer:
        yield auto_tokenizer
