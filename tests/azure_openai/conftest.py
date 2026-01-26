from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_from_openai() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.azure_openai.from_openai") as from_openai:
        yield from_openai


@pytest.fixture
def mock_azure_openai_generator() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.azure_openai.Generator") as azure_openai_generator:
        yield azure_openai_generator
