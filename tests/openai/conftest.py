from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_from_openai() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.openai.from_openai") as from_openai:
        yield from_openai


@pytest.fixture
def mock_openai_generator() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.openai.Generator") as openai_generator:
        yield openai_generator
