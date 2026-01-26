# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_from_llamacpp() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.llama_cpp.from_llamacpp") as from_llamacpp:
        yield from_llamacpp


@pytest.fixture
def mock_llamacpp_generator() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.llama_cpp.Generator") as llamacpp_generator:
        yield llamacpp_generator


@pytest.fixture
def mock_llama() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.llama_cpp.Llama") as llama:
        yield llama
