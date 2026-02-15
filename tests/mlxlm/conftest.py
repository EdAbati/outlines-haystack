# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Check if mlx_lm is available (only on macOS)
try:
    import mlx_lm  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def pytest_collection_modifyitems(items: list) -> None:
    """Skip mlxlm tests if mlx_lm is not available."""
    if not HAS_MLX:
        skip_mlx = pytest.mark.skip(reason="mlx_lm only available on macOS")
        for item in items:
            item.add_marker(skip_mlx)


@pytest.fixture
def mock_mlx_lm_load() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.mlxlm.mlx_lm.load") as mlx_lm_load:
        # Return a tuple of (model, tokenizer) mocks by default
        mock_mlx_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()
        mlx_lm_load.return_value = (mock_mlx_model, mock_tokenizer)
        yield mlx_lm_load


@pytest.fixture
def mock_from_mlxlm() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.mlxlm.outlines.from_mlxlm") as from_mlxlm:
        yield from_mlxlm


@pytest.fixture
def mock_mlxlm_generator() -> Generator[mock.MagicMock]:
    with mock.patch("outlines_haystack.generators.mlxlm.outlines.Generator") as generator:
        yield generator
