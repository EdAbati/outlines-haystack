# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from outlines_haystack.generators.utils import validate_choices


def test_validate_choices_valid() -> None:
    # Should not raise
    validate_choices(["yes", "no", "maybe"])


def test_validate_choices_invalid() -> None:
    with pytest.raises(ValueError, match="Choices must be a list of strings"):
        validate_choices(["yes", 1, "maybe"])  # type: ignore[list-item]
