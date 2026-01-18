# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import pytest
from pydantic import BaseModel

from outlines_haystack.generators.utils import (
    schema_object_to_json_str,
    validate_choices,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class MockUser(BaseModel):
    name: str


def mock_func(a: int) -> str:
    return str(a)


@pytest.mark.parametrize(
    ("schema_object", "expected_str"),
    [
        (
            MockUser,
            '{"properties": {"name": {"title": "Name", "type": "string"}}, "required": ["name"], "title": "MockUser", "type": "object"}',  # noqa: E501
        ),
        # For callables, we now just return a string representation
        (
            mock_func,
            str(mock_func),
        ),
        (
            '{"properties": {"a": {"title": "A", "type": "integer"}}, "required": ["a"], "title": "mock_func", "type": "object"}',  # noqa: E501
            '{"properties": {"a": {"title": "A", "type": "integer"}}, "required": ["a"], "title": "mock_func", "type": "object"}',  # noqa: E501
        ),
    ],
)
def test_schema_object_to_json_str(schema_object: Union[str, type[BaseModel], Callable], expected_str: str) -> None:
    assert schema_object_to_json_str(schema_object) == expected_str


def test_validate_choices_valid() -> None:
    # Should not raise
    validate_choices(["yes", "no", "maybe"])


def test_validate_choices_invalid() -> None:
    with pytest.raises(ValueError, match="Choices must be a list of strings"):
        validate_choices(["yes", 1, "maybe"])  # type: ignore[list-item]
