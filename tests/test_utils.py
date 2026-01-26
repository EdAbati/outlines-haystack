# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest
from outlines.types import JsonSchema
from pydantic import BaseModel

from outlines_haystack.generators.utils import process_schema_object, validate_choices


def test_validate_choices_valid() -> None:
    # Should not raise
    validate_choices(["yes", "no", "maybe"])


def test_validate_choices_invalid() -> None:
    with pytest.raises(ValueError, match="Choices must be a list of strings"):
        validate_choices(["yes", 1, "maybe"])  # type: ignore[list-item]


def test_process_schema_object_with_string() -> None:
    schema_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'
    serializable_str, output_type = process_schema_object(schema_str)

    assert serializable_str == schema_str
    assert isinstance(output_type, JsonSchema)


def test_process_schema_object_with_pydantic_model() -> None:
    class User(BaseModel):
        name: str
        age: int

    serializable_str, output_type = process_schema_object(User)

    # Should serialize to JSON schema string
    assert isinstance(serializable_str, str)
    schema_dict = json.loads(serializable_str)
    assert "properties" in schema_dict
    assert "name" in schema_dict["properties"]
    assert "age" in schema_dict["properties"]

    # Output type should be the Pydantic model class itself
    assert output_type is User


def test_process_schema_object_with_callable() -> None:
    def my_function(name: str, age: int) -> dict:
        return {"name": name, "age": age}

    serializable_str, output_type = process_schema_object(my_function)

    # Should convert to string representation
    assert isinstance(serializable_str, str)
    assert "my_function" in serializable_str

    # Output type should be the callable itself
    assert output_type is my_function
