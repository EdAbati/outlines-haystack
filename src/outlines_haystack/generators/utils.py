# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from outlines.types import JsonSchema
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable


def validate_choices(choices: list[str]) -> None:
    """Validate that choices are a list of str."""
    if not all(isinstance(choice, str) for choice in choices):
        msg = "Choices must be a list of strings. Got: {choices}"
        raise ValueError(msg)


def process_schema_object(
    schema_object: str | type[BaseModel] | Callable,
) -> tuple[str, JsonSchema | type[BaseModel] | Callable]:
    """Convert schema_object to (serializable_str, output_type) tuple.

    Args:
        schema_object: The schema object to process. Can be a Pydantic model class,
            a JSON schema string, or a callable (function signature will be used as schema).


    Returns:
        A tuple containing:
        - serializable_str: String representation for serialization/storage
        - output_type: The type to pass to Outlines Generator
    """
    if isinstance(schema_object, str):
        # Raw JSON schema string - wrap with JsonSchema
        serializable_str = schema_object
        output_type = JsonSchema(schema_object)
    elif isinstance(schema_object, type) and issubclass(schema_object, BaseModel):
        # Pydantic model - convert to JSON schema string for serialization
        serializable_str = json.dumps(schema_object.model_json_schema())
        output_type = schema_object
    else:
        # Callable - convert to string representation for serialization
        serializable_str = str(schema_object)
        output_type = schema_object

    return serializable_str, output_type
