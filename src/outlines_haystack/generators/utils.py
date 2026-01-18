# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Union

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable


def schema_object_to_json_str(schema_object: Union[str, type[BaseModel], Callable]) -> str:
    """Convert a schema object to a JSON string.

    Args:
        schema_object: The schema object to convert to a JSON string.
    """
    if isinstance(schema_object, type) and issubclass(schema_object, BaseModel):
        return json.dumps(schema_object.model_json_schema())
    if callable(schema_object):
        # For callables, we just return a string representation
        # The actual schema conversion is handled by Outlines internally
        return str(schema_object)
    return schema_object


def validate_choices(choices: list[str]) -> None:
    """Validate that choices are a list of str."""
    if not all(isinstance(choice, str) for choice in choices):
        msg = "Choices must be a list of strings. Got: {choices}"
        raise ValueError(msg)
