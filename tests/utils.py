# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from typing import Union

from pydantic import BaseModel


def mock_text_func(
    prompts: Union[str, list[str]],  # noqa: ARG001
    max_tokens: Union[int, None] = None,  # noqa: ARG001
    stop_at: Union[str, list[str], None] = None,  # noqa: ARG001
    seed: Union[int, None] = None,  # noqa: ARG001
    **model_specific_params,  # noqa: ANN003, ARG001
) -> str:
    return "Hello world."


class User(BaseModel):
    name: str


user_schema_str = '{"properties": {"name": {"title": "Name", "type": "string"}}, "required": ["name"], "title": "User", "type": "object"}'  # noqa: E501


def func(a: int) -> str:
    return str(a)


def mock_json_func(
    prompts: Union[str, list[str]],  # noqa: ARG001
    max_tokens: Union[int, None] = None,  # noqa: ARG001
    stop_at: Union[str, list[str], None] = None,  # noqa: ARG001
    seed: Union[int, None] = None,  # noqa: ARG001
    **model_specific_params,  # noqa: ANN003, ARG001
) -> str:
    return '{"name": "John"}'
