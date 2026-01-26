# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations


def validate_choices(choices: list[str]) -> None:
    """Validate that choices are a list of str."""
    if not all(isinstance(choice, str) for choice in choices):
        msg = "Choices must be a list of strings. Got: {choices}"
        raise ValueError(msg)
