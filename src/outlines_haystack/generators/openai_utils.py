# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from typing import Any, Union

from outlines.models.openai import OpenAIConfig


def set_openai_config(generation_kwargs: Union[dict[str, Any], None] = None) -> Union[OpenAIConfig, None]:
    """Set the OpenAIConfig from the generation_kwargs."""
    if generation_kwargs is None:
        return None
    try:
        return OpenAIConfig(**generation_kwargs)
    except TypeError as e:
        available_params = list(OpenAIConfig.__dataclass_fields__.keys())
        invalid_params = list(set(generation_kwargs.keys()) - set(available_params))
        msg = f"Invalid generation_kwargs: {invalid_params}. The available parameters are: {available_params}"
        raise ValueError(msg) from e
