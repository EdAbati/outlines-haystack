# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

import json
from collections.abc import Callable
from enum import Enum
from typing import Any, Union

from outlines import samplers
from outlines.fsm.json_schema import get_schema_from_signature
from pydantic import BaseModel


class SamplingAlgorithm(str, Enum):
    """Sampling algorithms supported by `outline`.

    For more info, see https://dottxt-ai.github.io/outlines/latest/reference/samplers
    """

    MULTINOMIAL = "multinomial"
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"


def get_sampling_algorithm(sampling_algorithm: Union[str, SamplingAlgorithm]) -> SamplingAlgorithm:
    """Get the sampling algorithm."""
    try:
        return SamplingAlgorithm(sampling_algorithm)
    except ValueError as e:
        msg = (
            f"'{sampling_algorithm}' is not a valid SamplingAlgorithm. "
            f"Please use one of {SamplingAlgorithm._member_names_}"
        )
        raise ValueError(msg) from e


def get_sampler(sampling_algorithm: SamplingAlgorithm, **kwargs: dict[str, Any]) -> samplers.Sampler:
    """Get a outline sampler based on the sampling algorithm."""
    if sampling_algorithm == SamplingAlgorithm.MULTINOMIAL:
        return samplers.MultinomialSampler(**kwargs)
    if sampling_algorithm == SamplingAlgorithm.GREEDY:
        return samplers.GreedySampler(**kwargs)
    if sampling_algorithm == SamplingAlgorithm.BEAM_SEARCH:
        return samplers.BeamSearchSampler(**kwargs)
    msg = (
        f"'{sampling_algorithm}' is not a valid SamplingAlgorithm. Please use one of {SamplingAlgorithm._member_names_}"
    )
    raise ValueError(msg)


def schema_object_to_json_str(schema_object: Union[str, type[BaseModel], Callable]) -> str:
    """Convert a schema object to a JSON string.

    Args:
        schema_object: The schema object to convert to a JSON string.
    """
    if isinstance(schema_object, type(BaseModel)):
        return json.dumps(schema_object.model_json_schema())
    if callable(schema_object):
        return json.dumps(get_schema_from_signature(schema_object))
    return schema_object


def validate_choices(choices: list[str]) -> None:
    """Validate that choices are a list of str."""
    if not all(isinstance(choice, str) for choice in choices):
        msg = "Choices must be a list of strings. Got: {choices}"
        raise ValueError(msg)
