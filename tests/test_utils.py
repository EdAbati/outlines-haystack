# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import Union

import pytest
from outlines import samplers
from pydantic import BaseModel

from outlines_haystack.generators.utils import (
    SamplingAlgorithm,
    get_sampler,
    get_sampling_algorithm,
    schema_object_to_json_str,
)


@pytest.mark.parametrize("sampling_algo_name", ["multinomial", "greedy", "beam_search"])
def test_sampling_algorithm(sampling_algo_name: str) -> None:
    sampler = SamplingAlgorithm(sampling_algo_name)
    assert sampler == sampling_algo_name


def test_sampling_algorithm_error() -> None:
    with pytest.raises(ValueError, match="'test' is not a valid SamplingAlgorithm"):
        SamplingAlgorithm("test")


@pytest.mark.parametrize(
    ("sampling_algo", "expected_sampling_algo"),
    [
        (SamplingAlgorithm.MULTINOMIAL, SamplingAlgorithm.MULTINOMIAL),
        (SamplingAlgorithm.GREEDY, SamplingAlgorithm.GREEDY),
        (SamplingAlgorithm.BEAM_SEARCH, SamplingAlgorithm.BEAM_SEARCH),
        ("multinomial", SamplingAlgorithm.MULTINOMIAL),
        ("greedy", SamplingAlgorithm.GREEDY),
        ("beam_search", SamplingAlgorithm.BEAM_SEARCH),
    ],
)
def test_get_sampling_algorithm(sampling_algo: SamplingAlgorithm, expected_sampling_algo: SamplingAlgorithm) -> None:
    assert get_sampling_algorithm(sampling_algo) == expected_sampling_algo


def test_get_sampling_algorithm_error() -> None:
    with pytest.raises(ValueError, match="'test' is not a valid SamplingAlgorithm. Please use one of"):
        get_sampling_algorithm("test")


@pytest.mark.parametrize(
    ("sampling_algo", "expected_sampler", "sampler_kwargs"),
    [
        (SamplingAlgorithm.MULTINOMIAL, samplers.MultinomialSampler, {"temperature": 0.5}),
        (SamplingAlgorithm.GREEDY, samplers.GreedySampler, {}),
        (SamplingAlgorithm.BEAM_SEARCH, samplers.BeamSearchSampler, {"beams": 5}),
        ("multinomial", samplers.MultinomialSampler, {"temperature": 0.5}),
        ("greedy", samplers.GreedySampler, {}),
        ("beam_search", samplers.BeamSearchSampler, {"beams": 5}),
    ],
)
def test_get_sample(sampling_algo: SamplingAlgorithm, expected_sampler: samplers.Sampler, sampler_kwargs: dict) -> None:
    sampler = get_sampler(sampling_algo, **sampler_kwargs)
    assert isinstance(sampler, expected_sampler)
    for key, value in sampler_kwargs.items():
        if sampling_algo == SamplingAlgorithm.BEAM_SEARCH and key == "beams":
            assert sampler.samples == value
        else:
            assert getattr(sampler, key) == value


def test_get_sampler_error() -> None:
    with pytest.raises(ValueError, match="'test' is not a valid SamplingAlgorithm. Please use one of"):
        get_sampler("test")


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
        (
            mock_func,
            '{"properties": {"a": {"title": "A", "type": "integer"}}, "required": ["a"], "title": "mock_func", "type": "object"}',  # noqa: E501
        ),
        (
            '{"properties": {"a": {"title": "A", "type": "integer"}}, "required": ["a"], "title": "mock_func", "type": "object"}',  # noqa: E501
            '{"properties": {"a": {"title": "A", "type": "integer"}}, "required": ["a"], "title": "mock_func", "type": "object"}',  # noqa: E501
        ),
    ],
)
def test_schema_object_to_json_str(schema_object: Union[str, type[BaseModel], Callable], expected_str: str) -> None:
    assert schema_object_to_json_str(schema_object) == expected_str
