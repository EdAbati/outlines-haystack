from enum import Enum
from typing import Any, Union

from outlines import samplers


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
