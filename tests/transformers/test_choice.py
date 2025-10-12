# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from unittest import mock

import pytest
from haystack import Pipeline
from outlines import samplers

from outlines_haystack.generators.transformers import TransformersChoiceGenerator
from tests.utils import CHOICES, mock_choice_func

MODEL_NAME = "hf_org/some_model"


def test_init_default() -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES, device="cpu")
    assert component.model_name == MODEL_NAME
    assert component.choices == CHOICES
    assert component.device == "cpu"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {}


def test_init_invalid_choices() -> None:
    with pytest.raises(ValueError, match="Choices must be a list of strings"):
        TransformersChoiceGenerator(model_name=MODEL_NAME, choices=[1, 2, 3])


def test_init_different_sampler() -> None:
    component = TransformersChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
        sampling_algorithm="multinomial",
        sampling_algorithm_kwargs={"temperature": 0.5},
    )
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}


@mock.patch("outlines_haystack.generators.transformers.generate.choice", return_value="mock_generator")
@mock.patch("outlines_haystack.generators.transformers.models.transformers", return_value="mock_model")
def test_warm_up(mock_model: mock.Mock, mock_generator: mock.Mock) -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES, device="cpu")
    assert component.model is None
    assert component.sampler is None
    assert not component._warmed_up
    component.warm_up()
    assert component.model == "mock_model"
    assert component._warmed_up
    mock_model.assert_called_once_with(
        model_name=MODEL_NAME,
        device="cpu",
        model_kwargs={},
        tokenizer_kwargs={},
    )
    mock_generator.assert_called_once_with(
        "mock_model",
        choices=CHOICES,
        sampler=mock.ANY,
    )
    assert isinstance(component.sampler, samplers.MultinomialSampler)


def test_run_not_warm() -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    with pytest.raises(
        RuntimeError,
        match="The component TransformersChoiceGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


def test_to_dict() -> None:
    component = TransformersChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
        device="cuda",
        sampling_algorithm_kwargs={"temperature": 0.5},
    )
    expected_dict = {
        "type": "outlines_haystack.generators.transformers.TransformersChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
            "device": "cuda",
            "model_kwargs": {},
            "tokenizer_kwargs": {},
            "sampling_algorithm": "multinomial",
            "sampling_algorithm_kwargs": {"temperature": 0.5},
        },
    }
    assert component.to_dict() == expected_dict


def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.transformers.TransformersChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
            "device": "cuda",
            "sampling_algorithm": "multinomial",
            "sampling_algorithm_kwargs": {"temperature": 0.5},
        },
    }
    component = TransformersChoiceGenerator.from_dict(component_dict)
    assert component.model_name == MODEL_NAME
    assert component.choices == CHOICES
    assert component.device == "cuda"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}


def test_pipeline() -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


def test_run() -> None:
    component = TransformersChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)

    with (
        mock.patch("outlines_haystack.generators.transformers.generate.choice") as mock_generate,
        mock.patch("outlines_haystack.generators.transformers.models.transformers") as mock_model,
    ):
        mock_model.return_value = "MockModel"
        mock_generate.return_value = mock_choice_func
        component.warm_up()
        response = component.run("What do you think?")

    assert isinstance(response, dict)
    assert "choice" in response
    assert response["choice"] == "yes"
