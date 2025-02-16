# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from unittest import mock

import pytest
from haystack import Pipeline
from outlines import samplers

from outlines_haystack.generators.mlxlm import MLXLMChoiceGenerator
from tests.utils import CHOICES, mock_choice_func

MODEL_NAME = "mlx-community/some_model"


def test_init_default() -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    assert component.model_name == MODEL_NAME
    assert component.tokenizer_config == {}
    assert component.model_config == {}
    assert component.adapter_path is None
    assert not component.lazy
    assert component.sampling_algorithm == "multinomial"
    assert component.sampling_algorithm_kwargs == {}
    assert component.choices == CHOICES


def test_init_invalid_choices() -> None:
    with pytest.raises(ValueError, match="Choices must be a list of strings"):
        MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=[1, 2, 3])


def test_init_different_sampler() -> None:
    component = MLXLMChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
        sampling_algorithm="multinomial",
        sampling_algorithm_kwargs={"temperature": 0.5},
    )
    assert component.sampling_algorithm == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}


@mock.patch("outlines_haystack.generators.mlxlm.generate.choice", return_value="mock_generator")
@mock.patch("outlines_haystack.generators.mlxlm.models.mlxlm", return_value="mock_model")
def test_warm_up(mock_mlxlm: mock.Mock, mock_generator: mock.Mock) -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    assert component.model is None
    assert component.sampler is None
    assert not component._warmed_up
    component.warm_up()
    assert component.model == "mock_model"
    assert component._warmed_up
    mock_mlxlm.assert_called_once_with(
        model_name=MODEL_NAME,
        tokenizer_config={},
        model_config={},
        adapter_path=None,
        lazy=False,
    )
    mock_generator.assert_called_once_with(
        "mock_model",
        choices=CHOICES,
        sampler=mock.ANY,
    )
    assert isinstance(component.sampler, samplers.MultinomialSampler)


def test_run_not_warm() -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    with pytest.raises(
        RuntimeError,
        match="The component MLXLMChoiceGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


def test_to_dict() -> None:
    component = MLXLMChoiceGenerator(
        model_name=MODEL_NAME,
        choices=CHOICES,
        tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True},
        sampling_algorithm_kwargs={"temperature": 0.5},
    )
    expected_dict = {
        "type": "outlines_haystack.generators.mlxlm.MLXLMChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
            "tokenizer_config": {"eos_token": "<|endoftext|>", "trust_remote_code": True},
            "model_config": {},
            "adapter_path": None,
            "lazy": False,
            "sampling_algorithm": "multinomial",
            "sampling_algorithm_kwargs": {"temperature": 0.5},
        },
    }
    assert component.to_dict() == expected_dict


def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.mlxlm.MLXLMChoiceGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "choices": CHOICES,
            "tokenizer_config": {"eos_token": "<|endoftext|>", "trust_remote_code": True},
        },
    }
    component = MLXLMChoiceGenerator.from_dict(component_dict)
    assert component.model_name == MODEL_NAME
    assert component.choices == CHOICES
    assert component.tokenizer_config == {"eos_token": "<|endoftext|>", "trust_remote_code": True}
    assert component.model_config == {}
    assert component.adapter_path is None
    assert not component.lazy
    assert component.sampling_algorithm == "multinomial"


def test_pipeline() -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


def test_run() -> None:
    component = MLXLMChoiceGenerator(model_name=MODEL_NAME, choices=CHOICES)

    with (
        mock.patch("outlines_haystack.generators.mlxlm.generate.choice") as mock_generate,
        mock.patch("outlines_haystack.generators.mlxlm.models.mlxlm") as mock_model,
    ):
        mock_model.return_value = "MockModel"
        mock_generate.return_value = mock_choice_func
        component.warm_up()
        response = component.run("What do you think?")

    assert isinstance(response, dict)
    assert "choice" in response
    assert response["choice"] == "yes"
