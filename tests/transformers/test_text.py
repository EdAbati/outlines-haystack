# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from unittest import mock

import pytest
from haystack import Pipeline
from outlines import samplers

from outlines_haystack.generators.transformers import TransformersTextGenerator
from tests.utils import mock_text_func

MODEL_NAME = "hf_org/some_model"


def test_init_default() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    assert component.model_name == MODEL_NAME
    assert component.device == "cpu"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}
    assert component.sampling_algorithm == "multinomial"
    assert component.sampling_algorithm_kwargs == {}


def test_init_different_sampler() -> None:
    component = TransformersTextGenerator(
        model_name=MODEL_NAME,
        device="cpu",
        sampling_algorithm="multinomial",
        sampling_algorithm_kwargs={"temperature": 0.5},
    )
    assert component.sampling_algorithm == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}


@mock.patch("outlines_haystack.generators.transformers.models.transformers", return_value="mock_model")
def test_warm_up(mock_mlxlm: mock.Mock) -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    assert component.model is None
    assert component.sampler is None
    assert not component._warmed_up
    component.warm_up()
    assert component.model == "mock_model"
    assert component._warmed_up
    mock_mlxlm.assert_called_once_with(
        model_name=MODEL_NAME,
        device="cpu",
        model_kwargs={},
        tokenizer_kwargs={},
    )
    assert isinstance(component.sampler, samplers.MultinomialSampler)


def test_run_not_warm() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    with pytest.raises(
        RuntimeError,
        match="The component TransformersTextGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


def test_to_dict() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    expected_dict = {
        "type": "outlines_haystack.generators.transformers.TransformersTextGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "device": "cpu",
            "model_kwargs": {},
            "tokenizer_kwargs": {},
            "sampling_algorithm": "multinomial",
            "sampling_algorithm_kwargs": {},
        },
    }
    assert component.to_dict() == expected_dict


def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.transformers.TransformersTextGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "device": "cpu",
        },
    }
    component = TransformersTextGenerator.from_dict(component_dict)
    assert component.model_name == MODEL_NAME
    assert component.device == "cpu"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}
    assert component.sampling_algorithm == "multinomial"
    assert component.sampling_algorithm_kwargs == {}


def test_pipeline() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


def test_run() -> None:
    component = TransformersTextGenerator(model_name=MODEL_NAME, device="cpu")

    with (
        mock.patch("outlines_haystack.generators.transformers.generate.text") as mock_generate_text,
        mock.patch("outlines_haystack.generators.transformers.models.transformers") as mock_model_mlxlm,
    ):
        mock_model_mlxlm.return_value = "MockModel"
        mock_generate_text.return_value = mock_text_func
        component.warm_up()
        response = component.run("How are you?")

    # check that the component returns the correct ChatMessage response
    assert isinstance(response, dict)
    assert "replies" in response
    assert isinstance(response["replies"], list)
    assert len(response["replies"]) == 1
    assert [isinstance(reply, str) for reply in response["replies"]]
