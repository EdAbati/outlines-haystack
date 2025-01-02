# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from unittest import mock

import pytest
from haystack import Pipeline
from outlines import samplers

from outlines_haystack.generators.transformers import TransformersJSONGenerator
from tests.utils import User, mock_json_func, user_schema_str

MODEL_NAME = "hf_org/some_model"


def test_init_default() -> None:
    component = TransformersJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        device="cpu",
    )
    assert component.model_name == MODEL_NAME
    assert component.device == "cpu"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}
    assert component.sampling_algorithm == "multinomial"
    assert component.sampling_algorithm_kwargs == {}
    assert component.schema_object == user_schema_str
    assert component.whitespace_pattern is None


def test_init_different_sampler() -> None:
    component = TransformersJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        device="cpu",
        sampling_algorithm="multinomial",
        sampling_algorithm_kwargs={"temperature": 0.5},
        whitespace_pattern=r"[\n\t ]*",
    )
    assert component.sampling_algorithm == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}
    assert component.whitespace_pattern == r"[\n\t ]*"


@mock.patch("outlines_haystack.generators.transformers.generate.json", return_value="mock_generator")
@mock.patch("outlines_haystack.generators.transformers.models.transformers", return_value="mock_model")
def test_warm_up(mock_mlxlm: mock.Mock, mock_generator: mock.Mock) -> None:
    component = TransformersJSONGenerator(model_name=MODEL_NAME, schema_object=User, device="cpu")
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
    mock_generator.assert_called_once_with(
        "mock_model",
        schema_object=user_schema_str,
        sampler=mock.ANY,
        whitespace_pattern=None,
    )
    assert isinstance(component.sampler, samplers.MultinomialSampler)


def test_run_not_warm() -> None:
    component = TransformersJSONGenerator(model_name=MODEL_NAME, schema_object=User, device="cpu")
    with pytest.raises(
        RuntimeError,
        match="The component TransformersJSONGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


def test_to_dict() -> None:
    component = TransformersJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        device="cpu",
        sampling_algorithm_kwargs={"temperature": 0.5},
        whitespace_pattern=r"[\n\t ]*",
    )
    expected_dict = {
        "type": "outlines_haystack.generators.transformers.TransformersJSONGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "device": "cpu",
            "schema_object": user_schema_str,
            "model_kwargs": {},
            "tokenizer_kwargs": {},
            "sampling_algorithm": "multinomial",
            "sampling_algorithm_kwargs": {"temperature": 0.5},
            "whitespace_pattern": r"[\n\t ]*",
        },
    }
    assert component.to_dict() == expected_dict


def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.transformers.TransformersJSONGenerator",
        "init_parameters": {
            "model_name": MODEL_NAME,
            "device": "cpu",
            "schema_object": user_schema_str,
            "whitespace_pattern": r"[\n\t ]*",
        },
    }
    component = TransformersJSONGenerator.from_dict(component_dict)
    assert component.model_name == MODEL_NAME
    assert component.schema_object == user_schema_str
    assert component.device == "cpu"
    assert component.model_kwargs == {}
    assert component.tokenizer_kwargs == {}
    assert component.sampling_algorithm == "multinomial"
    assert component.whitespace_pattern == r"[\n\t ]*"


def test_pipeline() -> None:
    component = TransformersJSONGenerator(model_name=MODEL_NAME, device="cpu", schema_object=User)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


def test_run() -> None:
    component = TransformersJSONGenerator(model_name=MODEL_NAME, device="cpu", schema_object=User)

    with (
        mock.patch("outlines_haystack.generators.transformers.generate.json") as mock_generate_text,
        mock.patch("outlines_haystack.generators.transformers.models.transformers") as mock_model_mlxlm,
    ):
        mock_model_mlxlm.return_value = "MockModel"
        mock_generate_text.return_value = mock_json_func
        component.warm_up()
        response = component.run("How are you?")

    # check that the component returns the correct ChatMessage response
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"], list)
    assert len(response["structured_replies"]) == 1
    assert [isinstance(reply, str) for reply in response["structured_replies"]]
