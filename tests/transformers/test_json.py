# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from haystack import Pipeline

from outlines_haystack.generators.transformers import TransformersJSONGenerator
from tests.utils import User, user_schema_str

if TYPE_CHECKING:
    from collections.abc import Generator

MODEL_NAME = "hf_org/some_model"


@pytest.fixture
def mock_transformers_generator_json(mock_transformers_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    json_string = json.dumps({"name": "John"})
    mock_generator_obj.return_value = json_string
    mock_transformers_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.fixture
def mock_transformers_generator_json_jane(mock_transformers_generator: mock.MagicMock) -> Generator[mock.MagicMock]:
    mock_generator_obj = mock.MagicMock()
    json_string = json.dumps({"name": "Jane"})
    mock_generator_obj.return_value = json_string
    mock_transformers_generator.return_value = mock_generator_obj
    return mock_generator_obj


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
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
    assert component.schema_object == user_schema_str
    assert component.whitespace_pattern is None


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_init_with_whitespace_pattern() -> None:
    component = TransformersJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        device="cpu",
        whitespace_pattern=r"[\n\t ]*",
    )
    assert component.whitespace_pattern == r"[\n\t ]*"


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_init_with_kwargs() -> None:
    component = TransformersJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        device="cpu",
        model_kwargs={"torch_dtype": "float16"},
        tokenizer_kwargs={"padding_side": "left"},
        whitespace_pattern=r"[\n\t ]*",
    )
    assert component.model_kwargs == {"torch_dtype": "float16"}
    assert component.tokenizer_kwargs == {"padding_side": "left"}
    assert component.whitespace_pattern == r"[\n\t ]*"


def test_warm_up(
    mock_from_transformers: mock.MagicMock,
    mock_transformers_generator: mock.MagicMock,
    mock_auto_model: mock.MagicMock,
    mock_auto_tokenizer: mock.MagicMock,
) -> None:
    component = TransformersJSONGenerator(model_name=MODEL_NAME, schema_object=User, device="cpu")
    assert component.model is None
    assert component._generator is None
    assert not component._warmed_up

    # Setup mocks
    mock_hf_model = mock.MagicMock()
    mock_hf_model.to.return_value = mock_hf_model  # Make .to() return itself
    mock_hf_tokenizer = mock.MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_hf_model
    mock_auto_tokenizer.from_pretrained.return_value = mock_hf_tokenizer
    mock_outlines_model = mock.MagicMock()
    mock_from_transformers.return_value = mock_outlines_model

    component.warm_up()

    assert component._warmed_up
    mock_auto_model.from_pretrained.assert_called_once_with(MODEL_NAME, device_map="cpu")
    mock_auto_tokenizer.from_pretrained.assert_called_once_with(MODEL_NAME)
    mock_hf_model.to.assert_called_once_with("cpu")
    mock_from_transformers.assert_called_once_with(mock_hf_model, mock_hf_tokenizer)
    assert component.model == mock_outlines_model
    # Verify Generator was called with the model and the output type (Pydantic model)
    mock_transformers_generator.assert_called_once_with(mock_outlines_model, User)


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_run_not_warm() -> None:
    component = TransformersJSONGenerator(model_name=MODEL_NAME, schema_object=User, device="cpu")
    with pytest.raises(
        RuntimeError,
        match="The component TransformersJSONGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_to_dict() -> None:
    component = TransformersJSONGenerator(
        model_name=MODEL_NAME,
        schema_object=User,
        device="cpu",
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
            "whitespace_pattern": r"[\n\t ]*",
        },
    }
    assert component.to_dict() == expected_dict


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
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
    assert component.whitespace_pattern == r"[\n\t ]*"


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_pipeline() -> None:
    component = TransformersJSONGenerator(model_name=MODEL_NAME, device="cpu", schema_object=User)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator_json", "mock_auto_model", "mock_auto_tokenizer"
)
def test_run() -> None:
    component = TransformersJSONGenerator(model_name=MODEL_NAME, device="cpu", schema_object=User)
    component.warm_up()
    response = component.run("How are you?")

    # check that the component returns the correct response
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"], list)
    assert len(response["structured_replies"]) == 1
    assert all(isinstance(reply, str) for reply in response["structured_replies"])
    json_string = json.dumps({"name": "John"})
    assert response["structured_replies"][0] == json_string


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator_json_jane", "mock_auto_model", "mock_auto_tokenizer"
)
def test_run_with_string_return() -> None:
    # Test case where Generator returns a JSON string
    component = TransformersJSONGenerator(model_name=MODEL_NAME, device="cpu", schema_object=User)
    component.warm_up()
    response = component.run("How are you?")

    json_string = json.dumps({"name": "Jane"})
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"][0], str)
    assert response["structured_replies"][0] == json_string


@pytest.mark.usefixtures(
    "mock_from_transformers", "mock_transformers_generator", "mock_auto_model", "mock_auto_tokenizer"
)
def test_run_empty_prompt() -> None:
    component = TransformersJSONGenerator(model_name=MODEL_NAME, device="cpu", schema_object=User)
    component.warm_up()
    response = component.run("")

    assert response == {"structured_replies": []}


@pytest.mark.usefixtures("mock_from_transformers", "mock_auto_model", "mock_auto_tokenizer")
def test_run_with_generation_kwargs(mock_transformers_generator_json: mock.MagicMock) -> None:
    component = TransformersJSONGenerator(model_name=MODEL_NAME, device="cpu", schema_object=User)
    component.warm_up()
    generation_kwargs = {"max_tokens": 100, "seed": 42}
    response = component.run("How are you?", generation_kwargs=generation_kwargs)

    # Verify the generator was called with the correct kwargs
    json_string = json.dumps({"name": "John"})
    mock_transformers_generator_json.assert_called_once_with("How are you?", **generation_kwargs)
    assert response["structured_replies"][0] == json_string
