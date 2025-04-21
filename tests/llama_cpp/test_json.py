# SPDX-FileCopyrightText: 2024-present Edoardo Abati
#
# SPDX-License-Identifier: MIT

from unittest import mock

import pytest
from haystack import Pipeline
from outlines import samplers

from outlines_haystack.generators.llama_cpp import LlamaCppJSONGenerator
from tests.utils import User, mock_json_func, user_schema_str

REPO_ID = "TheBloke/Llama-2-7B-GGUF"
FILE_NAME = "llama-2-7b.Q4_K_M.gguf"


def test_init_default() -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.schema_object == user_schema_str
    assert component.model_kwargs == {}
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {}
    assert component.generation_kwargs == {}
    assert component.whitespace_pattern is None


def test_init_with_string_schema() -> None:
    schema_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=schema_str)
    assert component.schema_object == schema_str


def test_init_different_sampler() -> None:
    component = LlamaCppJSONGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        schema_object=User,
        model_kwargs={"n_gpu_layers": 4},
        sampling_algorithm="multinomial",
        sampling_algorithm_kwargs={"temperature": 0.5},
        generation_kwargs={"max_tokens": 100},
        whitespace_pattern=r"\s+",
    )
    assert component.model_kwargs == {"n_gpu_layers": 4}
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}
    assert component.generation_kwargs == {"max_tokens": 100}
    assert component.whitespace_pattern == r"\s+"


@mock.patch("outlines_haystack.generators.transformers.generate.json", return_value="mock_generator")
@mock.patch("outlines_haystack.generators.llama_cpp.models.llamacpp", return_value="mock_model")
def test_warm_up(mock_model: mock.Mock, mock_generator: mock.Mock) -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    assert component.model is None
    assert component.sampler is None
    assert not component._warmed_up
    component.warm_up()
    assert component.model == "mock_model"
    assert component._warmed_up
    mock_model.assert_called_once_with(
        repo_id=REPO_ID,
        filename=FILE_NAME,
    )
    mock_generator.assert_called_once_with(
        "mock_model",
        schema_object=user_schema_str,
        sampler=mock.ANY,
        whitespace_pattern=None,
    )
    assert isinstance(component.sampler, samplers.MultinomialSampler)


def test_run_not_warm() -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    with pytest.raises(
        RuntimeError,
        match="The component LlamaCppJSONGenerator was not warmed up",
    ):
        component.run(prompt="test-prompt")


def test_to_dict() -> None:
    component = LlamaCppJSONGenerator(
        repo_id=REPO_ID,
        file_name=FILE_NAME,
        schema_object=User,
        model_kwargs={"n_gpu_layers": 4},
        sampling_algorithm_kwargs={"temperature": 0.5},
        whitespace_pattern=r"\s+",
    )
    expected_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppJSONGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "schema_object": user_schema_str,
            "model_kwargs": {"n_gpu_layers": 4},
            "sampling_algorithm": "multinomial",
            "sampling_algorithm_kwargs": {"temperature": 0.5},
            "generation_kwargs": {},
            "whitespace_pattern": r"\s+",
        },
    }
    assert component.to_dict() == expected_dict


def test_from_dict() -> None:
    component_dict = {
        "type": "outlines_haystack.generators.llama_cpp.LlamaCppJSONGenerator",
        "init_parameters": {
            "repo_id": REPO_ID,
            "file_name": FILE_NAME,
            "schema_object": user_schema_str,
            "model_kwargs": {"n_gpu_layers": 4},
            "sampling_algorithm": "multinomial",
            "sampling_algorithm_kwargs": {"temperature": 0.5},
            "whitespace_pattern": r"\s+",
        },
    }
    component = LlamaCppJSONGenerator.from_dict(component_dict)
    assert component.repo_id == REPO_ID
    assert component.file_name == FILE_NAME
    assert component.schema_object == user_schema_str
    assert component.model_kwargs == {"n_gpu_layers": 4}
    assert component.sampling_algorithm.value == "multinomial"
    assert component.sampling_algorithm_kwargs == {"temperature": 0.5}
    assert component.whitespace_pattern == r"\s+"


def test_pipeline() -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)
    p = Pipeline()
    p.add_component(instance=component, name="generator")
    p_str = p.dumps()
    q = Pipeline.loads(p_str)
    assert p.to_dict() == q.to_dict()


def test_run() -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)

    with (
        mock.patch("outlines_haystack.generators.llama_cpp.generate.json") as mock_generate_json,
        mock.patch("outlines_haystack.generators.llama_cpp.models.llamacpp") as mock_model_llamacpp,
    ):
        mock_model_llamacpp.return_value = "MockModel"
        mock_generate_json.return_value = mock_json_func
        component.warm_up()
        response = component.run("Create a user profile for John")

    # check that the component returns the correct response
    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"], list)
    assert len(response["structured_replies"]) == 1
    assert all(isinstance(reply, dict) for reply in response["structured_replies"])


def test_run_empty_prompt() -> None:
    component = LlamaCppJSONGenerator(repo_id=REPO_ID, file_name=FILE_NAME, schema_object=User)

    with (
        mock.patch("outlines_haystack.generators.llama_cpp.generate.json") as mock_generate_json,
        mock.patch("outlines_haystack.generators.llama_cpp.models.llamacpp") as mock_model_llamacpp,
    ):
        mock_model_llamacpp.return_value = "MockModel"
        mock_generate_json.return_value = mock_json_func
        component.warm_up()
        response = component.run("")

    assert isinstance(response, dict)
    assert "structured_replies" in response
    assert isinstance(response["structured_replies"], list)
    assert len(response["structured_replies"]) == 0
