from typing import Union


def mock_text_func(
    prompts: Union[str, list[str]],  # noqa: ARG001
    max_tokens: Union[int, None] = None,  # noqa: ARG001
    stop_at: Union[str, list[str], None] = None,  # noqa: ARG001
    seed: Union[int, None] = None,  # noqa: ARG001
    **model_specific_params,  # noqa: ANN003, ARG001
) -> str:
    return "Hello world."
