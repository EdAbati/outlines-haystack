import pytest

from outlines_haystack.generators.openai_utils import set_openai_config


def test_set_openai_config() -> None:
    assert set_openai_config(None) is None
    assert set_openai_config({"temperature": 0.5}).temperature == 0.5
    with pytest.raises(
        ValueError,
        match="Invalid generation_kwargs: ['temps']. The available parameters are",
    ):
        set_openai_config({"temps": 0.5, "temperature": 0.5})
