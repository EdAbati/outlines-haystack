from outlines_haystack.generators.openai import OpenAITextGenerator


def test_init_default(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    component = OpenAITextGenerator(model_name="gpt-4o-mini")
    assert component.model_name == "gpt-4o-mini"
    assert component.api_key.resolve_value() == "test-api-key"
    assert component.organization is None
    assert component.project is None
    assert component.base_url is None
    assert component.timeout == 30
    assert component.max_retries == 5
    assert component.default_headers is None
    assert component.default_query is None
