from outlines_haystack.generators.azure_openai import AzureOpenAITextGenerator


def test_init_default(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "test-endpoint")
    monkeypatch.setenv("OPENAI_API_VERSION", "test-api-version")
    component = AzureOpenAITextGenerator(model_name="gpt-4o-mini")
    assert component.model_name == "gpt-4o-mini"
    assert component.azure_endpoint == "test-endpoint"
    assert component.azure_deployment is None
    assert component.api_version is None
    assert component.api_key.resolve_value() == "test-api-key"
    assert component.azure_ad_token.resolve_value() is None
    assert component.organization is None
    assert component.project is None
    assert component.timeout == 30
    assert component.max_retries == 5
    assert component.default_headers is None
    assert component.default_query is None
