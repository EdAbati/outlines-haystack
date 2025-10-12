# Contributing

Do you have an idea for a new feature? Did you find a bug that needs fixing?

Feel free to [open an issue](https://github.com/EdAbati/outlines-haystack/issues) or submit a PR!

### Setup development environment

Requirements: [`uv`](https://docs.astral.sh/uv/), [`pre-commit`](https://pre-commit.com/#install)

1. Clone the repository
1. Run `uv venv` to create and `source .venv/bin/activate` to activate a virtual environment
1. Run `pre-commit install` to install the pre-commit hooks. This will force the linting and formatting checks.

### Run tests

We use `uv run` to run:

- Linting and formatting checks: `uv run task fmt`
- Unit tests:
    - Run tests (with the default python version): `uv run task test`
    - Run tests with coverage: `uv run task test-cov`
    - Run tests with a specific python versions: `uv run --python <version> task test`
