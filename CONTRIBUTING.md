# Contributing

Do you have an idea for a new feature? Did you find a bug that needs fixing?

Feel free to [open an issue](https://github.com/EdAbati/outlines-haystack/issues) or submit a PR!

### Setup development environment

Requirements: [`hatch`](https://hatch.pypa.io/latest/install/), [`pre-commit`](https://pre-commit.com/#install)

1. Clone the repository
1. Run `hatch shell` to create and activate a virtual environment
1. Run `pre-commit install` to install the pre-commit hooks. This will force the linting and formatting checks.

### Run tests

We use `hatch` to run:

- [Linting and formatting checks](https://hatch.pypa.io/dev/community/contributing/#lint): `hatch fmt`
- [Unit tests](https://hatch.pypa.io/dev/tutorials/testing/overview/)
    - Run tests (with the default python version): `hatch test`
    - Run tests with coverage: `hatch test --cov`
    - Run tests with all supported python versions: `hatch test --all`
