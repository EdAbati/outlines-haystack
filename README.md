# `outlines-haystack`

[![PyPI - Version](https://img.shields.io/pypi/v/outlines-haystack.svg)](https://pypi.org/project/outlines-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/outlines-haystack.svg?logo=python&logoColor=white)](https://pypi.org/project/outlines-haystack)
[![PyPI - License](https://img.shields.io/pypi/l/outlines-haystack)](https://pypi.org/project/outlines-haystack)


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![GH Actions Tests](https://github.com/EdAbati/outlines-haystack/actions/workflows/test.yml/badge.svg)](https://github.com/EdAbati/outlines-haystack/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/EdAbati/outlines-haystack/main.svg)](https://results.pre-commit.ci/latest/github/EdAbati/outlines-haystack/main)

-----

## Table of Contents

- [🛠️ Installation](#installation)
- [📃 Description](#description)
- [💻 Usage](#usage)

## 🛠️ Installation

```console
pip install outlines-haystack
```

## 📃 Description

> Outlines is a Python library that allows you to use Large Language Model in a simple and robust way (with structured generation).  It is built by [.txt](https://dottxt.co).
>
> -- <cite>[Outlines docs](https://dottxt-ai.github.io/outlines/latest/welcome/)</cite>

This library allow you to use [`outlines`](https://dottxt-ai.github.io/outlines/latest/) generators in your [Haystack](https://haystack.deepset.ai) pipelines!

This library currently supports the following generators:
- [x] [JSON](https://dottxt-ai.github.io/outlines/latest/reference/generation/json/): generate a JSON object with a given schema
- [x] [Text](https://dottxt-ai.github.io/outlines/latest/reference/text/): _simply_ generate text
- [ ] [Choices](https://dottxt-ai.github.io/outlines/latest/reference/generation/choices/): ⚠️ coming soon
- [ ] [Regex](https://dottxt-ai.github.io/outlines/latest/reference/generation/regex/): ⚠️ coming soon
- [ ] [Format](https://dottxt-ai.github.io/outlines/latest/reference/generation/format/): ⚠️ coming soon
- [ ] [Grammar](https://dottxt-ai.github.io/outlines/latest/reference/generation/cfg/): ⚠️ coming soon

`outlines` supports a wide range of models and frameworks, we are currently supporting:
- [x] [OpenAI/Azure OpenAI](https://dottxt-ai.github.io/outlines/latest/reference/models/openai/)
- [x] [🤗 Transformers](https://dottxt-ai.github.io/outlines/latest/reference/models/transformers/)
- [x] [`mlx-lm`](https://dottxt-ai.github.io/outlines/latest/reference/models/mlxlm/)

## 💻 Usage

> [!TIP]
> See the [Example Notebooks](./notebooks) for complete examples.
>
> All below examples only use the `transformers` models.

### JSON Generation

```python
>>> from enum import Enum
>>> from pydantic import BaseModel
>>> from outlines_haystack.generators.transformers import TransformersJSONGenerator
>>> class User(BaseModel):
...    name: str
...    last_name: str

>>> generator = TransformersJSONGenerator(
...     model_name="microsoft/Phi-3-mini-4k-instruct",
...     schema_object=User,
...     device="cuda",
...     sampling_algorithm_kwargs={"temperature": 0.5},
... )
>>> generator.warm_up()
>>> generator.run(prompt="Create a user profile with the fields name, last_name")
{'structured_replies': [{'name': 'John', 'last_name': 'Doe'}]}
```

### Text Generation

```python
>>> generator = TransformersTextGenerator(
...     model_name="microsoft/Phi-3-mini-4k-instruct",
...     device="cuda",
...     sampling_algorithm_kwargs={"temperature": 0.5},
... )
>>> generator.warm_up()
>>> generator.run(prompt="What is the capital of Italy?")
{'replies': ['\n\n# Answer\nThe capital of Italy is Rome.']}
```

## License

`outlines-haystack` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
