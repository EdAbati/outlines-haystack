# `outlines-haystack`

[![PyPI - Version](https://img.shields.io/pypi/v/outlines-haystack.svg)](https://pypi.org/project/outlines-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/outlines-haystack.svg?logo=python&logoColor=white)](https://pypi.org/project/outlines-haystack)
[![PyPI - License](https://img.shields.io/pypi/l/outlines-haystack)](https://pypi.org/project/outlines-haystack)
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
- [x] [Choices](https://dottxt-ai.github.io/outlines/latest/reference/generation/choices/): generate text from a list of options. Useful for classification tasks!
- [x] [Text](https://dottxt-ai.github.io/outlines/latest/reference/text/): _simply_ generate text
- [ ] [Regex](https://dottxt-ai.github.io/outlines/latest/reference/generation/regex/): ⚠️ coming soon
- [ ] [Format](https://dottxt-ai.github.io/outlines/latest/reference/generation/format/): ⚠️ coming soon
- [ ] [Grammar](https://dottxt-ai.github.io/outlines/latest/reference/generation/cfg/): ⚠️ coming soon

`outlines` supports a wide range of models and frameworks, we are currently supporting:
- [x] [OpenAI/Azure OpenAI](https://dottxt-ai.github.io/outlines/latest/reference/models/openai/)
- [x] [🤗 Transformers](https://dottxt-ai.github.io/outlines/latest/reference/models/transformers/)
- [x] [`llama-cpp`](https://dottxt-ai.github.io/outlines/latest/reference/models/llamacpp/)
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

### Choice Generation

```python
>>> from outlines_haystack.generators.transformers import TransformersChoiceGenerator

>>> generator = TransformersChoiceGenerator(
...     model_name="microsoft/Phi-3-mini-4k-instruct",
...     choices=["Positive", "Negative"],
...     device="cuda",
...     sampling_algorithm_kwargs={"temperature": 0.5},
... )
>>> generator.warm_up()
>>> generator.run(prompt="Classify the following statement: 'I love pizza'")
{'choice': 'Positive'}
```

### Text Generation

> [!TIP]
> While `outlines` supports classic text generation, it excels at structured generation.
> For text generation, consider using [Haystack's built-in text generators](https://docs.haystack.deepset.ai/docs/generators) that offer more features.

```python
>>> from outlines_haystack.generators.transformers import TransformersTextGenerator

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
