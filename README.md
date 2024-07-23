# Rules extraction ![Python checks](https://github.com/HES-XPLAIN/rules-extraction/actions/workflows/build.yml/badge.svg)

Rules extraction for eXplainable AI

* [Documentation](https://hes-xplain.github.io/rules-extraction/)

## Installation

```
pip install rules-extraction
```

## Contribution

### Install Python

Install [Python](https://www.python.org/), version 3.9 or newer (3.11 recommanded):

* **Linux, macOS, Windows/WSL**: Use your package manager to install `python3` and `python3-dev`
* **Windows**: `winget install Python.Python.3.11`

> [!WARNING]
> On Windows, avoid installing Python through the Microsoft Store as the package has additional permission restrictions.

### Install dependencies

Using pip

```shell
python -m venv .venv
source .venv/bin/activate
pip install .
```

> [!NOTE]
> On Windows, use `.venv\Scripts\activate` instead.

#### Add dependencies

To add new dependencies to the project, add them to the `pyproject.toml` file.
To add them to the virtualenv, use:

```
pip install .
```

### Work with virtualenv

To activate the virtualenv, use the standard methods:

* Unix: `source .venv/bin/activate`
* Windows: `.venv\Scripts\activate`

To leave the virtualenv, use `deactivate`.

### Install Pre-commit hooks

Git hooks are used to ensure quality checks are run by all developers every time
before a commit.

Install with `pip install pre-commit`.

To enable pre-commit:

```shell
pre-commit install
```

Pre-commit hooks can be run manually with:

```shell
pre-commit run --all-files
```

## Release

To publish the package on [PyPI](https://pypi.org/project/rules-extraction/), refer to [RELEASE](RELEASE.md).
