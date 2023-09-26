# Rules extraction ![Python checks](https://github.com/HES-XPLAIN/rules-extraction/actions/workflows/tests.yml/badge.svg)

Rules extraction for eXplainable AI

* [Documentation](https://hes-xplain.github.io/rules-extraction/docs/)
* [Coverage](https://hes-xplain.github.io/rules-extraction/cov/)

## Contribution

### Install Python and Poetry

* Install [Python](https://www.python.org/).
* Install [poetry](https://python-poetry.org/docs/#installation) and add it to your PATH.

Ensure `python` and `poetry` are accessible in the `$PATH` environment variable.

To check the installation, check the following commands return an output:

```shell
python --version
poetry --version
```

Install python dependencies and activate the virtualenv:

```shell
poetry install
poetry shell
```

### Install Pre-commit hooks

Git hooks are used to ensure quality checks are run by all developers every time
before a commit.

```shell
pre-commit install
```

Pre-commit hooks can be run manually with:

```shell
pre-commit run --all-files
```

## Release

To publish the package on [PyPI](https://pypi.org/project/rules-extraction/), refer to [RELEASE](RELEASE.md).
