# Contributing to LabOne Q Applications Library

We welcome contributions. This document includes the basics to get you started in contributing to `laboneq-applications`.

<!--- TODO: More detailed information -->

## First time setup procedure

<!--- TODO: More detailed information -->

1. Fork the repository
2. Create a new feature branch off of `main` branch
3. Commit your changes
4. Open a pull request
5. Add a reviewer to your feature

## Testing

To ensure proper code quality and style, the project runs certain tests before a pull request can be merged into main

The follow tests are executed:

- Tests
- Linting
<!--- TODO: Coverage -->
<!--- TODO: Typing -->

Installing the development requirements:

```
pip install -e requirements-dev.txt
```

### Tests

Each feature must be tested and the package used for testing is [pytest](https://docs.pytest.org/en/latest/).

Running the tests:

```
pytest
```

### Linting

The project uses [ruff](https://docs.astral.sh/ruff/) for code linting and formatting
The enforced rules are located in `pyproject.toml`.

Running `ruff`:

```
ruff check
```

Fix the errors until there is none. If you encounter lots of error, don't panic. You can either
start fixing them file by file or error code by error code.

## Documentation

Make sure that your code is documented.

The project follows [Google style docstrings](https://google.github.io/styleguide/pyguide.html)

For more information on how to build the documentation and inspect the outcome, see: [Documentation](docs/README.md)
