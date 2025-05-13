# LabOne Q Library documentation

This subdirectory contains the documentation for LabOne Q Library.

## Development

The documentation is generated with an internal version of
[MkDocs](https://www.mkdocs.org/), `mkdocs-zhinst`, that ensures styling.

### Build the documentation locally

* Get `uv` from [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/) if you
don't have it already.

* Change to the `docs` directory[^1]:

``` sh
cd docs
```

* Build documentation with:

``` sh
uv run mkdocs-zhinst build
```

* Build documentation and serve it at `localhost:8000` with:

``` sh
uv run mkdocs-zhinst serve
```

[^1]: If you want to stay in the project root, simply add `--directory docs` after `uv
    run` in all the commands shown.

