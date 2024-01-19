# LabOne Q Library documentation

This subdirectory contains the documentation for LabOneQ Library.

## Development

The Documentation is generated with [MkDocs](https://www.mkdocs.org/).

### Build the documentation locally

MkDocs is built with Python, making it easy to install all requirement with `pip`.
The documentation therefore has its own `requirements.txt`

Change to the `docs` directory:

```
cd docs
```

Inside the `docs` directory, execute:

```
pip install -r requirements.txt
```

After this you should be able to build the documentation with:

```
mkdocs build
```

MkDocs also comes with a handy live preview that automatically adapts to your
local changes. You can start the preview server using:

```
mkdocs serve
```
