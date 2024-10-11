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

To keep the live preview responsive mkdocstrings is disabled by default.
The environment variable `ENABLE_MKDOCSTRINGS`, which takes a boolean value, can
be used to enable the plugin.

Linux/Mac

```
export ENABLE_MKDOCSTRINGS=true
```

Windows

```
set "ENABLE_MKDOCSTRINGS=true"
```

After this you should be able to build the documentation with:

```
mkdocs-zhinst build
```

MkDocs also comes with a handy live preview that automatically adapts to your
local changes. You can start the preview server using:

```
mkdocs-zhinst serve
```

**IMPORTANT**: Using the `mkdocs` cli interface directly will result in a broken build. Make sure you use the `mkdocs-zhinst` cli instead. This ensures all the right plugins and theme is loaded.