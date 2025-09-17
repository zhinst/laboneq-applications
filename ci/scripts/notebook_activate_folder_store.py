#!/usr/bin/env python
# /// script
# dependencies = ["nbformat"]
# ///

"""A script to activate the FolderStore in notebooks.

In many example notebooks under ``docs/``, a ``FolderStore`` is created
and then immediately deactivated so as to not clutter the filesystems
of users trying the notebooks out.

This script searches for code cells that contain only ``folder_store.deactivate()``
and replaces them with ``folder_store.activate(); folder_store.save_mode("raise")``
so that saving to the folder store is tested when running notebooks.
"""

import sys
from pathlib import Path

import nbformat


def activate_folder_store(path: Path) -> None:
    """Activate the folder store in the given notebok if one exists."""
    nb = nbformat.read(path, as_version=4)

    updated = False
    for cell in nb.cells:
        if (
            cell.cell_type == "code"
            and cell.source.strip() == "folder_store.deactivate()"
        ):
            cell.source = "\n".join(
                [
                    "folder_store.activate()",
                    'folder_store.save_mode("raise")',
                ]
            )
            updated = True

    if updated:
        nbformat.write(nb, path)


def main(root: str) -> None:
    """Process all the notebooks under the given root."""
    paths = Path(root).glob("**/*.ipynb")
    for p in paths:
        activate_folder_store(p)


if __name__ == "__main__":
    root = sys.argv[1]
    main(root)
