# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Configuration for pytest."""

import pathlib

import filelock
import nbmake.nb_run
from nbclient import NotebookClient


class FileLockNotebookClient(NotebookClient):
    """Apply a local file lock while starting the notebook kernel.

    This is a workaround for https://github.com/jupyter/jupyter_client/issues/487
    which is triggered by running nbmake with pytest on a machine with many
    cores.
    """

    _LOCK = filelock.FileLock(
        pathlib.Path(__file__).parent /
        ".notebookclient.pytest.lock"
    )

    async def async_start_new_kernel(self, **kwargs) -> None:
        """Wrap the default .async_start_new_kernel with a file lock."""
        with self._LOCK:
            await super().async_start_new_kernel(**kwargs)


def pytest_configure(config: object) -> None:
    """Replace nbmake's default NotebookClient with FileLockNotebookClient."""
    nbmake.nb_run.NotebookClient = FileLockNotebookClient
