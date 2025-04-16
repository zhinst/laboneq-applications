# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Configuration for pytest."""

import pathlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

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

    @asynccontextmanager
    async def async_setup_kernel(self, **kwargs) -> AsyncGenerator[None, None]:
        """Wrap the default .async_setup_kernel with a file lock."""
        locked = True
        self._LOCK.acquire()

        try:
            async with super().async_setup_kernel(**kwargs):
                self._LOCK.release()
                locked = False
                yield
        finally:
            if locked:
                self._LOCK.release()


def pytest_configure(config: object) -> None:
    """Replace nbmake's default NotebookClient with FileLockNotebookClient."""
    nbmake.nb_run.NotebookClient = FileLockNotebookClient
