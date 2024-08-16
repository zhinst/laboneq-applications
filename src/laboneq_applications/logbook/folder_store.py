"""Implementation of logbook which stores data in a folder."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING

from laboneq_applications.logbook import Logbook, LogbookStore
from laboneq_applications.logbook.serializer import (
    SerializeOpener,
)
from laboneq_applications.logbook.serializer import (
    serialize as default_serialize,
)

if TYPE_CHECKING:
    from typing import IO, Callable

    from laboneq_applications.logbook import Artifact, SerializerOptions
    from laboneq_applications.workflow.engine.core import Workflow
    from laboneq_applications.workflow.task import Task


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename to make it Windows compatible."""
    # TODO: Make this more like slugify and contract multiple - into one.
    return (
        filename.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("_", "-")
    )


class FolderStore(LogbookStore):
    """A folder-based store that stores workflow results and artifacts in a folder."""

    def __init__(self, folder: Path | str, serialize: Callable | None = None):
        self._folder = Path(folder)
        self._folder.mkdir(parents=True, exist_ok=True)
        if serialize is None:
            serialize = default_serialize
        self._serialize = serialize

    def create_logbook(self, workflow: Workflow) -> Logbook:
        """Create a new logbook for the given workflow."""
        assert workflow.name is not None  # noqa: S101
        folder_name = self._unique_workflow_folder_name(workflow.name)
        return FolderLogbook(self._folder / folder_name, self._serialize)

    def _unique_workflow_folder_name(
        self,
        workflow_name: str,
    ) -> str:
        """Generate a unique workflow folder name within the storage folder.

        Arguments:
            workflow_name: The name of the workflow.

        Returns:
            A unique name for the folder.
        """
        # TODO: Decide whether UTC is the correct timezone.
        #       Likely local time is correct.
        #       What about daylight savings?
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
        workflow_name = _sanitize_filename(workflow_name)
        count = 0
        while True:
            if count > 0:
                potential_name = f"{ts}-{workflow_name}-{count}"
            else:
                potential_name = f"{ts}-{workflow_name}"
            workflow_path = self._folder / potential_name
            if not workflow_path.exists():
                return potential_name
            count += 1


class FolderLogbookOpener(SerializeOpener):
    """A serialization file opener for the FolderStore and FolderLogbook.

    Files are opened in the logbook folder with the base filename being
    `{artifact_name}-{suffix}.{ext}`. The dash is omitted if the suffix
    is empty.

    Arguments:
        logbook:
            The logbook files will be opened for.
        artifact_name:
            The name of the artifact being serialized.
        serializer_options:
            The options for the serializer.
    """

    def __init__(
        self,
        logbook: FolderLogbook,
        artifact_name: str,
        serializer_options: SerializerOptions,
    ):
        self._logbook = logbook
        self._artifact_name = artifact_name
        self._serializer_options = serializer_options
        self._artifact_files = ArtifactFiles()

    def open(
        self,
        ext: str,
        *,
        encoding: str | None = None,
        suffix: str | None = None,
        description: str | None = None,
        binary: bool = False,
    ) -> IO:
        """Open the requested file in the logbook folder."""
        mode = "wb" if binary else "w"
        suffix = f"-{suffix}" if suffix else ""
        filename = self._logbook._unique_filename(
            f"{self._artifact_name}{suffix}.{ext}",
        )
        path = self._logbook._folder / filename
        self._artifact_files.add(filename, description)
        return path.open(mode=mode, encoding=encoding)

    def options(self) -> dict:
        """Return the serializer options."""
        return self._serializer_options

    def artifact_files(self) -> ArtifactFiles:
        """Return the list of files that were opened."""
        return self._artifact_files


class ArtifactFiles:
    """A record of files opened for an artifact."""

    def __init__(self):
        self._files = []

    def add(self, filename: str, description: str | None = None) -> None:
        """Add the specified file to the list of opened files."""
        entry = {
            "filename": filename,
        }
        if description is not None:
            entry["description"] = description
        self._files.append(entry)

    def as_dicts(self) -> list[dict[str, str]]:
        """Return a list of dictionaries describing the opened files.

        The returned dictionaries consist of:
        ```python
        {
            "filename": <filename>,
            "description": <description>,
        }
        ```

        The `description` is omitted if none was supplied by the
        serializer.
        """
        return self._files


class FolderLogbook(Logbook):
    """A logbook that stores a workflow's results and artifacts in a folder."""

    def __init__(self, folder: Path | str, serialize: Callable) -> None:
        self._folder = Path(folder)
        self._folder.mkdir(parents=False, exist_ok=False)
        self._log = Path(folder / "log.jsonl")
        self._log.touch(exist_ok=False)
        self._serialize = serialize

    def _unique_filename(
        self,
        filename: str,
    ) -> str:
        """Generate a unique filename within the workflow folder.

        Arguments:
            filename: The name of the file with its extension.

        Returns:
            A unique name for the file.
        """
        filename = _sanitize_filename(filename)
        filepath = Path(filename)
        stem, suffix = filepath.stem, filepath.suffix

        count = 0
        while True:
            if count > 0:
                potential_name = f"{stem}-{count}{suffix}"
            else:
                potential_name = f"{stem}{suffix}"
            file_path = self._folder / potential_name
            if not file_path.exists():
                return potential_name
            count += 1

    def _append_log(self, data: dict[str, object]) -> None:
        with self._log.open(mode="a") as f:
            json.dump(data, f)
            f.write("\n")

    def on_start(self) -> None:
        """Called when the workflow execution starts."""
        self._append_log({"event": "start"})

    def on_end(self) -> None:
        """Called when the workflow execution ends."""
        self._append_log({"event": "end"})

    def on_error(self, error: Exception) -> None:
        """Called when the workflow raises an exception."""
        self._append_log({"event": "error", "error": repr(error)})

    def on_task_start(
        self,
        task: Task,
    ) -> None:
        """Called when a task begins execution."""
        self._append_log({"event": "task_start", "task": task.name})

    def on_task_end(self, task: Task) -> None:
        """Called when a task ends execution."""
        self._append_log({"event": "task_end", "task": task.name})

    def on_task_error(self, task: Task, error: Exception) -> None:
        """Called when a task raises an exception."""
        self._append_log(
            {"event": "task_error", "task": task.name, "error": repr(error)},
        )

    def comment(self, message: str) -> None:
        """Called to leave a comment."""
        self._append_log({"event": "comment", "message": message})

    def _save(self, artifact: Artifact) -> ArtifactFiles:
        """Store an artifact in one or more files."""
        opener = FolderLogbookOpener(self, artifact.name, artifact.options.copy())
        self._serialize(artifact.obj, opener)
        return opener.artifact_files()

    def save(
        self,
        artifact: Artifact,
    ) -> str:
        """Called to save an artifact."""
        ref = self._save(artifact)
        self._append_log(
            {
                "event": "artifact",
                "artifact_name": artifact.name,
                "artifact_type": type(artifact.obj).__name__,
                "artifact_metadata": artifact.metadata,
                "artifact_options": artifact.options,
                "artifact_files": ref.as_dicts(),
            },
        )
