"""Implementation of logbook which stores data in a folder."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING

from laboneq_applications.logbook import Logbook, LogbookStore

if TYPE_CHECKING:
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

    def __init__(self, folder: Path | str):
        self._folder = Path(folder)
        self._folder.mkdir(parents=True, exist_ok=True)

    def create_logbook(self, workflow: Workflow) -> Logbook:
        """Create a new logbook for the given workflow."""
        # TODO: Can workflow.name be None?
        folder_name = self._unique_workflow_folder_name(workflow.name)
        return FolderLogbook(self._folder / folder_name)

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


class FolderLogbook(Logbook):
    """A logbook that stores a workflow's results and artifacts in a folder."""

    def __init__(self, folder: Path | str) -> None:
        self._folder = Path(folder)
        self._folder.mkdir(parents=False, exist_ok=False)
        self._log = Path(folder / "log.jsonl")
        self._log.touch(exist_ok=False)

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

    def save_artifact(
        self,
        task: Task,
        name: str,
        artifact: object,
        metadata: dict[str, object] | None = None,
        serialization_options: dict[str, object] | None = None,
    ) -> str:
        """Called to save an artifact."""
        self._append_log(
            {
                "event": "artifact",
                "task": task.name,
                "artifact_name": name,
                "artifact_type": artifact.type_,
                "artifact_metadata": metadata,
                "serialization_options": serialization_options,
            },
        )
        # TODO: store artifact
        # X filename = _unique_reference(
        # X   path,
        # X   partial(self._create_subpath, where=artifact_folder),
        # X   timestamp=False,
        # X   create_folder=False,
        # X )
        # X serialize(artifact, artifact_folder / filename, serialization_options)
        # X event = ArtifactEvent(name=name, path=filename, metadata=metadata)
        # X self.on_event(EventType.SAVE_ARTIFACT, event_data=event)
        # X return filename
