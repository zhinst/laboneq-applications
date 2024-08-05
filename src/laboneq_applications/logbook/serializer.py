"""Logbook serializer for artifacts."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from PIL.Image import Image
from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path


def serialize(  # noqa: C901
    artifact: object,
    filename: Path,
    serialization_options: dict[str, object] | None,
) -> None:
    """Serialize an artifact to a file.

    Supported artifact types:
    - pydantic.BaseModel
    - bytes
    - str
    - PIL.Image.Image
    - matplotlib.pyplot.Figure
    - numpy.ndarray

    Args:
        artifact: The artifact to serialize.
        filename: The filename to save the artifact to.
        serialization_options: Options to pass to the respective save function
    """
    extension = filename.suffix
    if extension.startswith("."):
        extension = extension[1:]
    """Serialize an artifact to a file."""
    bytes_content: bytes | None = None
    text: str | None = None
    if serialization_options is None:
        serialization_options = {}
    if isinstance(artifact, BaseModel):
        text = artifact.model_dump_json(**serialization_options)
    elif isinstance(artifact, bytes):
        bytes_content = artifact
    elif isinstance(artifact, str):
        text = artifact
    elif isinstance(artifact, Image):
        buffer = io.BytesIO()
        # Convert the file extension to an appropriate format
        # For example, ".jpg" to "JPEG", ".png" to "PNG"
        image_format = extension.upper()
        if image_format == "JPG":
            image_format = "JPEG"  # PIL uses 'JPEG' instead of 'JPG'
        artifact.save(buffer, format=image_format, **serialization_options)
        bytes_content = buffer.getvalue()
    elif isinstance(artifact, plt.Figure):
        buffer = io.BytesIO()
        artifact.savefig(buffer, format=extension, **serialization_options)
        bytes_content = buffer.getvalue()
    elif isinstance(artifact, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, artifact, **serialization_options)
        bytes_content = buffer.getvalue()
    else:
        raise TypeError(f"Cannot serialize artifact of type {type(artifact)}")
    if bytes_content is not None:
        with open(filename, "wb") as file:
            file.write(bytes_content)
    elif text is not None:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
