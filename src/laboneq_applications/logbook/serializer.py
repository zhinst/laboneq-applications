"""Logbook serializer for artifacts.

See the functions below named `serialize_` for the list of types that can be serialized.
"""

from __future__ import annotations

import abc
from functools import singledispatch
from typing import TYPE_CHECKING

import matplotlib.figure as mpl_figure
import numpy as np
import PIL
import pydantic

if TYPE_CHECKING:
    from typing import IO


class SerializeOpener(abc.ABC):
    """An protocol allowing serializers to open files and access options.

    Serializers need to write out potentially multiple files without
    knowing precisely where the files will end up.

    Simultaneously the caller of serialize (e.g. the logbook) needs to
    keep a record of which files were created.

    This class allows all of this to happen by abstracting away the file
    creation interface.
    """

    @abc.abstractmethod
    def open(
        self,
        ext: str,
        *,
        encoding: str | None = None,
        suffix: str | None = None,
        description: str | None = None,
        binary: bool = False,
    ) -> IO:
        """Return an open file handle.

        Arguments:
            ext:
                The file extension to use (without a starting period).
            encoding:
                The encoding to use for text files.
            suffix:
                A suffix to add to the name of the file before the extension.
                This allows serializers that save multiple files to distinguish
                the files saved in a human-readable fashion.
            description:
                A description of the file contents. For example, a serializer
                saving a figure might save a `.png` file with the description
                "The plotted figure." and a `.json` file with the description
                "Metadata for the plot.".
            binary:
                If true, files are opened for writing in binary mode. If false,
                the default, files are opened in text mode.
        """

    @abc.abstractmethod
    def options(self) -> dict:
        """Return the serialization options."""


@singledispatch
def serialize(obj: object, opener: SerializeOpener) -> None:
    """Serialize an object.

    Arguments:
        obj:
            The object to serialize.
        opener:
            A `SerializeOpener` for retrieving options and opening
            files to write objects to.
    """
    raise TypeError(f"Type {type(obj)!r} not supported by the serializer.")


@serialize.register
def serialize_str(obj: str, opener: SerializeOpener) -> None:
    """Serialize a Python `str` object.

    String objects are saved as a text file with extension `.txt` and UTF-8 encoding.

    No options are supported.
    """
    with opener.open("txt", encoding="utf-8") as f:
        f.write(obj)


@serialize.register
def serialize_bytes(obj: bytes, opener: SerializeOpener) -> None:
    """Serialize a Python `bytes` object.

    Bytes objects are saved as a binary file with extension `.dat`.

    No options are supported.
    """
    with opener.open("dat", binary=True) as f:
        f.write(obj)


@serialize.register
def serialize_pydantic_model(obj: pydantic.BaseModel, opener: SerializeOpener) -> None:
    """Serialize a Pydantic model.

    Pydantic models are saved as JSON file with extension `.json`.

    Any options are passed directly as keyword arguments to the Pydantic model's
    `..model_dump_json` method.
    """
    json_txt = obj.model_dump_json(**opener.options())
    with opener.open("json") as f:
        f.write(json_txt)


@serialize.register
def serialize_pil_image(obj: PIL.Image.Image, opener: SerializeOpener) -> None:
    """Serialize a PIL image.

    PIL images are saved with `PIL.Image.save`.

    The format to save in is passed in the `format` option which defaults to `png`.
    The format `jpg` is automatically converted to `jpeg` for PIL.

    The remaining options are passed directly to `PIL.Image.save` as keyword
    arguments.
    """
    options = opener.options()
    ext = options.pop("format", "png")

    # Determine the PIL image format from the extension:
    image_format = ext.upper()
    if image_format == "JPG":
        image_format = "JPEG"

    with opener.open(ext, binary=True) as f:
        obj.save(f, format=image_format, **options)


@serialize.register
def serialize_matplotlib_figure(
    obj: mpl_figure.Figure,
    opener: SerializeOpener,
) -> None:
    """Serialize a matplotlib Figure.

    Matplotlib figures are saved with `matplotlib.figure.Figure.savefig`.

    The format to save in is passed in the `format` option which defaults to `png`.

    The remaining options are passed as the `pil_kwargs` argument to `.savefig`.
    """
    options = opener.options()
    ext = options.pop("format", "png")

    with opener.open(ext, binary=True) as f:
        obj.savefig(f, format=ext, pil_kwargs=options)


@serialize.register
def serialize_numpy_array(obj: np.ndarray, opener: SerializeOpener) -> None:
    """Serialize a NumPy `ndarray`.

    NumPy arrays are saved with `numpy.save` and the extension `.npy`.

    Any options are passed directly as keyword arguments to `.save`.
    """
    with opener.open("npy", binary=True) as f:
        np.save(f, obj, allow_pickle=False, **opener.options())
