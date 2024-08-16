"""Tests for laboneq_applications.logbook.serializer."""

from __future__ import annotations

import json
from typing import IO

import matplotlib.pyplot as plt
import numpy as np
import PIL
import pydantic
import pytest

from laboneq_applications.logbook.serializer import SerializeOpener, serialize


class FolderOpener(SerializeOpener):
    def __init__(self, store_path, options):
        self._store_path = store_path
        self._options = options
        self._descriptions = {}

    def open(
        self,
        ext: str,
        *,
        encoding: str | None = None,
        suffix: str | None = None,
        description: str | None = None,
        binary: bool = False,
    ) -> IO:
        mode = "wb" if binary else "w"
        suffix = f"-{suffix}" if suffix else ""
        filename = f"foo{suffix}.{ext}"
        if description:
            self._descriptions[filename] = description
        path = self._store_path / filename
        return path.open(mode=mode, encoding=encoding)

    def options(self):
        return self._options


class FolderFixture:
    def __init__(self, store_path):
        self._store_path = store_path
        self.options = {}
        self.opener = FolderOpener(self._store_path, self.options)

    def files(self):
        store_path = self._store_path
        return sorted(str(p.relative_to(store_path)) for p in store_path.iterdir())

    def descriptions(self):
        return self.opener._descriptions

    def path(self, filename):
        return self._store_path / filename


@pytest.fixture()
def store(tmp_path):
    store_path = tmp_path / "store"
    store_path.mkdir()
    return FolderFixture(store_path)


class TestSerializeStr:
    def test_serialize(self, store):
        serialize("Test me!", store.opener)

        assert store.files() == ["foo.txt"]
        assert store.path("foo.txt").read_text() == "Test me!"
        assert store.descriptions() == {}

    def test_serialize_nonascii(self, store):
        serialize("Tést nön-ascïï!", store.opener)

        assert store.files() == ["foo.txt"]
        assert store.path("foo.txt").read_text(encoding="utf-8") == "Tést nön-ascïï!"
        assert store.descriptions() == {}


class TestSerializeBytes:
    def test_serialize(self, store):
        serialize(b"Bytes \xff\x00", store.opener)

        assert store.files() == ["foo.dat"]
        assert store.path("foo.dat").read_bytes() == b"Bytes \xff\x00"
        assert store.descriptions() == {}


class TestSerializePydanticModel:
    def test_serialize(self, store):
        class SimpleModel(pydantic.BaseModel):
            i: int
            s: str

        serialize(SimpleModel(i=5, s="foo"), store.opener)

        assert store.files() == ["foo.json"]
        assert json.loads(store.path("foo.json").read_text()) == {"i": 5, "s": "foo"}
        assert store.descriptions() == {}


class TestSerializePILImage:
    def _image(self):
        return PIL.Image.effect_mandelbrot((10, 10), (-2.0, -1.5, 1.5, 1.5), 9)

    def test_serialize(self, store):
        im = self._image()

        serialize(im, store.opener)

        assert store.files() == ["foo.png"]
        im2 = PIL.Image.open(store.path("foo.png"))
        np.testing.assert_equal(np.asarray(im), np.asarray(im2))
        assert store.descriptions() == {}

    def test_serialize_format(self, store):
        im = self._image()
        store.options["format"] = "webp"
        store.options["lossless"] = True

        serialize(im, store.opener)

        assert store.files() == ["foo.webp"]
        im2 = PIL.Image.open(store.path("foo.webp"))
        im = im.convert("RGB")
        np.testing.assert_equal(np.asarray(im), np.asarray(im2))
        assert store.descriptions() == {}

    def test_serialize_jpg(self, store):
        # the serializer has special logic to convert the JPG extension
        # to the PIL JPEG format name, so we test that here.
        im = self._image()
        store.options["format"] = "jpg"

        serialize(im, store.opener)

        assert store.files() == ["foo.jpg"]
        im2 = PIL.Image.open(store.path("foo.jpg"))
        # JPEGs are lossy, so we only check the shape
        assert np.asarray(im2).shape == (10, 10)
        assert store.descriptions() == {}


class TestSerializeMatplotlibFigure:
    def _figure(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        x = np.linspace(0, 2 * np.pi, 100)
        y = 2 * np.cos(x)
        ax.plot(x, y)

        ax.set_title("Testing. 1 ... 2 ... 3 ...")
        return fig

    def test_serialize(self, store):
        fig = self._figure()

        serialize(fig, store.opener)

        assert store.files() == ["foo.png"]
        im = PIL.Image.frombytes(
            "RGBA",
            fig.canvas.get_width_height(),
            fig.canvas.buffer_rgba(),
        )
        im2 = PIL.Image.open(store.path("foo.png"))
        np.testing.assert_equal(np.asarray(im), np.asarray(im2))
        assert store.descriptions() == {}

    def test_serialize_format(self, store):
        fig = self._figure()
        store.options["format"] = "webp"
        store.options["lossless"] = True

        serialize(fig, store.opener)

        assert store.files() == ["foo.webp"]
        im = PIL.Image.frombytes(
            "RGBA",
            fig.canvas.get_width_height(),
            fig.canvas.buffer_rgba(),
        )
        im2 = PIL.Image.open(store.path("foo.webp"), formats=["webp"])
        im2.putalpha(255)
        np.testing.assert_equal(np.asarray(im), np.asarray(im2))
        assert store.descriptions() == {}


class TestSerializeNumpyArray:
    def test_serialize(self, store):
        serialize(np.array([1, 2, 3]), store.opener)

        assert store.files() == ["foo.npy"]
        arr = np.load(store.path("foo.npy"))
        np.testing.assert_equal(np.array([1, 2, 3]), arr)
        assert store.descriptions() == {}


class TestSerializeUnsupportedObject:
    def test_serialize(self, store):
        with pytest.raises(TypeError) as err:
            serialize(object(), store.opener)

        assert str(err.value) == (
            "Type <class 'object'> not supported by the serializer."
        )
        assert store.files() == []
        assert store.descriptions() == {}
