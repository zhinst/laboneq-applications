"""Provide some common types for tasks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from collections.abc import KeysView as BaseKeysView
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclass
class AcquiredResult:
    """This class represents the results acquired for a single result handle.

    The acquired result consists of actual data, axis name(s) and one or more axes,
    and resembles the structure of a LabOne Q result with the same name.

    Attributes:
        data (ArrayLike): A multidimensional `numpy` array, where each dimension
            corresponds to a sweep loop nesting level, the outermost sweep being the
            first dimension.
        axis_name (list[str | list[str]]): A list of axis names.
            Each element may be either a string or a list of strings.
        axis (list[ArrayLike | list[ArrayLike]]): A list of axis grids.
            Each element may be either a 1D numpy array or a list of such arrays.
    """

    data: ArrayLike | None = None
    axis_name: list[str | list[str]] = field(default_factory=list)
    axis: list[ArrayLike | list[ArrayLike]] = field(default_factory=list)


class AttributeWrapper(Mapping[str, Any]):
    """A wrapper for accessing members of a dict in dot notation.

    Input data is a dict, where each key is a string, where levels are separated by
    slashes. The wrapper allows to access the data in dot notation, where the levels
    are separated by dots. The wrapper also provides a read-only dict interface.

    Attributes:
        data: The dict to wrap.
        path: The path to the current level in the dict.

    Example:
    ```python
    data = {
        "cal_trace/q0/g": 12345,
    }
    wrapper = AttributeWrapper(data)
    assert wrapper.cal_trace.q0.g == 12345
    assert len(wrapper.cal_trace) == 1
    assert set(wrapper.cal_trace.keys()) == {"q0"}
    ```
    """

    class KeysView(BaseKeysView):
        """A view of the keys of an AttributeWrapper."""

        def __str__(self) -> str:
            return f"AttributesView({list(self._mapping.keys())})"

        def __repr__(self) -> str:
            return str(self)

    def _get_subkey(self, key: str) -> str:
        if len(self._path) == 0:
            return key.split(self._separator, 1)[0]
        prefix_len = len(self._path)
        return key[prefix_len + 1 :].split(self._separator, 1)[0]

    def _get_subkeys(self, key: str) -> str:
        if len(self._path) == 0:
            return key.replace(self._separator, ".")
        prefix_len = len(self._path)
        return key[prefix_len + 1 :].replace(self._separator, ".")

    def _add_path(self, key: str) -> str:
        return (self._path + self._separator + key) if self._path else key

    def __init__(self, data: dict[str, Any], path: str | None = None) -> None:
        super().__init__()
        self._data = data
        self._path = path or ""
        self._separator = "/"
        self._key_cache = {
            self._get_subkey(k) for k in self._data if k.startswith(self._path)
        }
        if not self._key_cache:
            raise KeyError(f"Key '{self._path}' not found in the data.")

    # Mapping interface
    def __len__(self) -> int:
        return len(self._key_cache)

    def __iter__(self) -> Iterator[str]:
        return iter(self._key_cache)

    def __getitem__(self, key: str) -> AttributeWrapper:
        path = self._add_path(key)
        try:
            return self._data[path]
        except KeyError:
            return AttributeWrapper(self._data, path)

    def keys(self) -> AttributeWrapper.KeysView[str]:
        """A set-like object providing a view on the available attributes."""
        return AttributeWrapper.KeysView(self)

    # End of Mapping interface

    def __getattr__(self, key: str) -> AttributeWrapper:
        try:
            return self.__getitem__(key)
        except KeyError as e:
            raise AttributeError(
                f"Key '{self._add_path(key)}' not found in the data.",
            ) from e

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AttributeWrapper):
            return NotImplemented
        return (
            self._data == value._data
            and self._path == value._path
            and self._separator == value._separator
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(
            {
                self._get_subkeys(k): v
                for k, v in self._data.items()
                if k.startswith(self._path)
            },
        )

    def __dir__(self):
        return super().__dir__() + list(self._key_cache)


class RunExperimentResults(AttributeWrapper):
    """The results of running an experiment.

    The results are accessible via dot notation, where the levels are separated by
    slashes in the handle.

    Example:
    ```python
    acquired = AcquiredResult(
        data=numpy.array([1, 2, 3]),
        axis_name=["Amplitude"],
        axis=[numpy.array([0, 1, 2])],
    )
    results = RunExperimentResults(data={"cal_trace/q0/g": acquired})
    assert results.cal_trace.q0.g is acquired
    assert list(results.cal_trace.q0.keys()) == ["g"]
    ```

    Attributes:
        data:
            The extracted sweep results from the experiment. The keys
            are the acquisition handles.
        neartime_callbacks:
            The results of the near-time user callbacks. The keys are the
            names of the near-time callback functions. The values are the
            list of results in execution order.
        errors:
            The errors that occurred during running the experiment. Each
            item in the list is a tuple of
            `(sweep indices, realt-time section uid, error message)`.
    """

    def __init__(
        self,
        data: dict[str, AcquiredResult],
        neartime_callbacks: dict[str, list[Any]] | None = None,
        errors: list[tuple[list[int], str, str]] | None = None,
    ):
        super().__init__(data)
        self._neartime_callbacks = neartime_callbacks or {}
        self.errors = errors
        self._key_cache.update(["neartime_callbacks", "errors"])

    @property
    def neartime_callbacks(self) -> dict[str, list[Any]]:
        """The results of the near-time user callbacks."""
        return AttributeWrapper(self._neartime_callbacks)

    def __getitem__(self, key: str) -> AttributeWrapper:
        if key == "neartime_callbacks":
            return AttributeWrapper(self._neartime_callbacks)
        if key == "errors":
            return AttributeWrapper(self._errors)
        return super().__getitem__(key)

    def __str__(self) -> str:
        return str(
            {
                self._get_subkeys(k): v
                for k, v in self._data.items()
                if k.startswith(self._path)
            }
            | {
                "neartime_callbacks": self._neartime_callbacks,
                "errors": self.errors,
            },
        )
