"""Provide some common types for tasks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
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


@dataclass
class RunExperimentResults:
    """The results of running an experiment.

    Attributes:
        acquired_results:
            The extracted sweep results from the experiment. The keys
            are the acquisition handles.
        neartime_callback_results:
            The results of the near-time user callbacks. The keys are the
            names of the near-time callback functions. The values are the
            list of results in execution order.
        execution_errors:
            The errors that occurred during running the experiment. Each
            item in the list is a tuple of
            `(sweep indices, realt-time section uid, error message)`.
    """

    acquired_results: dict[str, AcquiredResult] = field(default_factory=dict)
    neartime_callback_results: dict[str, list[Any]] = field(default=None)
    execution_errors: list[tuple[list[int], str, str]] = field(default=None)

    @property
    def results(self) -> AttributeWrapper:
        """The acquired results of the experiment, accessible in dot notation.

        Returns:
            A wrapper for the acquired results, which allows to access them in
            dot notation, where the levels are separated by slashes in the handle.

        Example:
        ```python
        results = RunExperimentResults(
            acquired_results={
                "cal_trace/q0/g": AcquiredResult(
                    data=numpy.array([1, 2, 3]),
                    axis_name=["Amplitude"],
                    axis=[numpy.array([0, 1, 2])],
                ),
            },
        )
        assert results.cal_trace.q0.g is results.acquired_results["cal_trace/q0/g"]
        assert list(results.cal_trace.q0.keys()) == ["g"]
        ```
        """
        return AttributeWrapper(self.acquired_results)


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

    def _get_subkey(self, key: str) -> str:
        prefix_len = len(self.path)
        return key[prefix_len + 1 :].split(self.separator, 1)[0]

    def _add_path(self, key: str) -> str:
        return (self.path + self.separator + key) if self.path else key

    def __init__(self, data: dict[str, Any], path: str | None = None) -> None:
        self.data = data
        self.path = path or ""
        self.separator = "/"
        self._key_cache = [
            self._get_subkey(k) for k in self.data if k.startswith(self.path)
        ]
        if not self._key_cache:
            raise AttributeError(f"Key '{self.path}' not found in the data.")

    def __len__(self) -> int:
        return len(self._key_cache)

    def __iter__(self) -> Iterator[str]:
        return iter(self._key_cache)

    def keys(self) -> list[str]:
        """Return the keys of the current path in the data."""
        return (k for k in self._key_cache)

    def __getitem__(self, key: str) -> AttributeWrapper:
        path = self._add_path(key)
        try:
            return self.data[path]
        except KeyError:
            return AttributeWrapper(self.data, path)

    def __getattr__(self, key: str) -> AttributeWrapper:
        return self.__getitem__(key)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AttributeWrapper):
            return NotImplemented
        return (
            self.data == value.data
            and self.path == value.path
            and self.separator == value.separator
        )
