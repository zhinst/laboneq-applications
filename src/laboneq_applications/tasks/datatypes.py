"""Provide some common types for tasks."""

from __future__ import annotations

from collections.abc import Collection, ItemsView, Iterable, Iterator, ValuesView
from collections.abc import KeysView as BaseKeysView
from dataclasses import dataclass, field
from io import StringIO
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

use_rich_pprint: bool = True
try:
    from rich.console import Console
    from rich.pretty import pprint as rich_pprint
except ImportError:
    from pprint import pprint as pprint_pprint

    use_rich_pprint = False


class HasAsStrDict(Protocol):
    """A protocol for classes that have a method to return a dictionary representation.

    The dictionary representation is used to pretty-print the object using the pprint
    function. The dictionary should contain all attributes that should be printed.
    Not to be confused with the __dict__ attribute, which contains the actual attributes
    of the object, including methods and private attributes.
    """

    def _as_str_dict(self) -> dict[str, Any]: ...


T = TypeVar("T", bound=HasAsStrDict)
P = ParamSpec("P")


def classformatter(cls: type[T]) -> type[T]:
    """A decorator to customize the string representation of class instances.

    This decorator overwrites the __str__ and __format__ methods of the decorated class.
    The new __str__ method pretty-prints the instance using the pprint function,
    ensuring a visually appealing output on compatible terminals. The __format__ method
    is overridden to return the class's original __repr__ representation. Also,
    the `_repr_pretty_` method is added to support pretty-printing in Jupyter notebooks.

    In contrast to the similar decorator with the same name from laboneq, this creates
    the string representation based on a dictionary provided by the class'
    (   )`_as_str_dict()`.

    If the global variable `use_rich_pprint` is `True`, the rich library will be used,
    otherwise `pprint.pprint`.

    Args:
        cls (type): The class to be decorated.

    Returns:
        type: The decorated class with modified __str__ and __format__ methods.
    """

    def new_str(self: T) -> str:
        as_dict = self._as_str_dict()
        with StringIO() as buffer:
            if use_rich_pprint:
                console = Console(file=buffer)
                rich_pprint(
                    as_dict,
                    console=console,
                    expand_all=True,
                    indent_guides=True,
                )
            else:
                pprint_pprint(as_dict, stream=buffer)  # noqa: T203
            return buffer.getvalue()

    def new_format(self: T, _: object) -> str:
        return format(self._as_str_dict())

    def repr_pretty(self, p, _cycle):  # noqa: ANN001, ANN202
        # For Notebooks
        p.text(str(self))

    cls.__str__ = new_str
    cls.__format__ = new_format
    cls._repr_pretty_ = repr_pretty

    return cls


def _check_prefix(keys: set[str], separator: str) -> None:
    """Check if there is no key which is also a prefix of another.

    Args:
        keys: A set of keys.
        separator: The symbol used to separate levels in the keys.

    Raises:
        ValueError: If there is a key which is also the prefix of another.
    """
    # Sort. Appending the separator helps to avoid cases if the key name
    # contains a character which is sorts before the separator, like
    # "a/a", "a/a/b" and "a/a.".
    sorted_keys = sorted(k + separator for k in keys)
    # Raise if a key is the prefix of the following key
    for i in range(len(sorted_keys) - 1):
        k1, k2 = sorted_keys[i], sorted_keys[i + 1]
        if k2.startswith(k1):
            raise ValueError(f"Key '{k1[:-1]}' is a prefix of '{k2[:-1]}'.")


def _check_attrs(keys: set[str], attrs: list[str], separator: str) -> None:
    """Check if no subkey matches an attribute of the class.

    Args:
        keys: A set of keys.
        attrs: A list of attribute names.
        separator: The symbol used to separate levels in the keys.

    Raises:
        ValueError: If a key is also an attribute of the class.
    """
    attrs_set = set(attrs)
    subkeys = {subkey for key in keys for subkey in key.split(separator)}
    if attrs_set & subkeys:
        raise ValueError(
            f"Handles {subkeys & attrs_set} aren't allowed names.",
        )


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


@classformatter
class AttributeWrapper(Collection[str]):
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

    class KeysView(BaseKeysView[str]):
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

    def __init__(
        self,
        data: dict[str, Any] | None,
        path: str | None = None,
        separator: str | None = None,
    ) -> None:
        super().__init__()
        self._data = data or {}
        self._path = path or ""
        self._separator = separator if separator is not None else "/"
        self._key_cache: set[str] = set()
        keys_set = set(self._data.keys())
        _check_attrs(keys_set, dir(self), self._separator)
        _check_prefix(keys_set, self._separator)
        self._key_cache = {
            self._get_subkey(k) for k in self._data if k.startswith(self._path)
        }

    # Partial Mapping interface
    def __len__(self) -> int:
        return len(self._key_cache)

    def __iter__(self) -> Iterator[str]:
        return iter(self._key_cache)

    def __getitem__(self, key: object) -> object:
        if not isinstance(key, str):
            raise TypeError(f"Key {key} has to be of type str.")
        path = self._add_path(key)
        try:
            return self._data[path]
        except KeyError as e:
            path_sep = path + self._separator
            if not any(k.startswith(path_sep) for k in self._data):
                raise KeyError(f"Key '{self._path}' not found in the data.") from e
            return AttributeWrapper(self._data, path)

    def __contains__(self, key: object) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def _keys(self) -> AttributeWrapper.KeysView:
        """A set-like object providing a view on the available attributes."""
        return AttributeWrapper.KeysView(self)

    def _items(self) -> ItemsView[str, Any]:
        """A set-like object providing a view on wrapper's items."""
        return ItemsView(self)

    def _values(self) -> ValuesView[Any]:
        """An object providing a view on the wrapper's values."""
        return ValuesView(self)

    # End of Mapping interface

    def __getattr__(self, key: str) -> object:
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

    def _as_str_dict(self) -> dict[str, Any]:
        return {
            key: (attr._as_str_dict() if isinstance(attr, AttributeWrapper) else attr)
            for key, attr in ((key, getattr(self, key)) for key in self._key_cache)
        }

    def __dir__(self) -> Iterable[str]:
        return list(super().__dir__()) + list(self._key_cache)

    def __repr__(self) -> str:
        return (
            f"AttributeWrapper(data={self._data!r}, path={self._path!r}, "
            f"separator={self._separator!r})"
        )


ErrorList = list[tuple[list[int], str, str]]


@classformatter
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
        errors: ErrorList | None = None,
    ):
        super().__init__(data)
        self._neartime_callbacks = neartime_callbacks or {}
        self._errors = errors or []
        self._key_cache.update(["neartime_callbacks", "errors"])

    @property
    def neartime_callbacks(self) -> AttributeWrapper:
        """The results of the near-time user callbacks."""
        return AttributeWrapper(self._neartime_callbacks)

    @property
    def errors(self) -> ErrorList:
        """The errors that occured during running the experiment."""
        return self._errors

    def __getitem__(self, key: object) -> AttributeWrapper | ErrorList | object:
        if key == "neartime_callbacks":
            return AttributeWrapper(self._neartime_callbacks)
        if key == "errors":
            return self.errors or []
        return super().__getitem__(key)

    def __dir__(self):
        return super().__dir__() + list(self._key_cache)

    def _as_str_dict(self) -> dict[str, Any]:
        return {
            key: (attr._as_str_dict() if isinstance(attr, AttributeWrapper) else attr)
            for key, attr in ((key, getattr(self, key)) for key in self._key_cache)
        } | {
            "neartime_callbacks": self._neartime_callbacks,
            "errors": self.errors,
        }

    def __repr__(self) -> str:
        return (
            f"RunExperimentResults(data={self._data!r}, "
            f"near_time_callbacks={self._neartime_callbacks!r}, errors={self.errors!r},"
            f" path = {self._path!r}, separator={self._separator!r})"
        )
