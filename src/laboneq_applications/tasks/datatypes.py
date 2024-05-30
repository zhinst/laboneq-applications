"""Provide some common types for tasks."""

from __future__ import annotations

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
