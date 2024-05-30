"""This module provides a task for extracting sweep experiment data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from laboneq_applications.utils.handle_helpers import (
    ACTIVE_RESET_PREFIX,
    CALIBRATION_TRACE_PREFIX,
    RESULT_PREFIX,
    parse_active_reset_handle,
    parse_calibration_trace_handle,
    parse_result_handle,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from laboneq_applications.tasks.datatypes import RunExperimentResults


def _extend_sweep_points_cal_traces(
    sweep_points: ArrayLike,
    num_cal_traces: int,
) -> np.ndarray:
    if sweep_points is None:
        sweep_points = [[]]
    if not (
        len(sweep_points) == 0
        or (
            isinstance(sweep_points, np.ndarray)
            and len(sweep_points.shape) == 2  # noqa: PLR2004
            and sweep_points.shape[0] <= 1
        )
        or (
            isinstance(sweep_points, list)
            and len(sweep_points) == 1
            and all(isinstance(e, (int, float)) for e in sweep_points[0])
        )
    ):
        raise ValueError("Sweep points are not a one dimensional list of numbers.")
    if len(sweep_points) == 0 or len(sweep_points[0]) == 0:
        return np.arange(num_cal_traces)
    dsp = sweep_points[0][1] - sweep_points[0][0] if len(sweep_points[0]) > 1 else 1
    start = 0 if len(sweep_points[0]) < 1 else sweep_points[0][-1] + dsp
    cal_traces_swpts = np.array([[start + i * dsp for i in range(num_cal_traces)]])
    return np.concatenate([sweep_points, cal_traces_swpts], axis=1)


@dataclass
class SingleQubitSweepResults:
    """A class to hold the sweep results and calibration traces for a single qubit.

    Attributes:
        sweep: The measured data for the sweep.
        calibration_traces: The measured calibration traces data, per state
        active_reset: The measured data for the active reset; the dimensions of the
            array correspond to the maximum repetiton number in the handles times the
            number of sweep points.
    """

    sweep_data: ArrayLike = field(default_factory=lambda: np.array([]))
    calibration_traces: dict[str, ArrayLike] = field(default_factory=dict)
    active_reset: dict[str, ArrayLike] = field(default_factory=dict)


@dataclass
class SweepResults:
    """A class to hold the sweep results and calibration.

    Attributes:
        sweep_points: The parameter values of the sweep, might be multidimensional and
            nested.
        sweep_points_calibration_traces: The parameter values of the sweep for
            calibration traces, might be multidimensional and nested.
        sweep_parameter_names: The name(s) of the sweep parameters.
        sweep_parameter_names_calibration_traces: The name(s) of the sweep parameters
            for calibration traces.
        single_qubit_data: The measured data, per qubit.
    """

    sweep_points: ArrayLike | None = None
    sweep_points_calibration_traces: ArrayLike | None = None
    sweep_parameter_names: list[str] | None = None
    sweep_parameter_names_calibration_traces: list[str] | None = None
    single_qubit_data: dict[str, SingleQubitSweepResults] = field(default_factory=dict)

    def _append_traces(
        self,
        sweep_points: ArrayLike,
        data: ArrayLike,
        traces: dict[str, ArrayLike],
    ) -> tuple[ArrayLike, ArrayLike] | None:
        if len(traces) == 0:
            return sweep_points, data
        number_of_states = len(traces)
        sweep_points = _extend_sweep_points_cal_traces(
            self.sweep_points,
            number_of_states,
        )
        if not isinstance(data, np.ndarray) and not isinstance(data, Sequence):
            data = [data]
        data = np.concatenate([data, list(traces.values())])
        return sweep_points, data

    def sweep_data(
        self,
        qubit_name: str,
        *,
        append_calibration_traces: bool = False,
    ) -> tuple[ArrayLike, ArrayLike] | tuple[None, None]:
        """Returns the sweep's data as tuple of sweep points and integration result.

        Args:
            qubit_name: The name of the qubit.
            append_calibration_traces: If True, the calibration traces are appended to
                the sweep data and additional sweep points are added for plotting.

        Returns:
            The sweep points and the integration result.
        """
        try:
            sweep_points = self.sweep_points
            data = self.single_qubit_data[qubit_name].sweep_data
        except KeyError:
            return (None, None)

        if append_calibration_traces:
            sweep_points, data = self._append_traces(
                self.sweep_points,
                data,
                self.single_qubit_data[qubit_name].calibration_traces,
            )
        return sweep_points, data

    def calibration_traces(
        self,
        qubit_name: str,
        state_names: str | Sequence[str],
    ) -> list[ArrayLike] | ArrayLike | None:
        """Returns the calibration trace data.

        Args:
            qubit_name: The name of the qubit.
            state_names: The name(s) of the state(s) for which to return the calibration
                traces.

        Returns:
            The calibration traces for the given state(s).
        """
        if single := isinstance(state_names, str):
            state_names = [state_names]
        result = []
        for state_name in state_names:
            try:
                result.append(
                    self.single_qubit_data[qubit_name].calibration_traces[state_name],
                )
            except KeyError:  # noqa: PERF203
                result.append(None)

        return result[0] if single else result

    def active_reset_data(
        self,
        qubit_name: str,
        tag_names: str | Sequence[str],
        *,
        append_calibration_traces: bool = False,
    ) -> list[ArrayLike] | ArrayLike | None:
        """Returns the measured data for the active reset.

        Args:
            qubit_name: The name of the qubit.
            tag_names: The name(s) of the tag for the active reset data.
            append_calibration_traces: If True, the calibration traces are appended to
                the active reset data and additional sweep points are added for
                plotting.

        Returns:
            The sweep points and the active reset data.
        """
        if single := isinstance(tag_names, str):
            tag_names = [tag_names]
        result = []
        for tag_name in tag_names:
            try:
                data = self.single_qubit_data[qubit_name].active_reset[tag_name]
                sweep_points = self.sweep_points
                if append_calibration_traces:
                    sweep_points, data = self._append_traces(
                        self.sweep_points,
                        data,
                        self.single_qubit_data[qubit_name].calibration_traces,
                    )
                result.append((sweep_points, data))
            except KeyError:  # noqa: PERF203
                result.append(None)
        return result[0] if single else result


def default_extract_sweep_results(
    results: RunExperimentResults,
    result_prefix: str = RESULT_PREFIX,
    calibration_trace_prefix: str = CALIBRATION_TRACE_PREFIX,
    active_reset_prefix: str = ACTIVE_RESET_PREFIX,
) -> SweepResults:
    """Extracts sweep experiment data from the results.

    This tasks extracts the sweep results and calibration traces from the results of a
    class of experiments using a single sweep and measuring calibration traces.

    The structure of the experiment is used for many tune-up style experiments.

    Args:
        results (RunExperimentResults): The results of the run task.
        result_prefix (str): The prefix for the main sweep results, if different from
            the default.
        calibration_trace_prefix (str): The prefix for the calibration traces, if
            different from the default.
        active_reset_prefix (str): The prefix for the active reset results, if different
            from the default.

    Returns:
        SweepResults: The sweep results and calibration traces.

    Example:
        ```python
        import numpy as np

        from laboneq_applications.tasks.datatypes import (
            AcquiredResult,
            RunExperimentResults,
        )
        from laboneq_applications.tasks.extract_sweep_results import (
            default_extract_sweep_results,
        )

        results = RunExperimentResults()
        results.acquired_results["my_result/qubit_1"] = AcquiredResult(
            data=[1, 2, 3, 4, 5],
            axis_name=["amplitude"],
            axis=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        results.acquired_results["my_result/qubit_2"] = AcquiredResult(
            data=[2, 2, 3, 4, 5],
            axis_name=["amplitude"],
            axis=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        )
        results.acquired_results["cal_trace/qubit_1/g"] = AcquiredResult(
            data=[2, 3, 4, 5, 6],
            axis_name=["amplitude_ct"],
            axis=[0.15, 0.2, 0.3, 0.4, 0.5],
        )
        results.acquired_results["active_reset/qubit_1/1"] = AcquiredResult(
            data=[4, 3, 4, 5, 6],
            axis_name=["amplitude"],
            axis=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        )
        sweep_results = default_extract_sweep_results(
            results,
            result_prefix="my_result"
        )

        np.testing.assert_array_equal(
            sweep_results.sweep_data("qubit_1"),
            ([0.1, 0.2, 0.3, 0.4, 0.5], [1, 2, 3, 4, 5]),
        )
        np.testing.assert_array_equal(
            sweep_results.sweep_data("qubit_2"),
            ([0.1, 0.2, 0.3, 0.4, 0.5], [2, 2, 3, 4, 5]),
        )
        np.testing.assert_array_equal(
            sweep_results.calibration_traces("qubit_1", "g"),
            [2, 3, 4, 5, 6],
        )
        np.testing.assert_array_equal(
            sweep_results.active_reset_data("qubit_1", "1"),
            ([0.1, 0.2, 0.3, 0.4, 0.5], [4, 3, 4, 5, 6]),
        )
        np.testing.assert_array_equal(
            sweep_results.sweep_points,
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )
        np.testing.assert_array_equal(
            sweep_results.sweep_points_calibration_traces,
            [0.15, 0.2, 0.3, 0.4, 0.5],
        )
        assert sweep_results.sweep_parameter_names == ["amplitude"]
        assert sweep_results.sweep_parameter_names_calibration_traces == (
            ["amplitude_ct"]
        )
        ```
    """
    sweep_results = SweepResults()
    acq = results.acquired_results

    def set_sweep_axis(handle: str, attr_suffix: str) -> None:
        data_attr_name = f"sweep_points{attr_suffix}"
        names_attr_name = f"sweep_parameter_names{attr_suffix}"
        if getattr(sweep_results, data_attr_name) is None:
            setattr(sweep_results, data_attr_name, np.array(acq[handle].axis))
            setattr(sweep_results, names_attr_name, acq[handle].axis_name)
        elif not np.array_equal(
            acq[handle].axis,
            getattr(sweep_results, data_attr_name),
        ) or acq[handle].axis_name != getattr(sweep_results, names_attr_name):
            raise ValueError(f"Sweep points are not consistent for handle '{handle}'.")

    def get_qubit(name: str) -> SingleQubitSweepResults:
        return sweep_results.single_qubit_data.setdefault(
            name,
            SingleQubitSweepResults(),
        )

    for handle in acq:
        data = acq[handle].data
        qubit_name = parse_result_handle(handle, prefix=result_prefix)
        if qubit_name:
            set_sweep_axis(handle, "")
            get_qubit(qubit_name).sweep_data = data
            continue
        qubit_name, state = parse_calibration_trace_handle(
            handle,
            prefix=calibration_trace_prefix,
        )
        if qubit_name and state:
            set_sweep_axis(handle, "_calibration_traces")
            get_qubit(qubit_name).calibration_traces[state] = data
            continue
        qubit_name, tag = parse_active_reset_handle(handle, prefix=active_reset_prefix)
        if qubit_name and tag:
            get_qubit(qubit_name).active_reset[tag] = data
            continue
    return sweep_results
