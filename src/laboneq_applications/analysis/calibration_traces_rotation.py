"""This module contains functions for rotating raw experiment data.

The raw data needs to be translated into qubit populations. This is done either by
using principal-component analysis or rotating and projecting the data along the line
in the complex plane given by data obtained from calibration traces. The latter is
obtained by measuring the qubit after preparing it in a known state, for instance
g, e,  or f.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from laboneq_applications import dsl, workflow
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_result,
)
from laboneq_applications.experiments.options import TuneupAnalysisOptions

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from laboneq_applications.tasks.run_experiment import RunExperimentResults
    from laboneq_applications.typing import Qubits, QubitSweepPoints


def principal_component_analysis(raw_data: ArrayLike) -> ArrayLike:
    """Rotates and projects 1D data using principal component analysis (PCA).

    Args:
        raw_data: Array of complex data corresponding to the results of an
            integrated average result, usually of dimension (nr_sweep_points, 1).

    Returns:
        data array after PCA with the same dimension as raw_data.
    """
    real, imag = np.real(raw_data), np.imag(raw_data)

    # translate each column in the data by its mean
    mean_real, mean_imag = np.mean(real), np.mean(imag)
    trans_real, trans_imag = real - mean_real, imag - mean_imag
    row_data_trans = np.array([trans_real, trans_imag])

    # compute the covariance 2x2 matrix
    cov_matrix = np.cov(row_data_trans)

    # find eigenvalues and eigenvectors of the covariance matrix
    [eigvals, eigvecs] = np.linalg.eig(cov_matrix)

    # compute the transposed feature vector
    row_feature_vector = np.array(
        [
            (eigvecs[0, np.argmin(eigvals)], eigvecs[1, np.argmin(eigvals)]),
            (eigvecs[0, np.argmax(eigvals)], eigvecs[1, np.argmax(eigvals)]),
        ],
    )

    # compute final, projected data; only the first row is of interest (it is the
    # principal axis
    final_data = np.dot(row_feature_vector, row_data_trans)
    return final_data[1, :]


def calculate_rotation_matrix(
    delta_i: float | ArrayLike,
    delta_q: float | ArrayLike,
) -> NDArray[np.float64]:
    """Calculates the matrix that rotates the data to lie along the Q-axis.

    Input can be either the I and Q coordinates of the zero cal_point or
    the difference between the 1 and 0 cal points.

    Args:
        delta_i: difference between the real parts of the first and second
            cal-state data, usually of dimension (nr_sweep_points, 1).
        delta_q: difference between the imaginary parts of the first and second
            cal-state data, usually of dimension (nr_sweep_points, 1).

    Returns:
        the 2x2 rotation matrix as a numpy array
    """
    angle = np.arctan2(delta_q, delta_i)
    return np.transpose(
        np.array([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]]),
    )


def rotate_data_to_cal_trace_results(
    raw_data: ArrayLike,
    raw_data_cal_pt_0: ArrayLike | complex,
    raw_data_cal_pt_1: ArrayLike | complex,
) -> ArrayLike:
    """Rotates and projects the raw data.

    The raw data is projected onto the line in the IQ plane between
    two calibration points, then normalised to the distance in the IQ plane between
    the two calibration points.
    The calibration points are the results of preparing two transmon states.

    If more than one calibration point per state
    (i.e. len(raw_data_cal_pt_0) > 1), the mean of the cal points will be taken.

    Args:
        raw_data: array of complex data corresponding to the results of an
            integrated average result, usually of dimension (nr_sweep_points, 1).
        raw_data_cal_pt_0: array of complex data or a single complex number
            corresponding to the data recorded when the qubit is prepared in the first
            calibration state. "First" here means the lowest transmon state.
            For example, g if the cal traces measured were g and e or g and f;
            e if the cal traces measured were e an f
        raw_data_cal_pt_1: array of complex data or a single complex number
            corresponding to the data recorded when the qubit is prepared in the second
            calibration state. "Second" here means the highest transmon state.
            For example, e if the cal traces measured were g and e;
            f if the cal traces measured were g and f or e an f

    Returns:
        rotated, projected, and normalised real-data array with the same dimension as
        raw_data.
    """
    data_iq = np.array([np.real(raw_data), np.imag(raw_data)])
    data_cal0 = np.array(
        [np.mean(np.real(raw_data_cal_pt_0)), np.mean(np.imag(raw_data_cal_pt_0))],
    )
    data_cal1 = np.array(
        [np.mean(np.real(raw_data_cal_pt_1)), np.mean(np.imag(raw_data_cal_pt_1))],
    )

    # Translate the data
    trans_data = data_iq - np.repeat(data_cal0[:, np.newaxis], data_iq.shape[1], axis=1)

    # Rotate the data
    diff_data_cal = data_cal1 - data_cal0
    mtx = calculate_rotation_matrix(diff_data_cal[0], diff_data_cal[1])
    rotated_data = mtx @ trans_data

    # Normalize the data
    one_zero_dist = np.sqrt(diff_data_cal[0] ** 2 + diff_data_cal[1] ** 2)
    normalised_data = rotated_data / one_zero_dist

    return normalised_data[0]


def _extend_sweep_points_cal_traces(
    sweep_points: ArrayLike,
    num_cal_traces: int = 0,
) -> np.ndarray:
    """Extends sweep_points by the number of calibration traces.

    The sweep_points are extended in increments of sweep_points[1] - sweep_points[0],
    or 1 if sweep_points contains a single entry.

    Arguments:
        sweep_points: the array of sweep points
        num_cal_traces: the number of calibration traces

    Returns:
        the extended sweep points array
    """
    if num_cal_traces == 0:
        return sweep_points

    dsp = sweep_points[1] - sweep_points[0] if len(sweep_points) > 1 else 1
    cal_traces_swpts = np.array(
        [sweep_points[-1] + (i + 1) * dsp for i in range(num_cal_traces)],
    )
    return np.concatenate([sweep_points, cal_traces_swpts])


def calculate_population_1d(
    raw_data: ArrayLike,
    sweep_points: ArrayLike,
    calibration_traces: list[ArrayLike | complex] | None = None,
    *,
    do_pca: bool = False,
) -> dict:
    """Rotate and project 1D data.

    The data is projected along the line between the points in the calibration traces.

    Arguments:
        raw_data: array of complex data corresponding to the results of an
            integrated average result, usually of dimension (nr_sweep_points, 1).
        sweep_points: Sweep points of the acquisition.
        calibration_traces: A list with two entries of the complex data corresponding to
            the calibration traces, from the lowest transmon state to the highest.
        do_pca: whether to do principal component analysis on the data. If False, the
            data will be rotated along the line in the complex plane between the two
            calibration points and then projected onto it.

    Returns:
        dictionary with the following data:
            sweep_points,
            sweep_points extended with as many points as there are cal traces,
            the artificially added sweep_points for the cal traces,
            raw data,
            raw data with calibration traces appended,
            raw data of the calibration traces,
            rotated data,
            rotated data with the rotated calibration traces appended,
            rotated calibration traces data
    """
    if calibration_traces is None:
        calibration_traces = []
    num_cal_traces = len(calibration_traces)
    if num_cal_traces == 0:
        # Doing pca
        data_rot = principal_component_analysis(raw_data)
        data_raw_w_cal_tr = raw_data
    else:
        raw_data_cal_pt_0 = calibration_traces[0]
        raw_data_cal_pt_1 = calibration_traces[1]
        cal_traces = np.array([raw_data_cal_pt_0, raw_data_cal_pt_1])
        data_raw_w_cal_tr = np.concatenate([raw_data, cal_traces])
        if do_pca:
            data_rot = principal_component_analysis(data_raw_w_cal_tr)
        elif num_cal_traces == 2:  # noqa: PLR2004
            data_rot = rotate_data_to_cal_trace_results(
                data_raw_w_cal_tr,
                raw_data_cal_pt_0,
                raw_data_cal_pt_1,
            )
        else:
            raise NotImplementedError("Only 0 or 2 calibration states are supported")
    swpts_w_cal_tr = _extend_sweep_points_cal_traces(sweep_points, num_cal_traces)

    return {
        "sweep_points": np.array(sweep_points),
        "sweep_points_with_cal_traces": swpts_w_cal_tr,
        "sweep_points_cal_traces": swpts_w_cal_tr[
            len(swpts_w_cal_tr) - num_cal_traces :
        ],
        "data_raw": raw_data,
        "data_raw_with_cal_traces": data_raw_w_cal_tr,
        "data_raw_cal_traces": data_raw_w_cal_tr[
            len(data_raw_w_cal_tr) - num_cal_traces :
        ],
        "population": data_rot[: len(raw_data)],
        "population_with_cal_traces": data_rot,
        "population_cal_traces": data_rot[len(data_rot) - num_cal_traces :],
        "num_cal_traces": num_cal_traces,
    }


@workflow.task
def calculate_qubit_population(
    qubits: Qubits,
    result: RunExperimentResults,
    sweep_points: QubitSweepPoints,
    options: TuneupAnalysisOptions | None = None,
) -> dict[str, dict[str, ArrayLike]]:
    """Calculates the qubit population from the raw data.

     The data is processed in the following way:

     - If calibration traces were used in the experiment, the raw data is rotated based
     on the calibration traces.
     See [calibration_traces_rotation.py/rotate_data_to_cal_trace_results] for more
     details.
     - If no calibration traces were used in the experiment, or do_pca = True is passed
     in options, principal-component analysis is performed on the data.
     See [calibration_traces_rotation.py/principal_component_analysis] for more details.

    Arguments:
        qubits:
            The qubits on which the amplitude-Rabi experiments was run. May be either
            a single qubit or a list of qubits.
        result: the result of the experiment, returned by the run_experiment task.
        sweep_points:
            The sweep points used in the experiment for each qubit. If `qubits` is a
            single qubit, `sweep_points` must be an array. Otherwise, it must be a list
            of arrays.
        options:
            The options for building the workflow as an instance of
            [TuneupAnalysisOptions]. See the docstrings of this class for more
            details.

    Returns:
        dict with qubit UIDs as keys and the dictionary of processed data for each qubit
        as values. See [calibration_traces_rotation.py/calculate_population_1d] for what
        this dictionary looks like.

    Raises:
        TypeError:
            If result is not an instance of RunExperimentResults.
        ValueError:
            If the conditions in validate_and_convert_qubits_sweeps are not met.
    """
    opts = TuneupAnalysisOptions() if options is None else options
    validate_result(result)
    qubits, sweep_points = validate_and_convert_qubits_sweeps(qubits, sweep_points)
    processed_data_dict = {}
    for q, swpts in zip(qubits, sweep_points):
        raw_data = result[dsl.handles.result_handle(q.uid)].data
        if opts.use_cal_traces:
            calibration_traces = [
                result[dsl.handles.calibration_trace_handle(q.uid, cs)].data
                for cs in opts.cal_states
            ]
            do_pca = opts.do_pca
        else:
            calibration_traces = []
            do_pca = True
        data_dict = calculate_population_1d(
            raw_data,
            swpts,
            calibration_traces,
            do_pca=do_pca,
        )
        processed_data_dict[q.uid] = data_dict
    return processed_data_dict
