from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Sequence


if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def principal_component_analysis(raw_data: ArrayLike) -> ArrayLike:
    """
    Rotates and projects 1D data using principal component analysis (PCA).

    Args:
        raw_data: Array of complex data corresponding to the results of an
            integrated average result

    Returns:
        data array after PCA
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
        ]
    )

    # compute final, projected data; only the first row is of interest (it is the
    # principal axis
    final_data = np.dot(row_feature_vector, row_data_trans)
    data_pca = final_data[1, :]

    return data_pca


def calculate_rotation_matrix(delta_I, delta_Q):
    """
    Calculates a matrix that rotates the data to lie along the Q-axis.
    Input can be either the I and Q coordinates of the zero cal_point or
    the difference between the 1 and 0 cal points.

    Args:
        delta_I: difference between the real parts of the first and second
            cal points
        delta_Q: difference between the imaginary parts of the first and second
            cal points

    Returns:
        the rotation matrix as a numpy array
    """
    angle = np.arctan2(delta_Q, delta_I)
    rotation_matrix = np.transpose(
        np.array([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    )
    return rotation_matrix


def rotate_data_to_cal_trace_results(
    raw_data: ArrayLike,
    raw_data_cal_pt_0: ArrayLike,
    raw_data_cal_pt_1: ArrayLike,
) -> ArrayLike:
    """
    Rotates and projects the raw data onto the line in the IQ plane between
    two calibration points, then normalises the rotated and projected data to
    the distance in the IQ plane between the two calibration points.
    The calibration points are the results of preparing two transmon states.

    If more than one calibration point per state
    (i.e. len(raw_data_cal_pt_0) > 1), the mean of the cal points will be taken.

    Args:
        raw_data: array of complex data corresponding to the results of an
            integrated average result
        raw_data_cal_pt_0: array of complex data corresponding to the first
            calibration trace. First here corresponds to the lowest transmon
            state. For example, g if the cal traces measured were g and e or
            g and f; e if the cal traces measured were e an f
        raw_data_cal_pt_1: array of complex data corresponding to the second
            calibration trace. Second here corresponds to the highest transmon
            state. For example, e if the cal traces measured were g and e;
            f if the cal traces measured were g and f or e an f

    Returns:
        rotated, projected, and normalised data array
    """
    data_iq = np.array([np.real(raw_data), np.imag(raw_data)])
    data_cal0 = np.array(
        [np.mean(np.real(raw_data_cal_pt_0)), np.mean(np.imag(raw_data_cal_pt_0))]
    )
    data_cal1 = np.array(
        [np.mean(np.real(raw_data_cal_pt_1)), np.mean(np.imag(raw_data_cal_pt_1))]
    )

    # Translate the data
    trans_data = data_iq - np.repeat(data_cal0[:, np.newaxis], data_iq.shape[1], axis=1)

    # Rotate the data
    diff_data_cal = data_cal1 - data_cal0
    M = calculate_rotation_matrix(diff_data_cal[0], diff_data_cal[1])
    rotated_data = M @ trans_data

    # Normalize the data
    one_zero_dist = np.sqrt(diff_data_cal[0] ** 2 + diff_data_cal[1] ** 2)
    normalised_data = rotated_data / one_zero_dist

    return normalised_data[0]


def _extend_sweep_points_cal_traces(sweep_points: ArrayLike, num_cal_traces=0) -> np.ndarray:
    if num_cal_traces == 0:
        return sweep_points

    if len(sweep_points) > 1:
        dsp = sweep_points[1] - sweep_points[0]
    else:
        dsp = 1
    cal_traces_swpts = np.array(
        [sweep_points[-1] + (i + 1) * dsp for i in range(num_cal_traces)]
    )
    return np.concatenate([sweep_points, cal_traces_swpts])


def rotate_data_1d(
    raw_data: ArrayLike,
    sweep_points: ArrayLike,
    calibration_traces: list[ArrayLike | complex] = None,
    do_pca: bool = False
) -> dict:
    """Rotate 1D data.

    Arguments:
        raw_data: Measurement raw data.
        sweep_points: Sweep points of the acquisition.
        calibration_traces: A list of calibration traces from the lowest state to the highest.
        do_pca: Do principal component analysis on the data.
    """
    num_cal_traces = len(calibration_traces)
    if num_cal_traces not in (0, 2):
        raise NotImplementedError("Only 0 or 2 calibration states are supported")
    if num_cal_traces == 2:
        raw_data_cal_pt_0 = calibration_traces[0]
        raw_data_cal_pt_1 = calibration_traces[1]
        cal_traces = np.array([raw_data_cal_pt_0, raw_data_cal_pt_1])
        data_raw_w_cal_tr = np.concatenate([raw_data, cal_traces])
        if do_pca:
            data_rot = principal_component_analysis(data_raw_w_cal_tr)
        else:
            data_rot = rotate_data_to_cal_trace_results(
                data_raw_w_cal_tr, raw_data_cal_pt_0, raw_data_cal_pt_1
            )
    else:
        data_rot = principal_component_analysis(raw_data)
        data_raw_w_cal_tr = raw_data
    swpts_w_cal_tr = _extend_sweep_points_cal_traces(sweep_points, num_cal_traces)

    data_dict = {
        "sweep_points": sweep_points,
        "sweep_points_w_cal_traces": swpts_w_cal_tr,
        "sweep_points_cal_traces": swpts_w_cal_tr[
            len(swpts_w_cal_tr) - num_cal_traces :
        ],
        "data_raw": raw_data,
        "data_raw_w_cal_traces": data_raw_w_cal_tr,
        "data_raw_cal_traces": data_raw_w_cal_tr[
            len(data_raw_w_cal_tr) - num_cal_traces :
        ],
        "data_rotated": data_rot[: len(raw_data)],
        "data_rotated_w_cal_traces": data_rot,
        "data_rotated_cal_traces": data_rot[len(data_rot) - num_cal_traces :],
        "num_cal_traces": num_cal_traces,
    }
    return data_dict


def rotate_data_2d(
    raw_data: ArrayLike,
    sweep_points: Sequence[ArrayLike],
    calibration_traces: list[ArrayLike | complex] = None,
    do_pca: bool = False
) -> dict:
    """Rotate 2D data.

    Arguments:
        raw_data: Measurement raw data.
        sweep_points: Sweep points of the acquisition.
        calibration_traces: A list of calibration traces from the lowest state to the highest.
        do_pca: Do principal component analysis on the data.
    """
    num_cal_traces = len(calibration_traces)
    if num_cal_traces not in (0, 2):
        raise NotImplementedError("Only 0 or 2 calibration states are supported")
    swpts_nt = sweep_points[0]
    swpts_rt = sweep_points[1]
    data_rot = np.zeros(shape=raw_data.shape)
    if num_cal_traces == 2:
        raw_data_cal_pt_0 = calibration_traces[0]
        raw_data_cal_pt_1 = calibration_traces[1]
        cal_traces = np.array([raw_data_cal_pt_0, raw_data_cal_pt_1]).T
        data_raw_w_cal_tr = np.concatenate([raw_data, cal_traces], axis=1)
        data_rot = np.zeros(shape=data_raw_w_cal_tr.shape)
        if do_pca:
            for i in range(data_raw_w_cal_tr.shape[0]):
                data_rot[i, :] = principal_component_analysis(
                    data_raw_w_cal_tr[i, :]
                )
        else:
            for i in range(data_raw_w_cal_tr.shape[0]):
                data_rot[i, :] = rotate_data_to_cal_trace_results(
                    data_raw_w_cal_tr[i, :], raw_data_cal_pt_0[i], raw_data_cal_pt_1[i]
                )
    else:
        for i in range(raw_data.shape[0]):
            data_rot[i, :] = principal_component_analysis(raw_data[i, :])
        data_raw_w_cal_tr = raw_data

    swpts_rt_w_cal_tr = _extend_sweep_points_cal_traces(swpts_rt, num_cal_traces)
    data_dict = {
        "sweep_points": swpts_rt,
        "sweep_points_nt": swpts_nt,
        "sweep_points_w_cal_traces": swpts_rt_w_cal_tr,
        "sweep_points_cal_traces": swpts_rt_w_cal_tr[
            len(swpts_rt_w_cal_tr) - num_cal_traces :
        ],
        "data_raw": raw_data,
        "data_raw_w_cal_traces": data_raw_w_cal_tr,
        "data_raw_cal_traces": data_raw_w_cal_tr[
            :, data_raw_w_cal_tr.shape[1] - num_cal_traces :
        ],
        "data_rotated": data_rot[:, : raw_data.shape[1]],
        "data_rotated_w_cal_traces": data_rot,
        "data_rotated_cal_traces": data_rot[:, data_rot.shape[1] - num_cal_traces :],
        "num_cal_traces": num_cal_traces,
    }

    return data_dict
