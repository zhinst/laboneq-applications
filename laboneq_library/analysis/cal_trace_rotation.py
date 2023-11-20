import matplotlib.pyplot as plt
import numpy as np


def principal_component_analysis(raw_data=None, results=None, handle=None):
    """
    Rotates and projects data using principal component analysis (PCA).

    Args:
        raw_data: array of complex data corresponding to the results of an
            integrated average result
        results: instance of a Results class
        handle: handle inside the results instance pointing to the data on
            which to do PCA

    Returns:
        data array after PCA
    """

    if raw_data is None:
        if results is None or handle is None:
            raise ValueError('Please provide either the raw_data array, or '
                             'a Results instance and the data handle.')
        raw_data = results.get_data(handle)
    real, imag = np.real(raw_data), np.imag(raw_data)

    # translate each column in the data by its mean
    mean_real, mean_imag = np.mean(real), np.mean(imag)
    trans_real, trans_imag = real - mean_real, imag-mean_imag
    row_data_trans = np.array([trans_real, trans_imag])

    # compute the covariance 2x2 matrix
    cov_matrix = np.cov(row_data_trans)

    # find eigenvalues and eigenvectors of the covariance matrix
    [eigvals, eigvecs] = np.linalg.eig(cov_matrix)

    # compute the transposed feature vector
    row_feature_vector = np.array([(eigvecs[0, np.argmin(eigvals)],
                                    eigvecs[1, np.argmin(eigvals)]),
                                   (eigvecs[0, np.argmax(eigvals)],
                                    eigvecs[1, np.argmax(eigvals)])])

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
        np.array([[np.cos(angle), -1*np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]]))
    return rotation_matrix


def rotate_data_to_cal_trace_results(raw_data=None, raw_data_cal_pt_0=None,
                                     raw_data_cal_pt_1=None,
                                     results=None, handle_data=None,
                                     handle_cal_pt_0=None, handle_cal_pt_1=None):
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
        results: instance of a Results class
        handle: handle inside the results instance pointing to the data on
            which to do PCA

    Returns:
        rotated, projected, and normalised data array
    """

    def check_for_results_handle():
        if results is None or handle_data is None:
            raise ValueError('Please provide either the raw_data array, '
                             'raw_data_cal_pt_0 and raw_data_cal_pt_0, or '
                             'a Results instance and the handles for the raw '
                             'data and for the two calibration traces.')

    if raw_data is None:
        check_for_results_handle()
        raw_data = results.get_data(handle_data)

    if raw_data_cal_pt_0 is None:
        check_for_results_handle()
        raw_data_cal_pt_0 = results.get_data(handle_cal_pt_0)

    if handle_cal_pt_1 is None:
        check_for_results_handle()
        handle_cal_pt_1 = results.get_data(handle_cal_pt_1)

    data_iq = np.array([np.real(raw_data), np.imag(raw_data)])
    data_cal0 = np.array([np.mean(np.real(raw_data_cal_pt_0)),
                          np.mean(np.imag(raw_data_cal_pt_0))])
    data_cal1 = np.array([np.mean(np.real(raw_data_cal_pt_1)),
                          np.mean(np.imag(raw_data_cal_pt_1))])

    # Translate the data
    trans_data = data_iq - data_cal0

    # Rotate the data
    diff_data_cal = data_cal1 - data_cal0
    M = calculate_rotation_matrix(diff_data_cal[0], diff_data_cal[1])
    rotated_data = M @ trans_data

    # Normalize the data
    one_zero_dist = np.sqrt(diff_data_cal[0] ** 2 + diff_data_cal[1] ** 2)
    normalised_data = rotated_data / one_zero_dist

    return normalised_data
