import matplotlib.pyplot as plt
import numpy as np


def principal_component_analysis(raw_data=None, results=None, handle=None):
    """
    Rotates and projects data using principal component analysis (PCA).

    Args:
        raw_data: array of complex data representing the results of an
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