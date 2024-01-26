import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from laboneq_library.analysis.cal_trace_rotation import principal_component_analysis, rotate_data_to_cal_trace_results


@pytest.mark.parametrize("input, output", [
    (np.array([1+5j, 2+2j, 5+2j]), np.array([-2.54012 ,  0.080577,  2.459543])),
    (np.array([1, 2, 5]), np.array([-1.666667, -0.666667,  2.333333]))
])
def test_principal_component_analysis(input, output):
    # TODO: Expand tests
    assert_array_almost_equal(principal_component_analysis(input), output)


def test_rotate_data_to_cal_trace_results():
    # TODO: Expand tests
    raw_data = np.array([1+2j, 3+2j])
    pts_1 = np.array([0.5+2j])
    pts_2 = np.array([0.1+2j])
    assert_array_almost_equal(rotate_data_to_cal_trace_results(raw_data, pts_1, pts_2), [-1.25, -6.25])
