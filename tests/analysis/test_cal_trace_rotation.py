import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from laboneq_library.analysis.cal_trace_rotation import principal_component_analysis


@pytest.mark.parametrize("input, output", [
    (np.array([1+5j, 2+2j, 5+2j]), np.array([-2.54012 ,  0.080577,  2.459543])),
    (np.array([1, 2, 5]), np.array([-1.666667, -0.666667,  2.333333]))
])
def test_principal_component_analysis(input, output):
    assert_array_almost_equal(principal_component_analysis(input), output)
