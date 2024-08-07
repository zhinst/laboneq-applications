"""Tests for laboneq_applications.core.validation."""

import numpy as np
import pytest

from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps


class TestValidateAndConvertQubitSweeps:
    def test_single_qubit(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        qubits, sweeps = validate_and_convert_qubits_sweeps(q0, [1, 2, 3])
        assert qubits == [q0]
        assert sweeps == [[1, 2, 3]]

    def test_single_qubit_with_numpy_array(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        sweep_0 = np.array([1, 2, 3])
        qubits, sweeps = validate_and_convert_qubits_sweeps(q0, sweep_0)
        assert qubits == [q0]
        np.testing.assert_equal(sweeps, [[1, 2, 3]])

    def test_sequence_of_qubits(self, two_tunable_transmon_platform):
        [q0, q1] = two_tunable_transmon_platform.qpu.qubits
        qubits, sweeps = validate_and_convert_qubits_sweeps(
            [q0, q1],
            [[1, 2, 3], [4, 5, 6]],
        )
        assert qubits == [q0, q1]
        assert sweeps == [[1, 2, 3], [4, 5, 6]]

    def test_sequence_of_qubits_with_numpy_arrays(self, two_tunable_transmon_platform):
        [q0, q1] = two_tunable_transmon_platform.qpu.qubits
        sweep_0 = np.array([1, 2, 3])
        sweep_1 = np.array([4, 5, 6])
        qubits, sweeps = validate_and_convert_qubits_sweeps(
            [q0, q1],
            [sweep_0, sweep_1],
        )
        assert qubits == [q0, q1]
        np.testing.assert_equal(sweeps, [[1, 2, 3], [4, 5, 6]])

    def test_not_sequence_of_quantum_elements_error(self):
        with pytest.raises(ValueError) as err:
            validate_and_convert_qubits_sweeps([1], [[1, 2, 3]])

        assert str(err.value) == (
            "Qubits must be a QuantumElement or a sequence of QuantumElements."
        )

    def test_single_qubit_with_sequence_of_sweeps_error(
        self,
        single_tunable_transmon_platform,
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits

        with pytest.raises(ValueError) as err:
            validate_and_convert_qubits_sweeps(q0, [[1, 2, 3]])

        assert str(err.value) == (
            "If a single qubit is passed, the sweep points must be a list of numbers."
        )

    def test_number_of_qubits_and_sweeps_not_equal_error(
        self,
        single_tunable_transmon_platform,
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits

        with pytest.raises(ValueError) as err:
            validate_and_convert_qubits_sweeps([q0], [[1, 2], [3, 4]])

        assert str(err.value) == ("Length of qubits and sweep points must be the same.")

    def test_not_all_sweeps_valid(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.qubits

        with pytest.raises(ValueError) as err:
            validate_and_convert_qubits_sweeps([q0], [[1, "a"]])

        assert str(err.value) == (
            "All elements of sweep points must be lists of numbers."
        )
