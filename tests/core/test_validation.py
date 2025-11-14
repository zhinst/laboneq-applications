# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.core.validation."""

import re

import numpy as np
import pytest
from laboneq.dsl.quantum import QPUTopology, QuantumElement
from laboneq.dsl.quantum.qpu_topology import TopologyEdge
from laboneq.dsl.quantum.quantum_element import QuantumParameters
from laboneq.dsl.quantum.transmon import Transmon
from laboneq.workflow.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)

from laboneq_applications.core.validation import (
    convert_qubits_sweeps_to_lists,
    validate_and_convert_qubits_sweeps,
    validate_and_convert_single_qubit_sweeps,
    validate_and_convert_sweeps_to_arrays,
    validate_and_extract_edges_from_qubit_pairs,
    validate_length_qubits_sweeps,
    validate_parallel_two_qubit_experiment,
    validate_result,
)
from laboneq_applications.qpu_types.tunable_transmon import TunableTransmonQubit


class TestValidateAndConvertQubitSweeps:
    def test_single_qubit(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        qubits, sweeps = validate_and_convert_qubits_sweeps(q0, [1, 2, 3])
        assert qubits == [q0]
        np.testing.assert_almost_equal(sweeps, [np.array([1, 2, 3])])

    def test_single_qubit_with_numpy_array(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        sweep_0 = np.array([1, 2, 3])
        qubits, sweeps = validate_and_convert_qubits_sweeps(q0, sweep_0)
        assert qubits == [q0]
        np.testing.assert_equal(sweeps, [[1, 2, 3]])

    def test_sequence_of_qubits(self, two_tunable_transmon_platform):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements
        qubits, sweeps = validate_and_convert_qubits_sweeps(
            [q0, q1],
            [[1, 2, 3], [4, 5, 6]],
        )
        assert qubits == [q0, q1]
        np.testing.assert_almost_equal(
            sweeps, [np.array([1, 2, 3]), np.array([4, 5, 6])]
        )

    def test_sequence_of_qubits_with_numpy_arrays(self, two_tunable_transmon_platform):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements
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
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements

        with pytest.raises(ValueError) as err:
            validate_and_convert_qubits_sweeps(q0, [[1, 2, 3]])

        assert str(err.value) == (
            "If a single qubit is passed, the sweep points must be an array or a list "
            "of numbers."
        )

    def test_number_of_qubits_and_sweeps_not_equal_error(
        self,
        single_tunable_transmon_platform,
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements

        with pytest.raises(ValueError) as err:
            validate_and_convert_qubits_sweeps([q0], [[1, 2], [3, 4]])

        assert str(err.value) == ("Length of qubits and sweep points must be the same.")

    def test_not_all_sweeps_valid(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements

        with pytest.raises(ValueError) as err:
            validate_and_convert_qubits_sweeps([q0], [[1, "a"]])

        assert str(err.value) == (
            "All elements of sweep points must be arrays or lists of numbers."
        )

    def test_sweeps_none(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        qubits = validate_and_convert_qubits_sweeps([q0], None)
        assert qubits == [q0]

        qubits = validate_and_convert_qubits_sweeps(q0, None)
        assert qubits == [q0]


class TestValidateLengthQubitsSweeps:
    def test_single_qubit(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        qubits, sweeps = validate_length_qubits_sweeps(q0, [1, 2, 3])
        assert qubits == q0
        np.testing.assert_almost_equal(sweeps, np.array([1, 2, 3]))

    def test_single_qubit_with_numpy_array(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        sweep_0 = np.array([1, 2, 3])
        qubits, sweeps = validate_length_qubits_sweeps(q0, sweep_0)
        assert qubits == q0
        np.testing.assert_equal(sweeps, [1, 2, 3])

    def test_sequence_of_qubits(self, two_tunable_transmon_platform):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements
        qubits, sweeps = validate_length_qubits_sweeps(
            [q0, q1],
            [[1, 2, 3], [4, 5, 6]],
        )
        assert qubits == [q0, q1]
        np.testing.assert_almost_equal(
            sweeps, [np.array([1, 2, 3]), np.array([4, 5, 6])]
        )

    def test_sequence_of_qubits_with_numpy_arrays(self, two_tunable_transmon_platform):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements
        sweep_0 = np.array([1, 2, 3])
        sweep_1 = np.array([4, 5, 6])
        qubits, sweeps = validate_length_qubits_sweeps(
            [q0, q1],
            [sweep_0, sweep_1],
        )
        assert qubits == [q0, q1]
        np.testing.assert_equal(sweeps, [[1, 2, 3], [4, 5, 6]])

    def test_not_sequence_of_quantum_elements_error(self):
        with pytest.raises(ValueError) as err:
            validate_length_qubits_sweeps([1], [[1, 2, 3]])

        assert str(err.value) == (
            "Qubits must be a QuantumElement or a sequence of QuantumElements."
        )

    def test_single_qubit_with_sequence_of_sweeps_error(
        self,
        single_tunable_transmon_platform,
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements

        with pytest.raises(ValueError) as err:
            validate_length_qubits_sweeps(q0, [[1, 2, 3]])

        assert str(err.value) == (
            "If a single qubit is passed, the sweep points must be an array or a list "
            "of numbers."
        )

    def test_number_of_qubits_and_sweeps_not_equal_error(
        self,
        single_tunable_transmon_platform,
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements

        with pytest.raises(ValueError) as err:
            validate_length_qubits_sweeps([q0], [[1, 2], [3, 4]])

        assert str(err.value) == "Length of qubits and sweep points must be the same."

    def test_not_all_sweeps_valid(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements

        with pytest.raises(ValueError) as err:
            validate_length_qubits_sweeps([q0], [[1, "a"]])

        assert str(err.value) == (
            "All elements of sweep points must be arrays or lists of numbers."
        )

    def test_sweeps_none(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        qubits = validate_length_qubits_sweeps([q0], None)
        assert qubits == [q0]

        qubits = validate_length_qubits_sweeps(q0, None)
        assert qubits == q0


class TestConvertQubitsSweepsToLists:
    def test_single_qubit(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        qubits, sweeps = convert_qubits_sweeps_to_lists(q0, [1, 2, 3])
        assert qubits == [q0]
        np.testing.assert_almost_equal(sweeps, [np.array([1, 2, 3])])

    def test_single_qubit_with_numpy_array(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        sweep_0 = np.array([1, 2, 3])
        qubits, sweeps = convert_qubits_sweeps_to_lists(q0, sweep_0)
        assert qubits == [q0]
        np.testing.assert_equal(sweeps, [[1, 2, 3]])

    def test_sequence_of_qubits(self, two_tunable_transmon_platform):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements
        qubits, sweeps = convert_qubits_sweeps_to_lists(
            [q0, q1],
            [[1, 2, 3], [4, 5, 6]],
        )
        assert qubits == [q0, q1]
        np.testing.assert_almost_equal(
            sweeps, [np.array([1, 2, 3]), np.array([4, 5, 6])]
        )

    def test_sequence_of_qubits_with_numpy_arrays(self, two_tunable_transmon_platform):
        [q0, q1] = two_tunable_transmon_platform.qpu.quantum_elements
        sweep_0 = np.array([1, 2, 3])
        sweep_1 = np.array([4, 5, 6])
        qubits, sweeps = convert_qubits_sweeps_to_lists(
            [q0, q1],
            [sweep_0, sweep_1],
        )
        assert qubits == [q0, q1]
        np.testing.assert_equal(sweeps, [[1, 2, 3], [4, 5, 6]])

    def test_not_sequence_of_quantum_elements_error(self):
        with pytest.raises(ValueError) as err:
            convert_qubits_sweeps_to_lists([1], [[1, 2, 3]])

        assert str(err.value) == (
            "Qubits must be a QuantumElement or a sequence of QuantumElements."
        )

    def test_single_qubit_with_sequence_of_sweeps(
        self,
        single_tunable_transmon_platform,
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        qubits, sweeps = convert_qubits_sweeps_to_lists(q0, [[1, 2, 3]])
        assert qubits == [q0]
        np.testing.assert_equal(sweeps, [np.array([[1, 2, 3]])])

    def test_number_of_qubits_and_sweeps_not_equal(
        self,
        single_tunable_transmon_platform,
    ):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements

        qubits, sweeps = convert_qubits_sweeps_to_lists([q0], [[1, 2], [3, 4]])
        assert qubits == [q0]
        np.testing.assert_equal(sweeps, [[1, 2], [3, 4]])

    def test_not_all_sweeps_valid(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements

        with pytest.raises(ValueError) as err:
            convert_qubits_sweeps_to_lists([q0], [[1, "a"]])

        assert str(err.value) == (
            "All elements of sweep points must be arrays or lists of numbers."
        )

    def test_sweeps_none(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        qubits = convert_qubits_sweeps_to_lists([q0], None)
        assert qubits == [q0]

        qubits = convert_qubits_sweeps_to_lists(q0, None)
        assert qubits == [q0]


class TestValidateConvertSweepsToArrays:
    def test_list_of_numbers(self):
        sweeps = validate_and_convert_sweeps_to_arrays([1, 2, 3])
        np.testing.assert_almost_equal(sweeps, np.array([1, 2, 3]))

    def test_list_of_lists_of_numbers(self):
        sweeps = validate_and_convert_sweeps_to_arrays([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_almost_equal(
            sweeps, [np.array([1, 2, 3]), np.array([4, 5, 6])]
        )

    def test_array_of_numbers(self):
        sweeps = validate_and_convert_sweeps_to_arrays(np.array([1, 2, 3]))
        np.testing.assert_almost_equal(sweeps, np.array([1, 2, 3]))

    def test_list_of_arrays_of_numbers(self):
        sweeps = validate_and_convert_sweeps_to_arrays(
            [np.array([1, 2, 3]), np.array([4, 5, 6])]
        )
        np.testing.assert_almost_equal(
            sweeps, [np.array([1, 2, 3]), np.array([4, 5, 6])]
        )

    def test_array_of_arrays_of_numbers(self):
        sweeps = validate_and_convert_sweeps_to_arrays(
            np.array([np.array([1, 2, 3]), np.array([4, 5, 6])])
        )
        np.testing.assert_almost_equal(
            sweeps, np.array([np.array([1, 2, 3]), np.array([4, 5, 6])])
        )

    def test_invalid_sweeps_type(self):
        with pytest.raises(TypeError) as err:
            validate_and_convert_sweeps_to_arrays(1)
        assert str(err.value) == "The sweep points must be an array or a list."

    def test_invalid_sweeps_inner_type(self):
        with pytest.raises(ValueError) as err:
            validate_and_convert_sweeps_to_arrays([1, "a"])
        assert str(err.value) == (
            "The sweep points must be an array or a list of numbers."
        )

    def test_invalid_sweeps_inner_type_iterables(self):
        with pytest.raises(ValueError) as err:
            validate_and_convert_sweeps_to_arrays([[1, "a"], [1, 2]])
        assert str(err.value) == (
            "All elements of sweep points must be arrays or lists of numbers."
        )


class TestValidateAndConvertSingleQubitSweeps:
    def test_list_of_numbers(self):
        sweeps = validate_and_convert_sweeps_to_arrays([1, 2, 3])
        np.testing.assert_almost_equal(sweeps, np.array([1, 2, 3]))

    def test_sweeps_none(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        qubit = validate_and_convert_single_qubit_sweeps(q0, None)
        assert qubit == q0

    def test_sweep_list(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        qubit, sweeps = validate_and_convert_single_qubit_sweeps(q0, [1, 2, 3])
        assert qubit == q0
        np.testing.assert_almost_equal(sweeps, np.array([1, 2, 3]))

    def test_sweep_array(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        qubit, sweeps = validate_and_convert_single_qubit_sweeps(
            q0, np.array([1, 2, 3])
        )
        assert qubit == q0
        np.testing.assert_almost_equal(sweeps, np.array([1, 2, 3]))

    def test_invalid_qubits(self, two_tunable_transmon_platform):
        qubits = two_tunable_transmon_platform.qpu.quantum_elements
        with pytest.raises(TypeError) as err:
            validate_and_convert_single_qubit_sweeps(qubits, [1, 2, 3])
        assert str(err.value) == "Only a single qubit is supported."

    def test_invalid_qubit_sweeps(self, single_tunable_transmon_platform):
        [q0] = single_tunable_transmon_platform.qpu.quantum_elements
        with pytest.raises(ValueError) as err:
            validate_and_convert_single_qubit_sweeps(q0, [[1, 2, 3], [1, 2, 3]])
        assert str(err.value) == (
            "If a single qubit is passed, the sweep points must be an array or a list "
            "of numbers."
        )


@pytest.fixture
def result():
    """Results from AmplitudeRabi experiment."""
    data = {
        "result": {
            "q0": AcquiredResult(
                data=np.array(
                    [
                        0.05290302 - 0.13215136j,
                        0.06067577 - 0.12907117j,
                        0.05849071 - 0.09401458j,
                        0.0683788 - 0.04265771j,
                        0.07369121 + 0.0238058j,
                        0.08271086 + 0.10077513j,
                        0.09092848 + 0.1884216j,
                        0.1063583 + 0.28337206j,
                        0.11472132 + 0.38879551j,
                        0.13147716 + 0.49203866j,
                        0.13378882 + 0.59027211j,
                        0.15108762 + 0.70302525j,
                        0.16102455 + 0.77474721j,
                        0.16483135 + 0.83853894j,
                        0.17209631 + 0.88743935j,
                        0.17435144 + 0.90659384j,
                        0.17877636 + 0.92026812j,
                        0.17153804 + 0.90921755j,
                        0.17243493 + 0.88099388j,
                        0.164842 + 0.82561295j,
                        0.15646681 + 0.76574749j,
                    ]
                ),
                axis=[
                    np.array(
                        [
                            0.0,
                            0.0238155,
                            0.04763101,
                            0.07144651,
                            0.09526201,
                            0.11907752,
                            0.14289302,
                            0.16670852,
                            0.19052403,
                            0.21433953,
                            0.23815503,
                            0.26197054,
                            0.28578604,
                            0.30960154,
                            0.33341705,
                            0.35723255,
                            0.38104805,
                            0.40486356,
                            0.42867906,
                            0.45249456,
                            0.47631007,
                        ]
                    )
                ],
                axis_name=[],
            )
        },
    }
    return RunExperimentResults(data=data), data


class TestValidateResult:
    def test_valid_result(self, result):
        validate_result(result[0])

    def test_invalid_result(self, result):
        with pytest.raises(TypeError) as err:
            validate_result(result[1])

        assert str(err.value) == (
            "The result must be either an instance of RunExperimentResults "
            "or a sequence of RunExperimentResults."
        )


@pytest.fixture
def two_tunable_transmon_platform_with_topology(two_tunable_transmon_platform):
    platform = two_tunable_transmon_platform
    qpu = platform.qpu
    qubits = qpu.quantum_elements

    topo = QPUTopology(qubits)
    topo.add_edge("coupler_1", "q0", "q1", quantum_element=qubits[0])
    topo.add_edge("coupler_2", "q0", "q1", quantum_element=qubits[1])
    topo.add_edge("coupler_3", "q0", "q1", quantum_element=qubits[0])
    topo.add_edge("coupler_1", "q1", "q0", quantum_element=qubits[1])
    topo.add_edge("coupler_2", "q1", "q0", parameters=QuantumParameters())
    topo.add_edge("coupler_3", "q1", "q0")
    qpu.topology = topo

    return platform


class TestValidateAndExtractEdgesFromQubitPairs:
    def test_valid_element_class(self, two_tunable_transmon_platform_with_topology):
        qpu = two_tunable_transmon_platform_with_topology.qpu

        edge_list = validate_and_extract_edges_from_qubit_pairs(
            qpu,
            "coupler_1",
            [["q0", "q1"], ["q1", "q0"]],
            element_class=TunableTransmonQubit,
        )

        assert isinstance(edge_list, list)
        assert len(edge_list) == 2
        assert isinstance(edge_list[0], TopologyEdge)
        assert isinstance(edge_list[1], TopologyEdge)
        assert isinstance(edge_list[0].quantum_element, TunableTransmonQubit)
        assert isinstance(edge_list[1].quantum_element, TunableTransmonQubit)
        assert (
            edge_list[0].tag,
            edge_list[0].source_node.uid,
            edge_list[0].target_node.uid,
        ) == ("coupler_1", "q0", "q1")
        assert (
            edge_list[1].tag,
            edge_list[1].source_node.uid,
            edge_list[1].target_node.uid,
        ) == ("coupler_1", "q1", "q0")

    def test_valid_element_subclass(self, two_tunable_transmon_platform_with_topology):
        qpu = two_tunable_transmon_platform_with_topology.qpu

        edge_list = validate_and_extract_edges_from_qubit_pairs(
            qpu, "coupler_2", [["q0", "q1"]], element_class=QuantumElement
        )
        assert isinstance(edge_list, list)
        assert len(edge_list) == 1
        assert isinstance(edge_list[0], TopologyEdge)
        assert isinstance(edge_list[0].quantum_element, QuantumElement)
        assert isinstance(edge_list[0].quantum_element, TunableTransmonQubit)
        assert (
            edge_list[0].tag,
            edge_list[0].source_node.uid,
            edge_list[0].target_node.uid,
        ) == ("coupler_2", "q0", "q1")

    def test_invalid_element_class(self, two_tunable_transmon_platform_with_topology):
        qpu = two_tunable_transmon_platform_with_topology.qpu

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Quantum element on edge (coupler_1, q0, q1) has invalid "
                "type: <class 'laboneq_applications.qpu_types."
                "tunable_transmon.qubit_types.TunableTransmonQubit'>. "
                "Expected type: <class 'laboneq.dsl.quantum.transmon."
                "Transmon'>."
            ),
        ):
            validate_and_extract_edges_from_qubit_pairs(
                qpu, "coupler_1", [["q0", "q1"], ["q1", "q0"]], element_class=Transmon
            )


class ValidateParallelTwoQubitExperiment:
    def test_valid_qubit_pairs(self, two_tunable_transmon_platform_with_topology):
        qpu = two_tunable_transmon_platform_with_topology.qpu
        qubits = two_tunable_transmon_platform_with_topology.qpu.qubits

        q_list = validate_parallel_two_qubit_experiment(qpu, [["q0", "q1"]])

        assert isinstance(q_list, list)
        assert len(q_list) == 2
        assert isinstance(q_list[0], QuantumElement)
        assert isinstance(q_list[1], QuantumElement)
        assert q_list[0].uid == "q0"
        assert q_list[0] is qubits[0]
        assert q_list[1].uid == "q1"
        assert q_list[1] is qubits[1]

    def test_invalid_qubit_pairs(self, two_tunable_transmon_platform_with_topology):
        qpu = two_tunable_transmon_platform_with_topology.qpu

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Quantum elements appear more than once in the edges. "
                "Calibration cannot be parallelized."
            ),
        ):
            validate_parallel_two_qubit_experiment(qpu, [["q0", "q1"], ["q0", "q1"]])
