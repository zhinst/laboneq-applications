# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications._automation.workflow.workflow_automation"""

from __future__ import annotations

import copy
import inspect

import networkx as nx
import numpy as np
import pytest
from laboneq._automation import AutomationElementStatus as Status
from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.quantum import (
    QPU,
    QuantumPlatform,
)
from laboneq.dsl.session import Session
from laboneq.workflow import WorkflowBuilder
from laboneq.workflow.result import WorkflowResult

from laboneq_applications._automation.workflow.workflow_automation import (
    WorkflowAutomation,
)
from laboneq_applications._automation.workflow.workflow_layer import WorkflowLayer
from laboneq_applications._automation.workflow.workflow_logic import (
    FixedParameterUpdate,
)
from laboneq_applications.experiments import (
    amplitude_fine,
    qubit_spectroscopy,
    ramsey,
)
from laboneq_applications.qpu_types.tunable_transmon import demo_platform


@pytest.fixture
def qt_platform() -> QuantumPlatform:
    return demo_platform(n_qubits=4)


@pytest.fixture
def qpu(qt_platform) -> QPU:
    return qt_platform.qpu


@pytest.fixture
def device_setup(qt_platform) -> DeviceSetup:
    return qt_platform.setup


@pytest.fixture
def session(device_setup) -> Session:
    s = Session(device_setup)
    s.connect(do_emulation=True)
    return s


@pytest.fixture
def automation_parameters() -> dict:
    return {
        "qs1": {
            "q0": {"frequencies": np.linspace(6.0e9, 6.50e9, 101)},
            "q1": {"frequencies": np.linspace(6.0e9, 6.50e9, 101)},
            "q2": {"frequencies": np.linspace(6.0e9, 6.50e9, 101)},
            "q3": {"frequencies": np.linspace(6.0e9, 6.50e9, 101)},
            "options": {
                "evaluate": True,
                "update": True,
                "count": 2048,
                "active_reset": True,
            },
        },
        "qs2": {
            "q0": {"frequencies": np.linspace(6.1e9, 6.5e9, 101)},
            "q1": {"frequencies": np.linspace(6.1e9, 6.5e9, 101)},
            "q2": {"frequencies": np.linspace(6.1e9, 6.5e9, 101)},
            "q3": {"frequencies": np.linspace(6.1e9, 6.5e9, 101)},
            "options": {
                "evaluate": True,
                "update": False,
            },
        },
        "r1": {
            "q0": {"delays": np.linspace(0.0e00, 2.0e-05, 50), "detunings": 670000.0},
            "q1": {"delays": np.linspace(2e-05, 5e-05, 50), "detunings": 670000.0},
            "options": {
                "evaluate": True,
                "update": True,
                "active_reset": True,
            },
        },
        "qs3": {
            "q0": {"frequencies": np.linspace(6.1e9, 6.5e9, 55)},
            "q1": {"frequencies": np.linspace(6.1e9, 6.5e9, 55)},
            "q2": {"frequencies": np.linspace(6.1e9, 6.5e9, 55)},
            "q3": {"frequencies": np.linspace(6.1e9, 6.5e9, 55)},
            "options": {
                "evaluate": True,
                "update": True,
                "active_reset": True,
            },
        },
        "qs4": {
            "q0": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
            "q2": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
            "options": {
                "evaluate": True,
                "update": True,
                "count": 2048,
                "active_reset": True,
            },
        },
        "qs5": {
            "q0": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
            "q1": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
            "q2": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
            "q3": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
            "options": {
                "evaluate": True,
                "update": True,
                "count": 2048,
                "active_reset": True,
            },
        },
        "qs6": {
            "q0": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
        },
        "r2": {
            "q0": {"delays": np.linspace(0, 2.0e-05, 33), "detunings": 670000.0},
            "q3": {"delays": np.linspace(0, 9.3e-05, 33), "detunings": 670000.0},
            "options": {
                "evaluate": False,
                "update": False,
            },
        },
        "af1": {
            "q0": {},
            "q1": {},
            "q2": {},
            "q3": {},
            "repetitions": [
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2],
            ],
        },
        "ra1": {
            "q0": {"amplitudes": np.linspace(0, 1, 11)},
            "q1": {"amplitudes": np.linspace(0, 1, 11)},
            "options": {"evaluate": False, "update": False, "active_reset": True},
        },
    }


@pytest.fixture
def auto(session, qpu, automation_parameters) -> WorkflowAutomation:
    return WorkflowAutomation(
        session, qpu=qpu, automation_parameters=automation_parameters
    )


@pytest.fixture
def workflow_parameters() -> dict:
    return {
        "q0": {
            "frequencies": np.linspace(6e9, 6.2e9, 101),
            "evaluation_fit_r2_thresholds": 1.0,
        },
        "q1": {"frequencies": np.linspace(6e9, 6.2e9, 101)},
        "q2": {"frequencies": np.linspace(6e9, 6.2e9, 101)},
        "q3": {"frequencies": np.linspace(6e9, 6.2e9, 101)},
    }


@pytest.fixture
def qubit_spectroscopy_workflow() -> WorkflowBuilder:
    return qubit_spectroscopy.experiment_workflow


@pytest.fixture
def ramsey_workflow() -> WorkflowBuilder:
    return ramsey.experiment_workflow


@pytest.fixture
def amplitude_fine_workflow() -> WorkflowBuilder:
    return amplitude_fine.experiment_workflow_x180


class TestWorkflowAutomation:
    def test_create(self, session, qpu, automation_parameters):
        auto = WorkflowAutomation(
            session, qpu, automation_parameters=automation_parameters
        )
        assert auto.automation_parameters == automation_parameters
        assert auto.session == session
        assert auto.qpu == qpu
        assert isinstance(auto._node_graph, nx.DiGraph)
        assert list(auto._node_graph.nodes) == ["__root__"]
        assert auto._node_lookup == {"__root__": None}
        assert auto._layer_lookup == {"__root__": None}

        # WorkflowAutomation methods
        assert hasattr(auto, "run_node")
        method1 = auto.run_node
        assert callable(method1)
        assert len(inspect.signature(method1).parameters) == 2
        assert hasattr(auto, "run_layer")
        method2 = auto.run_layer
        assert callable(method2)
        assert len(inspect.signature(method2).parameters) == 8

    def test_run(
        self, auto, qubit_spectroscopy_workflow, ramsey_workflow, workflow_parameters
    ):
        qs1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on=["__root__"],
        )
        qs2 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs2",
            depends_on=["qs1"],
            workflow_parameters=workflow_parameters,
        )
        r1 = WorkflowLayer(
            ramsey_workflow,
            ["q0", "q1"],
            key="r1",
            depends_on=["qs2"],
        )
        r2 = WorkflowLayer(
            ramsey_workflow,
            ["q0", "q3"],
            key="r2",
            depends_on=["qs2"],
        )
        qs3 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0"],
            key="qs3",
            depends_on=["qs2"],
        )
        qs4 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q2"],
            key="qs4",
            depends_on=["r2"],
        )
        qs5 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            [],
            key="qs5",
            depends_on=["qs4"],
        )
        auto.add_layer(qs1)
        auto.add_layer(qs2)
        auto.add_layer(r1)
        auto.add_layer(r2)
        auto.add_layer(qs3)
        auto.add_layer(qs4)
        auto.add_layer(qs5)

        # Assert the initial state of some layers
        assert [n.status for n in qs1.nodes] == [Status.READY] * 4
        assert [n.status for n in r1.nodes] == [
            Status.READY,
            Status.READY,
            Status.EMPTY,
            Status.EMPTY,
        ]
        assert qs1.status == Status.READY
        assert r1.status == Status.READY
        assert qs5.status == Status.EMPTY

        # Run the automation graph
        auto.run()

        # Assert status of layers after runing
        assert qs1.status == Status.PASSED
        assert qs2.status == Status.PASSED
        assert r1.status == Status.PASSED
        assert r2.status == Status.PASSED
        assert qs3.status == Status.DEACTIVATED
        assert qs4.status == Status.PASSED
        assert qs5.status == Status.DEACTIVATED

        # Assert status of nodes after runing
        assert [n.status for n in qs1.nodes] == [Status.PASSED] * 4
        assert [n.status for n in qs2.nodes] == [
            Status.DEACTIVATED,
            Status.PASSED,
            Status.PASSED,
            Status.PASSED,
        ]
        assert [n.status for n in r1.nodes] == [
            Status.DEACTIVATED,
            Status.PASSED,
            Status.EMPTY,
            Status.EMPTY,
        ]
        assert [n.status for n in r2.nodes] == [
            Status.DEACTIVATED,
            Status.PASSED,
            Status.EMPTY,
            Status.EMPTY,
        ]
        assert [n.status for n in qs3.nodes] == [
            Status.DEACTIVATED,
            Status.EMPTY,
            Status.EMPTY,
            Status.EMPTY,
        ]
        assert [n.status for n in qs3.nodes] == [
            Status.DEACTIVATED,
            Status.EMPTY,
            Status.EMPTY,
            Status.EMPTY,
        ]

    def test_run_layer(self, auto, qubit_spectroscopy_workflow):
        qs1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on=["__root__"],
        )
        auto.add_layer(qs1)
        output = auto.run_layer("qs1")
        assert isinstance(output, tuple)
        assert len(output) == 3
        eval_outputs = output[0]
        assert all(isinstance(k, str) for k in eval_outputs)
        assert all(isinstance(v, dict) for v in eval_outputs.values())
        for eval_output in eval_outputs.values():
            for k, v in eval_output.items():
                assert isinstance(k, str)
                assert isinstance(v, bool)
        assert isinstance(qs1.workflow_results, list)
        assert all(
            isinstance(workflow_result, WorkflowResult)
            for workflow_result in qs1.workflow_results
        )

    def test_run_layer_sequentially(self, auto, qubit_spectroscopy_workflow):
        qs1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on=["__root__"],
        )
        auto.add_layer(qs1)
        qs1.sequential = True
        output = auto.run_layer("qs1")
        assert isinstance(output, tuple)
        assert len(output) == 3
        eval_outputs = output[0]
        assert all(isinstance(k, str) for k in eval_outputs)
        assert all(isinstance(v, dict) for v in eval_outputs.values())
        for eval_output in eval_outputs.values():
            for k, v in eval_output.items():
                assert isinstance(k, str)
                assert isinstance(v, bool)
        assert isinstance(qs1.workflow_results, list)
        assert all(
            isinstance(workflow_result, WorkflowResult)
            for workflow_result in qs1.workflow_results
        )

    def test_reset(self, auto, qubit_spectroscopy_workflow, workflow_parameters):
        layer1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on=["__root__"],
        )
        layer2 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs2",
            depends_on=["qs1"],
            workflow_parameters=workflow_parameters,
        )
        auto.add_layer(layer1)
        auto.add_layer(layer2)

        assert layer1.fail_count == 0
        assert layer1.success_count == 0
        assert layer1.timestamp is None
        assert layer1.workflow_results is None
        assert auto.get_node("qs2_q0").status == Status.READY
        assert auto.get_node("qs1_q1").status == Status.READY
        assert auto.get_node("qs2_q0").fail_count == 0
        assert auto.get_node("qs2_q0").timestamp is None
        assert auto.get_node("qs2_q0").workflow_result is None

        auto.run()

        assert layer1.fail_count == 0
        assert layer1.success_count == 1
        assert isinstance(layer1.timestamp, str)
        assert isinstance(layer1.workflow_results, list)
        assert all(
            isinstance(workflow_result, WorkflowResult)
            for workflow_result in layer1.workflow_results
        )

        assert auto.get_node("qs2_q0").status == Status.DEACTIVATED
        assert auto.get_node("qs1_q1").status == Status.PASSED
        assert auto.get_node("qs1_q1").success_count == 1
        assert (
            auto.get_node("qs2_q0").fail_count == auto.get_node("qs2_q0").max_fail_count
        )
        assert isinstance(auto.get_node("qs2_q0").timestamp, str)
        assert isinstance(auto.get_node("qs2_q0").workflow_result, WorkflowResult)

        auto.reset()

        assert layer1.fail_count == 0
        assert layer1.success_count == 0
        assert layer1.timestamp is None
        assert layer1.workflow_results is None
        assert auto.get_node("qs2_q0").status == Status.READY
        assert auto.get_node("qs1_q1").status == Status.READY
        assert auto.get_node("qs1_q1").success_count == 0
        assert auto.get_node("qs2_q0").fail_count == 0
        assert auto.get_node("qs2_q0").timestamp is None
        assert auto.get_node("qs2_q0").workflow_result is None

    def test_set_temp_quantum_elements(
        self,
        auto,
        qubit_spectroscopy_workflow,
    ):
        quantum_elements = ["q0", "q1", "q2", "q3"]
        layer1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            quantum_elements,
            key="qs1",
            depends_on=["__root__"],
        )

        # Test passing a single quantum element as list
        assert layer1.quantum_elements == quantum_elements
        temp_layer = auto._set_temp_parameters(layer1, ["q0"])
        assert temp_layer.quantum_elements == ["q0"]
        assert layer1.quantum_elements == ["q0"]

        with pytest.raises(
            ValueError,
            match=r"The set of quantum elements {'q1'} is not in the layer. ",
        ):
            temp_layer = auto._set_temp_parameters(layer1, ["q1"])

        # Test passing a single quantum element as string
        layer1.quantum_elements = quantum_elements
        assert layer1.quantum_elements == quantum_elements
        temp_layer = auto._set_temp_parameters(layer1, "q0")
        assert temp_layer.quantum_elements == ["q0"]

        # Test recovery of parameters after execution of run layer
        layer1.quantum_elements = quantum_elements
        auto.add_layer(layer1)
        auto.run_layer("qs1", quantum_elements=["q0"])
        assert layer1.workflow_results[0].input["qubits"] == "q0"
        assert layer1.quantum_elements == quantum_elements

        auto.run_layer("qs1", quantum_elements=["q0", "q1"])
        assert layer1.workflow_results[0].input["qubits"] == ["q0", "q1"]
        assert layer1.quantum_elements == quantum_elements

    def test_set_temp_workflow_parameters(
        self, auto, qubit_spectroscopy_workflow, workflow_parameters
    ):
        quantum_elements = ["q0", "q1", "q2", "q3"]
        layer1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            quantum_elements,
            key="qs1",
            depends_on=["__root__"],
        )

        # Test passing temporary workflow parameters
        temp_wf_parameters = {
            "q1": {
                "frequencies": np.linspace(5.5e9, 5.9e9, 101),
                "evaluation_fit_r2_thresholds": 1.0,
            }
        }
        layer1.quantum_elements = quantum_elements
        layer1.workflow_parameters = workflow_parameters
        assert layer1.workflow_parameters == workflow_parameters
        temp_layer = auto._set_temp_parameters(
            layer1, quantum_elements, workflow_parameters=temp_wf_parameters
        )
        expected_wf_parameters = {
            "q0": {
                "frequencies": np.linspace(6e9, 6.2e9, 101),
                "evaluation_fit_r2_thresholds": 1.0,
            },
            "q1": {
                "frequencies": np.linspace(5.5e9, 5.9e9, 101),
                "evaluation_fit_r2_thresholds": 1.0,
            },
            "q2": {"frequencies": np.linspace(6e9, 6.2e9, 101)},
            "q3": {"frequencies": np.linspace(6e9, 6.2e9, 101)},
        }
        assert layer1.workflow_parameters["q1"]["evaluation_fit_r2_thresholds"] == 1.0
        np.testing.assert_equal(layer1.workflow_parameters, expected_wf_parameters)

        assert (
            temp_layer.workflow_parameters["q1"]["evaluation_fit_r2_thresholds"] == 1.0
        )
        np.testing.assert_equal(temp_layer.workflow_parameters, expected_wf_parameters)

        # Test recovery of parameters after execution of run layer
        layer1.quantum_elements = quantum_elements
        layer1.workflow_parameters = workflow_parameters
        assert layer1.workflow_parameters == workflow_parameters
        auto.add_layer(layer1)
        auto.run_layer("qs1", workflow_parameters=temp_wf_parameters)
        workflow_input = layer1.workflow_results[0].input
        np.testing.assert_almost_equal(
            workflow_input["frequencies"],
            [v["frequencies"] for v in expected_wf_parameters.values()],
        )
        for qubit, qubit_parameters in layer1.workflow_parameters.items():
            for qubit_parameter, values in qubit_parameters.items():
                np.testing.assert_almost_equal(
                    values, workflow_parameters[qubit][qubit_parameter]
                )

    def test_set_temp_general_workflow_parameters(self, auto, amplitude_fine_workflow):
        quantum_elements = ["q0", "q1", "q2", "q3"]
        layer1 = WorkflowLayer(
            amplitude_fine_workflow,
            quantum_elements,
            key="af1",
            depends_on=["__root__"],
        )
        auto.add_layer(layer1)

        general_workflow_parameters = {
            "repetitions": [
                [1, 2],
                [1, 2],
                [1, 2],
                [1, 2],
            ],
        }

        # Test passing temporary general workflow parameters
        temp_general_wf_parameters = {
            "repetitions": [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
            ],
        }
        layer1.quantum_elements = quantum_elements
        layer1.general_workflow_parameters = general_workflow_parameters
        assert layer1.general_workflow_parameters == general_workflow_parameters
        auto._set_temp_parameters(
            layer1,
            quantum_elements,
            general_workflow_parameters=temp_general_wf_parameters,
        )
        assert layer1.general_workflow_parameters == temp_general_wf_parameters

        # Test recovery of parameters after execution of run layer
        auto.run_layer("af1", general_workflow_parameters=temp_general_wf_parameters)
        assert layer1.status == Status.PASSED
        np.testing.assert_equal(
            layer1.workflow_results[0].input["repetitions"],
            temp_general_wf_parameters["repetitions"],
        )
        np.testing.assert_equal(
            layer1.workflow_results[0].output.data.q0.result.axis[0],
            temp_general_wf_parameters["repetitions"],
        )
        np.testing.assert_equal(
            layer1.general_workflow_parameters, general_workflow_parameters
        )

    def test_set_temp_qpu_parameters(self, auto, qubit_spectroscopy_workflow):
        quantum_elements = ["q0", "q1", "q2", "q3"]
        layer1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            quantum_elements,
            key="qs1",
            depends_on=["__root__"],
        )
        auto.add_layer(layer1)

        # Test passing temporary temporary qpu parameters
        temporary_qpu_parameters = {
            "q0": {"drive_lo_frequency": 6.6e9},
            "q1": {"drive_lo_frequency": 6.6e9},
            "q2": {"drive_lo_frequency": 6.6e9},
            "q3": {"drive_lo_frequency": 6.6e9},
        }

        assert layer1.temporary_qpu_parameters is None
        auto._set_temp_parameters(
            layer1, quantum_elements, temporary_qpu_parameters=temporary_qpu_parameters
        )
        assert layer1.temporary_qpu_parameters == temporary_qpu_parameters

        # Test recovery of parameters after execution of run layer
        layer1.temporary_qpu_parameters = None
        auto.run_layer("qs1", temporary_qpu_parameters=temporary_qpu_parameters)
        assert (
            layer1.workflow_results[0].input["temporary_parameters"]
            == temporary_qpu_parameters
        )
        assert layer1.temporary_qpu_parameters is None

    def test_set_temp_logic(self, auto, ramsey_workflow):
        layer1 = WorkflowLayer(
            ramsey_workflow,
            ["q0", "q1"],
            key="r1",
            depends_on=["__root__"],
        )
        layer2 = WorkflowLayer(
            ramsey_workflow,
            ["q0", "q3"],
            key="r2",
            depends_on=["r1"],
        )
        auto.add_layer(layer1)
        auto.add_layer(layer2)

        # Test passing temporary logic
        l2_logic = FixedParameterUpdate(
            new_layer_key="r2",
            parameter_changes={
                "q0": {"detunings": -0.1},
                "q3": {"detunings": -0.1},
            },
            relative=True,
            iterations=3,
        )

        assert layer2.workflow_parameters["q0"]["detunings"] == 670000.0
        assert layer2.workflow_parameters["q3"]["detunings"] == 670000.0
        auto._set_temp_parameters(layer2, ["q0", "q3"], logic=l2_logic)
        auto.run()
        np.testing.assert_almost_equal(
            layer2.workflow_parameters["q0"]["detunings"], 670000.0 * 0.9**3
        )
        np.testing.assert_almost_equal(
            layer2.workflow_parameters["q3"]["detunings"], 670000.0 * 0.9**3
        )

        l1_logic = FixedParameterUpdate(
            new_layer_key="r1",
            parameter_changes={
                "q0": {"delays": 1e-5},
                "q1": {"delays": 1e-5},
            },
            relative=False,
            iterations=3,
        )

        # Test recovery of parameters after execution of run layer
        np.testing.assert_equal(
            layer1.workflow_parameters["q0"]["delays"], np.linspace(0, 2e-5, 50)
        )
        np.testing.assert_equal(
            layer1.workflow_parameters["q1"]["delays"], np.linspace(2e-5, 5e-5, 50)
        )
        eval_outputs, new_layer_key, new_params = auto.run_layer("r1", logic=l1_logic)
        assert new_layer_key == "r1"
        np.testing.assert_equal(
            new_params["q0"]["delays"], np.linspace(0, 2e-5, 50) + 1e-5
        )
        np.testing.assert_equal(
            new_params["q1"]["delays"], np.linspace(2e-5, 5e-5, 50) + 1e-5
        )
        assert eval_outputs == {
            "q0": {"success": True, "update": False},
            "q1": {"success": True, "update": False},
        }

    def test_set_temp_workflow_options(self, auto, ramsey_workflow):
        quantum_elements = ["q0", "q1"]

        layer1 = WorkflowLayer(
            ramsey_workflow,
            quantum_elements,
            key="r1",
            depends_on=["__root__"],
        )
        auto.add_layer(layer1)

        # Test passing temporary workflow options
        assert layer1.workflow_options.update[0].option.update
        auto._set_temp_parameters(
            layer1, quantum_elements, workflow_options={"update": False}
        )
        assert not layer1.workflow_options.update[0].option.update

        workflow_options = copy.deepcopy(layer1.workflow_options)
        workflow_options.update(True)
        assert not layer1.workflow_options.update[0].option.update
        auto._set_temp_parameters(
            layer1, quantum_elements, workflow_options=workflow_options
        )
        assert layer1.workflow_options.update[0].option.update

        # Test recovery of options after execution of run layer
        layer1.workflow_options.update[0].option.update = False
        assert not layer1.workflow_options.update[0].option.update
        auto.run_layer("r1", workflow_options={"update": True})
        assert layer1.workflow_results[0].input["options"].update
        assert not layer1.workflow_options.update[0].option.update

        layer2 = WorkflowLayer(
            ramsey_workflow,
            ["q0", "q3"],
            key="r2",
            depends_on=["__root__"],
        )
        auto.add_layer(layer2)

        assert not layer2.workflow_options.update[0].option.update
        opts = layer2.workflow_options
        opts.update(True)
        auto.run_layer("r2", workflow_options=opts)
        assert layer2.workflow_results[0].input["options"].update
