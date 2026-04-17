# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the workflow automation layer."""

import numpy as np
import pytest
from laboneq.automation import AutomationLayerResult, AutomationStatus
from laboneq.dsl import Session
from laboneq.dsl.quantum import QPU
from laboneq.workflow import WorkflowResult

from laboneq_applications.automation import (
    WorkflowAutomation,
    WorkflowLayer,
    WorkflowNode,
)
from laboneq_applications.experiments import amplitude_fine, qubit_spectroscopy, ramsey
from laboneq_applications.qpu_types.tunable_transmon import demo_platform


@pytest.fixture
def auto() -> WorkflowAutomation:

    platform = demo_platform(n_qubits=4)
    setup = platform.setup
    qpu = platform.qpu
    qubits = qpu.quantum_elements
    session = Session(setup)
    session.connect(do_emulation=True)

    af1_params = {}
    af1_params["workflow_parameters"] = {
        q.uid: {"repetitions": [1, 2, 3, 4]} for q in qubits
    }
    af1_params["workflow_parameters"] |= {
        "__common__": {
            "amplification_qop": "x180",
            "target_angle": 1.0,
            "phase_offset": 0.0,
        }
    }
    af1_params["evaluation_parameters"] = {
        "fit_r2_thresholds": {q.uid: 0.02 for q in qubits}
    }

    temporary_parameters_q0 = qubits[0].parameters.copy()
    temporary_parameters_q0.drive_lo_frequency = 1e9

    af1_params["temporary_parameters"] = {
        qubits[0].uid: temporary_parameters_q0,
        "q1": {"readout_lo_frequency": 8e9},
    }

    af1_params["options"] = {
        "evaluate": True,
        "update": True,
        "count": 2048,
        "active_reset": True,
    }
    auto_params = {"af1": af1_params}

    auto = WorkflowAutomation(
        session, qpu, automation_parameters=auto_params, name="example"
    )

    af1 = WorkflowLayer(
        amplitude_fine.experiment_workflow,
        ["q0", "q1", "q2", "q3"],
        key="af1",
        depends_on={"root"},
    )
    auto.add_layer(af1)

    return auto


@pytest.fixture
def layer() -> WorkflowLayer:
    return WorkflowLayer(
        qubit_spectroscopy.experiment_workflow,
        ["q0", "q1", "q2", "q3"],
        key="qs1",
        depends_on={"root"},
        qpu=demo_platform(n_qubits=4).qpu,
    )


class TestWorkflowLayer:
    def test_create(self, auto, layer):
        assert auto["af1"].qpu is None
        assert isinstance(layer.qpu, QPU)

    def test_nodes(self, auto, layer):
        assert auto["af1"].nodes == {
            q.uid: WorkflowNode(key=q.uid, depends_on=set(), layer_key="af1")
            for q in auto.qpu.quantum_elements
        }

        layer.depends_on = {"af1"}
        assert layer.nodes == {
            q.uid: WorkflowNode(key=q.uid, depends_on={f"af1_{q.uid}"}, layer_key="qs1")
            for q in layer.qpu.quantum_elements
        }

    def test_workflow_builder(self, layer):
        assert layer.workflow_builder is layer.function
        assert layer.workflow_builder is qubit_spectroscopy.experiment_workflow
        layer.workflow_builder = ramsey.experiment_workflow
        assert layer.workflow_builder is ramsey.experiment_workflow

    def test_quantum_elements(self, layer):
        assert layer.quantum_elements is layer.node_keys
        assert layer.quantum_elements == ["q0", "q1", "q2", "q3"]
        layer.quantum_elements = ["q0", "q1"]
        assert layer.quantum_elements == ["q0", "q1"]

    def test_active_quantum_elements(self, layer):
        for node in layer.nodes.values():
            node.status = AutomationStatus.READY
        assert layer.active_quantum_elements == ["q0", "q1", "q2", "q3"]

        layer.nodes["q1"].status = AutomationStatus.DEACTIVATED
        layer.nodes["q3"].status = AutomationStatus.DEACTIVATED_FAIL
        assert layer.active_quantum_elements == ["q0", "q2"]

        layer.nodes["q0"].status = AutomationStatus.FAILED
        assert layer.active_quantum_elements == ["q0", "q2"]

    def test_workflow_parameters(self, auto):
        assert np.array_equal(
            auto["af1"].workflow_parameters,
            {q.uid: {"repetitions": [1, 2, 3, 4]} for q in auto.qpu.quantum_elements}
            | {
                "__common__": {
                    "amplification_qop": "x180",
                    "target_angle": 1.0,
                    "phase_offset": 0.0,
                }
            },
        )
        auto["af1"].workflow_parameters = {
            q.uid: {"repetitions": [2, 3, 4, 5]} for q in auto.qpu.quantum_elements
        } | {
            "__common__": {
                "amplification_qop": "x180",
                "target_angle": 1.0,
                "phase_offset": 0.1,
            }
        }
        assert np.array_equal(
            auto["af1"].workflow_parameters,
            {q.uid: {"repetitions": [2, 3, 4, 5]} for q in auto.qpu.quantum_elements}
            | {
                "__common__": {
                    "amplification_qop": "x180",
                    "target_angle": 1.0,
                    "phase_offset": 0.1,
                }
            },
        )

    def test_element_workflow_parameters(self, auto):
        assert np.array_equal(
            auto["af1"].element_workflow_parameters,
            {q.uid: {"repetitions": [1, 2, 3, 4]} for q in auto.qpu.quantum_elements},
        )
        auto["af1"].element_workflow_parameters = {
            q.uid: {"repetitions": [2, 3, 4, 5]} for q in auto.qpu.quantum_elements
        }
        assert np.array_equal(
            auto["af1"].element_workflow_parameters,
            {q.uid: {"repetitions": [2, 3, 4, 5]} for q in auto.qpu.quantum_elements},
        )

    def test_common_workflow_parameters(self, auto):
        assert auto["af1"].common_workflow_parameters == {
            "amplification_qop": "x180",
            "target_angle": 1.0,
            "phase_offset": 0.0,
        }
        auto["af1"].common_workflow_parameters = {
            "amplification_qop": "x180",
            "target_angle": 1.0,
            "phase_offset": 0.1,
        }
        assert auto["af1"].common_workflow_parameters == {
            "amplification_qop": "x180",
            "target_angle": 1.0,
            "phase_offset": 0.1,
        }

    def test_evaluation_parameters(self, auto):
        assert auto["af1"].evaluation_parameters == {
            "fit_r2_thresholds": {q.uid: 0.02 for q in auto.qpu.quantum_elements}
        }
        auto["af1"].evaluation_parameters = {
            "fit_r2_thresholds": {q.uid: 0.99 for q in auto.qpu.quantum_elements}
        }
        assert auto["af1"].evaluation_parameters == {
            "fit_r2_thresholds": {q.uid: 0.99 for q in auto.qpu.quantum_elements}
        }

    def test_temporary_parameters(self, auto):
        qubits = auto.qpu.quantum_elements
        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        temp_params = {
            qubits[0].uid: temporary_parameters_q0,
            "q1": {"readout_lo_frequency": 8e9},
        }

        assert auto["af1"].temporary_parameters == temp_params
        auto["af1"].temporary_parameters = {"q1": {"readout_lo_frequency": 1e9}}
        assert auto["af1"].temporary_parameters == {"q1": {"readout_lo_frequency": 1e9}}

    def test_options(self, auto):
        assert auto["af1"].options == {
            "evaluate": True,
            "update": True,
            "count": 2048,
            "active_reset": True,
        }
        auto["af1"].options = {
            "evaluate": False,
            "update": False,
            "count": 1024,
            "active_reset": False,
        }
        assert auto["af1"].options == {
            "evaluate": False,
            "update": False,
            "count": 1024,
            "active_reset": False,
        }

    def test_workflow_results(self, auto):
        assert auto["af1"].workflow_results is auto["af1"].results
        auto["af1"].temporary_parameters = {}
        auto["af1"].options = {
            "evaluate": False,
            "update": False,
            "count": 2048,
            "active_reset": True,
        }
        output = auto["af1"].run_executable(auto)
        assert output.successes == {}
        workflow_results = auto["af1"].workflow_results
        assert list(workflow_results.keys()) == [("q0", "q1", "q2", "q3")]
        assert isinstance(workflow_results[("q0", "q1", "q2", "q3")], WorkflowResult)
        auto["af1"].workflow_results = {}
        assert auto["af1"].workflow_results == {}

    def test_run_executable(self, auto):
        auto["af1"].temporary_parameters = {}
        for node in auto["af1"].nodes.values():
            assert node.status == AutomationStatus.READY
        output = auto["af1"].run_executable(auto)
        assert output == AutomationLayerResult(
            results=auto["af1"].workflow_results,
            successes={"q0": False, "q1": False, "q2": False, "q3": False},
        )
        workflow_results = auto["af1"].workflow_results
        for node in auto["af1"].nodes.values():
            assert node.status == AutomationStatus.FAILED
        assert list(workflow_results.keys()) == [("q0", "q1", "q2", "q3")]
        assert isinstance(workflow_results[("q0", "q1", "q2", "q3")], WorkflowResult)
        assert workflow_results[("q0", "q1", "q2", "q3")].tasks[
            "analysis_workflow"
        ].output == {
            "old_parameter_values": {"q0": {}, "q1": {}, "q2": {}, "q3": {}},
            "new_parameter_values": {"q0": {}, "q1": {}, "q2": {}, "q3": {}},
        }
