# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications._automation.workflow.workflow_automation"""

from __future__ import annotations

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
from laboneq_applications.experiments import qubit_spectroscopy, ramsey
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
                "update": False,
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
                "update": True,
            },
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
        auto.run()

    def test_run_layer_sequentially(self, auto, qubit_spectroscopy_workflow):
        qs1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on=["__root__"],
        )
        auto.add_layer(qs1)
        qs1.sequential = True
        auto.run_layer("qs1")

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
        assert auto.get_node("qs2_q0").workflow_results is None

        auto.run()

        assert layer1.fail_count == 0
        assert layer1.success_count == 1
        assert type(layer1.timestamp) is str
        assert type(layer1.workflow_results) is WorkflowResult
        assert auto.get_node("qs2_q0").status == Status.DEACTIVATED
        assert auto.get_node("qs1_q1").status == Status.PASSED
        assert auto.get_node("qs1_q1").success_count == 1
        assert (
            auto.get_node("qs2_q0").fail_count == auto.get_node("qs2_q0").max_fail_count
        )
        assert type(auto.get_node("qs2_q0").timestamp) is str
        assert type(auto.get_node("qs2_q0").workflow_results) is WorkflowResult

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
        assert auto.get_node("qs2_q0").workflow_results is None
