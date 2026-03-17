# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the workflow automation."""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pytest
from laboneq.automation import AutomationStatus as Status
from laboneq.automation.layer import RootLayer
from laboneq.automation.logic import FixedParameterUpdate
from laboneq.automation.node import RootNode
from laboneq.dsl.device.connection import create_connection
from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.device.instruments import HDAWG
from laboneq.dsl.quantum import (
    QPU,
    QuantumElement,
    QuantumPlatform,
)
from laboneq.dsl.session import Session
from laboneq.workflow import WorkflowBuilder, logbook
from laboneq.workflow.result import WorkflowResult

from laboneq_applications.automation.web_viewer.server import start_web_viewer
from laboneq_applications.automation.workflow.automation import (
    WorkflowAutomation,
)
from laboneq_applications.automation.workflow.layer import WorkflowLayer
from laboneq_applications.contrib.experiments import zz_coupling_strength
from laboneq_applications.experiments import (
    amplitude_fine,
    qubit_spectroscopy,
    ramsey,
)
from laboneq_applications.qpu_types.tunable_coupler import TunableCoupler
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonOperations,
    demo_platform,
)


@pytest.fixture
def folder_store(tmp_path):
    store = logbook.FolderStore(tmp_path)
    store.activate()
    yield store
    store.deactivate()


@pytest.fixture
def qt_platform() -> QuantumPlatform:
    return demo_platform(n_qubits=4)


@pytest.fixture
def couplings() -> dict[str, tuple[str, str]]:
    return {
        "c_q0q1": ("q0", "q1"),
        "c_q1q2": ("q1", "q2"),
        "c_q2q3": ("q2", "q3"),
        "c_q3q0": ("q3", "q0"),
    }


@pytest.fixture
def couplers(qt_platform, device_setup, couplings) -> list[QuantumElement]:
    for n, key in enumerate(couplings):
        channel_id = len(qt_platform.qpu.quantum_elements) + n
        device_setup.add_connections(
            "hdawg_0",
            create_connection(to_signal=f"{key}/flux", ports=f"SIGOUTS/{channel_id}"),
        )
    return TunableCoupler.from_device_setup(
        device_setup, qubit_uids=list(couplings.keys())
    )


@pytest.fixture
def qpu(qt_platform, couplers, couplings) -> QPU:
    qops = TunableTransmonOperations()
    qpu = QPU(qt_platform.qpu.quantum_elements + couplers, quantum_operations=qops)
    for coupler, (q0, q1) in couplings.items():
        qpu.topology.add_edge(
            source_node=q0, target_node=q1, quantum_element=coupler, tag="coupler"
        )
        qpu.topology.add_edge(
            source_node=q1, target_node=q0, quantum_element=coupler, tag="coupler"
        )
    return qpu


@pytest.fixture
def device_setup(qt_platform) -> DeviceSetup:
    qt_platform.setup.add_instruments(HDAWG(uid="hdawg_0", address="dev8800"))
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
            "workflow_parameters": {
                "q0": {"frequencies": np.linspace(6.0e9, 6.50e9, 101)},
                "q1": {"frequencies": np.linspace(6.0e9, 6.50e9, 101)},
                "q2": {"frequencies": np.linspace(6.0e9, 6.50e9, 101)},
                "q3": {"frequencies": np.linspace(6.0e9, 6.50e9, 101)},
            },
            "options": {
                "evaluate": True,
                "update": True,
                "count": 2048,
                "active_reset": True,
            },
        },
        "qs2": {
            "workflow_parameters": {
                "q0": {"frequencies": np.linspace(6.1e9, 6.5e9, 101)},
                "q1": {"frequencies": np.linspace(6.1e9, 6.5e9, 101)},
                "q2": {"frequencies": np.linspace(6.1e9, 6.5e9, 101)},
                "q3": {"frequencies": np.linspace(6.1e9, 6.5e9, 101)},
            },
            "options": {
                "evaluate": True,
                "update": False,
            },
        },
        "r1": {
            "workflow_parameters": {
                "q0": {
                    "delays": np.linspace(0.0e00, 2.0e-05, 50),
                    "detunings": 670000.0,
                },
                "q1": {"delays": np.linspace(2e-05, 5e-05, 50), "detunings": 670000.0},
            },
            "options": {
                "evaluate": True,
                "update": True,
            },
        },
        "qs3": {
            "workflow_parameters": {
                "q0": {"frequencies": np.linspace(6.1e9, 6.5e9, 55)},
                "q1": {"frequencies": np.linspace(6.1e9, 6.5e9, 55)},
                "q2": {"frequencies": np.linspace(6.1e9, 6.5e9, 55)},
                "q3": {"frequencies": np.linspace(6.1e9, 6.5e9, 55)},
            },
            "options": {
                "evaluate": True,
                "update": True,
                "active_reset": True,
            },
        },
        "qs4": {
            "workflow_parameters": {
                "q0": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
                "q2": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
            },
            "options": {
                "evaluate": True,
                "update": True,
                "count": 2048,
                "active_reset": True,
            },
        },
        "qs5": {
            "workflow_parameters": {
                "q0": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
                "q1": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
                "q2": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
                "q3": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
            },
            "options": {
                "evaluate": True,
                "update": True,
                "count": 2048,
                "active_reset": True,
            },
        },
        "qs6": {
            "workflow_parameters": {
                "q0": {"frequencies": np.linspace(6.0e9, 6.5e9, 55)},
            },
        },
        "r2": {
            "workflow_parameters": {
                "q0": {"delays": np.linspace(0, 2.0e-05, 33), "detunings": 670000.0},
                "q3": {"delays": np.linspace(0, 9.3e-05, 33), "detunings": 670000.0},
            },
            "options": {
                "evaluate": False,
                "update": False,
            },
        },
        "af1": {
            "workflow_parameters": {
                "q0": {},
                "q1": {},
                "q2": {},
                "q3": {},
            },
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
            "options": {
                "evaluate": False,
                "update": False,
                "active_reset": True,
            },
        },
        "zz": {
            "workflow_parameters": {
                ("q0", "q1"): {
                    "biases": list(np.linspace(-0.06, 0.06, 11)),
                    "delays": list(np.linspace(0, 10e-6, 11)),
                },
                ("q2", "q3"): {
                    "biases": list(np.linspace(-0.06, 0.06, 11)),
                    "delays": list(np.linspace(0, 10e-6, 11)),
                },
            }
        },
    }


@pytest.fixture
def auto(session, qpu, automation_parameters) -> WorkflowAutomation:
    return WorkflowAutomation(
        session, qpu=qpu, automation_parameters=automation_parameters, name="test"
    )


@pytest.fixture
def workflow_parameters() -> dict:
    return {
        "workflow_parameters": {
            "q0": {
                "frequencies": np.linspace(6e9, 6.2e9, 101),
            },
            "q1": {"frequencies": np.linspace(6e9, 6.2e9, 101)},
            "q2": {"frequencies": np.linspace(6e9, 6.2e9, 101)},
            "q3": {"frequencies": np.linspace(6e9, 6.2e9, 101)},
        },
        "evaluation_parameters": {
            "fit_r2_thresholds": {
                "q0": 1,
            }
        },
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
            session, qpu, automation_parameters=automation_parameters, name="test"
        )
        assert auto.name == "test"
        assert auto.automation_parameters == automation_parameters
        assert auto.session == session
        assert auto.qpu == qpu
        assert isinstance(auto.timestamp, str)
        assert isinstance(auto._node_graph, nx.DiGraph)
        assert list(auto._node_graph.nodes) == ["root_root"]
        assert auto._node_lookup == {"root_root": RootNode()}
        assert auto._layer_lookup == {"root": RootLayer()}

        # WorkflowAutomation methods
        assert hasattr(auto, "_run_layer")
        method2 = auto._run_layer
        assert callable(method2)
        assert len(inspect.signature(method2).parameters) == 1

    def test_run(
        self,
        auto,
        qubit_spectroscopy_workflow,
        ramsey_workflow,
        workflow_parameters,
    ):
        qs1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on={"root"},
        )
        qs2 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs2",
            depends_on={"qs1"},
            parameters=workflow_parameters,
        )
        r1 = WorkflowLayer(
            ramsey_workflow,
            ["q0", "q1"],
            key="r1",
            depends_on={"qs2"},
        )
        r2 = WorkflowLayer(
            ramsey_workflow,
            ["q0", "q3"],
            key="r2",
            depends_on={"qs2"},
        )
        qs3 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0"],
            key="qs3",
            depends_on={"qs2"},
        )
        qs4 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q2"],
            key="qs4",
            depends_on={"r2"},
        )
        auto.add_layer(qs1)
        auto.add_layer(qs2)
        auto.add_layer(r1)
        auto.add_layer(r2)
        auto.add_layer(qs3)
        with pytest.raises(
            ValueError,
            match=r"Layer `qs4` cannot depend on layer `r2` "
            r"because they have no common node keys.",
        ):
            auto.add_layer(qs4)

        # Assert the initial state of some layers
        assert [n.status for n in qs1.nodes.values()] == [Status.READY] * 4
        assert [n.status for n in r1.nodes.values()] == [
            Status.READY,
            Status.READY,
        ]
        assert qs1.status == Status.READY
        assert r1.status == Status.READY

        # Run the automation graph
        auto.run()

        # Assert status of layers after runing
        assert qs1.status == Status.PASSED
        assert qs2.status == Status.PASSED
        assert r1.status == Status.PASSED
        assert r2.status == Status.PASSED
        assert qs3.status == Status.DEACTIVATED

        # Assert status of nodes after running
        assert [n.status for n in qs1.nodes.values()] == [Status.PASSED] * 4
        assert [n.status for n in qs2.nodes.values()] == [
            Status.DEACTIVATED_FAIL,
            Status.PASSED,
            Status.PASSED,
            Status.PASSED,
        ]
        assert [n.status for n in r1.nodes.values()] == [
            Status.DEACTIVATED,
            Status.PASSED,
        ]
        assert [n.status for n in r2.nodes.values()] == [
            Status.DEACTIVATED,
            Status.PASSED,
        ]
        assert [n.status for n in qs3.nodes.values()] == [
            Status.DEACTIVATED,
        ]

    def test_run_layer(self, auto, qubit_spectroscopy_workflow):
        qs1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on={"root"},
        )
        auto.add_layer(qs1)
        output = auto.run_layer("qs1")
        assert isinstance(output, tuple)
        assert len(output) == 2
        eval_outputs = qs1.eval_outputs
        assert all(isinstance(k, str) for k in eval_outputs)
        assert all(isinstance(v, dict) for v in eval_outputs.values())
        for eval_output in eval_outputs.values():
            for k, v in eval_output.items():
                assert isinstance(k, str)
                assert isinstance(v, bool)
        assert isinstance(qs1.workflow_results, dict)
        assert all(
            isinstance(workflow_result, WorkflowResult)
            for workflow_result in qs1.workflow_results.values()
        )

    def test_run_layer_sequentially(self, auto, qubit_spectroscopy_workflow):
        qs1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on={"root"},
        )
        auto.add_layer(qs1)
        qs1.sequential = True
        output = auto.run_layer("qs1")
        assert isinstance(output, tuple)
        assert len(output) == 2
        eval_outputs = qs1.eval_outputs
        assert list(eval_outputs.keys()) == ["q0", "q1", "q2", "q3"]
        assert all(isinstance(v, dict) for v in eval_outputs.values())
        for eval_output in eval_outputs.values():
            assert eval_output == {"success": True, "update": False}
        assert list(qs1.workflow_results.keys()) == [
            (q,) for q in ["q0", "q1", "q2", "q3"]
        ]
        assert all(
            isinstance(workflow_result, WorkflowResult)
            for workflow_result in qs1.workflow_results.values()
        )

    def test_run_layer_sequentially_with_web_view(
        self, auto, qubit_spectroscopy_workflow
    ):
        start_web_viewer(auto, port=5003)
        qs1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on={"root"},
        )
        auto.add_layer(qs1)
        qs1.sequential = True
        output = auto.run_layer("qs1")
        assert isinstance(output, tuple)
        assert len(output) == 2
        eval_outputs = qs1.eval_outputs
        assert list(eval_outputs.keys()) == ["q0", "q1", "q2", "q3"]
        assert all(isinstance(v, dict) for v in eval_outputs.values())
        for eval_output in eval_outputs.values():
            assert eval_output == {"success": True, "update": False}
        assert list(qs1.workflow_results.keys()) == [
            (q,) for q in ["q0", "q1", "q2", "q3"]
        ]
        assert all(
            isinstance(workflow_result, WorkflowResult)
            for workflow_result in qs1.workflow_results.values()
        )

    def test_run_layer_sequentially_with_folder_store(
        self, auto, qubit_spectroscopy_workflow, folder_store
    ):

        qs1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on={"root"},
        )
        auto.add_layer(qs1)
        qs1.sequential = True
        auto.run_layer("qs1")

        [day_folder] = folder_store.folder.iterdir()
        [automation_folder] = day_folder.iterdir()
        assert [f.name for f in automation_folder.iterdir()] == ["qs1"]

        for qubit in ["q0", "q1", "q2", "q3"]:
            data_folders = [
                p for p in (automation_folder / qs1.key / qubit).iterdir() if p.is_dir()
            ]
            assert len(data_folders) == 1
            assert "qubit-spectroscopy" in data_folders[0].name

    def test_run_tuple_layer_sequentially_with_folder_store(self, auto, folder_store):

        zz = WorkflowLayer(
            zz_coupling_strength.experiment_workflow,
            [("q0", "q1"), ("q2", "q3")],
            key="zz",
            depends_on={"root"},
        )
        auto.add_layer(zz)
        zz.sequential = True
        auto.run_layer("zz")

        [day_folder] = folder_store.folder.iterdir()
        [automation_folder] = day_folder.iterdir()
        assert [f.name for f in automation_folder.iterdir()] == ["zz"]

        for element_name in ["q0-q1", "q2-q3"]:
            data_folders = [
                p
                for p in (automation_folder / zz.key / element_name).iterdir()
                if p.is_dir()
            ]
            assert len(data_folders) == 1
            assert "zz-coupling-strength" in data_folders[0].name

    def test_run_layer_with_folder_store(
        self, auto, qubit_spectroscopy_workflow, folder_store
    ):

        qs1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on={"root"},
        )
        auto.add_layer(qs1)
        auto.run_layer("qs1")

        [day_folder] = folder_store.folder.iterdir()
        [automation_folder] = day_folder.iterdir()
        assert [f.name for f in automation_folder.iterdir()] == ["qs1"]

        data_folders = [
            p for p in (automation_folder / qs1.key).iterdir() if p.is_dir()
        ]
        assert len(data_folders) == 1
        assert "qubit-spectroscopy" in data_folders[0].name

    def test_reset(self, auto, qubit_spectroscopy_workflow, workflow_parameters):
        layer1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on={"root"},
        )
        layer2 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs2",
            depends_on={"qs1"},
            parameters=workflow_parameters,
        )
        auto.add_layer(layer1)
        auto.add_layer(layer2)

        assert layer1.fail_count == {"q0": 0, "q1": 0, "q2": 0, "q3": 0}
        assert layer1.pass_count == {"q0": 0, "q1": 0, "q2": 0, "q3": 0}
        assert layer1.timestamp == {"q0": None, "q1": None, "q2": None, "q3": None}
        assert not layer1.workflow_results
        assert auto.get_node("qs2_q0").status == Status.READY
        assert auto.get_node("qs1_q1").status == Status.READY
        assert auto.get_node("qs2_q0").fail_count == 0
        assert auto.get_node("qs2_q0").timestamp is None

        auto.run()

        assert layer1.fail_count == {"q0": 0, "q1": 0, "q2": 0, "q3": 0}
        assert layer1.pass_count == {"q0": 1, "q1": 1, "q2": 1, "q3": 1}
        assert isinstance(layer1.timestamp, dict)
        assert isinstance(layer1.workflow_results, dict)
        assert all(
            isinstance(workflow_result, WorkflowResult)
            for workflow_result in layer1.results.values()
        )
        assert layer1.eval_outputs
        assert auto.get_node("qs2_q0").status == Status.DEACTIVATED_FAIL
        assert auto.get_node("qs1_q1").status == Status.PASSED
        assert auto.get_node("qs1_q1").pass_count == 1
        assert auto.get_node("qs2_q0").fail_count == 1
        assert isinstance(auto.get_node("qs2_q0").timestamp, str)

        auto.reset()

        assert layer1.fail_count == {"q0": 0, "q1": 0, "q2": 0, "q3": 0}
        assert layer1.pass_count == {"q0": 0, "q1": 0, "q2": 0, "q3": 0}
        assert layer1.timestamp == {"q0": None, "q1": None, "q2": None, "q3": None}
        assert not layer1.workflow_results
        assert not layer1.eval_outputs
        assert auto.get_node("qs2_q0").status == Status.READY
        assert auto.get_node("qs1_q1").status == Status.READY
        assert auto.get_node("qs1_q1").pass_count == 0
        assert auto.get_node("qs2_q0").fail_count == 0
        assert auto.get_node("qs2_q0").timestamp is None

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
            depends_on={"root"},
        )

        # Test recovery of parameters after execution of run layer
        layer1.quantum_elements = quantum_elements
        auto.add_layer(layer1)
        auto.run_layer("qs1", node_keys=["q0"])
        assert next(iter(layer1.workflow_results.values())).input["qubits"] == "q0"
        assert layer1.quantum_elements == quantum_elements

        auto.run_layer("qs1", node_keys=["q0", "q1"])
        assert [wr.input["qubits"] for wr in layer1.workflow_results.values()][1] == [
            "q0",
            "q1",
        ]
        assert layer1.quantum_elements == quantum_elements

    def test_set_temp_workflow_parameters(
        self, auto, qubit_spectroscopy_workflow, workflow_parameters
    ):
        quantum_elements = ["q0", "q1", "q2", "q3"]
        layer1 = WorkflowLayer(
            qubit_spectroscopy_workflow,
            quantum_elements,
            key="qs1",
            depends_on={"root"},
        )

        # Test passing temporary workflow parameters
        temp_parameters = {
            "workflow_parameters": {
                "q0": {
                    "frequencies": np.linspace(6e9, 6.2e9, 101),
                },
                "q1": {
                    "frequencies": np.linspace(5.5e9, 5.9e9, 101),
                },
                "q2": {"frequencies": np.linspace(6e9, 6.2e9, 101)},
                "q3": {"frequencies": np.linspace(6e9, 6.2e9, 101)},
            },
            "evaluation_parameters": {
                "fit_r2_thresholds": {
                    "q0": 1,
                    "q1": 1,
                }
            },
        }

        # Test recovery of parameters after execution of run layer
        layer1.quantum_elements = quantum_elements
        layer1.parameters = workflow_parameters
        assert layer1.parameters == workflow_parameters
        auto.add_layer(layer1)
        auto.run_layer("qs1", parameters=temp_parameters)
        workflow_input = next(iter(layer1.workflow_results.values())).input
        np.testing.assert_almost_equal(
            workflow_input["frequencies"],
            [v["frequencies"] for v in temp_parameters["workflow_parameters"].values()],
        )
        for qubit, qubit_parameters in layer1.parameters["workflow_parameters"].items():
            for qubit_parameter, values in qubit_parameters.items():
                np.testing.assert_almost_equal(
                    values,
                    workflow_parameters["workflow_parameters"][qubit][qubit_parameter],
                )

    def test_set_temp_logic(self, auto, ramsey_workflow):
        layer1 = WorkflowLayer(
            ramsey_workflow,
            ["q0", "q1"],
            key="r1",
            depends_on={"root"},
        )
        layer2 = WorkflowLayer(
            ramsey_workflow,
            ["q0", "q3"],
            key="r2",
            depends_on={"r1"},
        )
        auto.add_layer(layer1)
        auto.add_layer(layer2)

        # Test passing temporary logic
        l2_logic = FixedParameterUpdate(
            new_layer_key="r2",
            parameter_changes={
                "workflow_parameters": {
                    "q0": {"detunings": -0.1},
                    "q3": {"detunings": -0.1},
                },
            },
            relative=True,
            iterations=3,
        )

        assert layer2.parameters["workflow_parameters"]["q0"]["detunings"] == 670000.0
        assert layer2.parameters["workflow_parameters"]["q3"]["detunings"] == 670000.0
        layer2.logic = l2_logic
        auto.run()
        np.testing.assert_almost_equal(
            layer2.parameters["workflow_parameters"]["q0"]["detunings"],
            670000.0 * 0.9**3,
        )
        np.testing.assert_almost_equal(
            layer2.parameters["workflow_parameters"]["q3"]["detunings"],
            670000.0 * 0.9**3,
        )

        l1_logic = FixedParameterUpdate(
            new_layer_key="r1",
            parameter_changes={
                "workflow_parameters": {
                    "q0": {"delays": 1e-5},
                    "q1": {"delays": 1e-5},
                }
            },
            relative=False,
            iterations=3,
        )

        # Test recovery of parameters after execution of run layer
        np.testing.assert_equal(
            layer1.parameters["workflow_parameters"]["q0"]["delays"],
            np.linspace(0, 2e-5, 50),
        )
        np.testing.assert_equal(
            layer1.parameters["workflow_parameters"]["q1"]["delays"],
            np.linspace(2e-5, 5e-5, 50),
        )
        layer1.logic = l1_logic
        new_layer_key, _ = auto.run_layer("r1")
        assert new_layer_key == "r1"
        np.testing.assert_equal(
            layer1.parameters["workflow_parameters"]["q0"]["delays"],
            np.linspace(0, 2e-5, 50) + 1e-5,
        )
        np.testing.assert_equal(
            layer1.parameters["workflow_parameters"]["q1"]["delays"],
            np.linspace(2e-5, 5e-5, 50) + 1e-5,
        )
        assert layer1.eval_outputs == {
            "q0": {"success": True, "update": False},
            "q1": {"success": True, "update": False},
        }

    def test_zz_coupling(self, auto):
        qs_layer = WorkflowLayer(
            qubit_spectroscopy.experiment_workflow,
            ["q0", "q1", "q2", "q3"],
            key="qs1",
            depends_on={"root"},
        )
        zz_layer = WorkflowLayer(
            zz_coupling_strength.experiment_workflow,
            [("q0", "q1"), ("q2", "q3")],
            key="zz",
            depends_on={"qs1"},
        )
        ramsey_layer = WorkflowLayer(
            ramsey.experiment_workflow,
            ["q0", "q1", "q2", "q3"],
            key="r1",
            depends_on={"zz"},
        )

        auto.add_layer(qs_layer)
        auto.add_layer(zz_layer)
        auto.add_layer(ramsey_layer)

        auto.run()
