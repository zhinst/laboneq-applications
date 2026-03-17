# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the workflow automation utilities."""

import numpy as np
import pytest
from laboneq.dsl import Session

from laboneq_applications.automation import WorkflowAutomation, WorkflowLayer
from laboneq_applications.automation.workflow.utils import (
    get_eval_outputs,
    group_element_workflow_parameters,
)
from laboneq_applications.experiments import qubit_spectroscopy
from laboneq_applications.qpu_types.tunable_transmon import demo_platform


@pytest.fixture
def auto() -> WorkflowAutomation:

    platform = demo_platform(n_qubits=4)
    setup = platform.setup
    qpu = platform.qpu
    qubits = qpu.quantum_elements
    session = Session(setup)
    session.connect(do_emulation=True)

    qs1_params = {}
    qs1_params["workflow_parameters"] = {
        q.uid: {"frequencies": np.linspace(6e9, 6.3e9, 101)} for q in qubits
    }
    qs1_params["options"] = {
        "evaluate": True,
        "update": True,
        "count": 2048,
        "active_reset": True,
    }
    auto_params = {"qs1": qs1_params}

    auto = WorkflowAutomation(
        session, qpu, automation_parameters=auto_params, name="example"
    )

    af1 = WorkflowLayer(
        qubit_spectroscopy.experiment_workflow,
        ["q0", "q1", "q2", "q3"],
        key="qs1",
        depends_on={"root"},
    )
    auto.add_layer(af1)

    return auto


class TestUtils:
    def test_group_element_workflow_parameters(self):

        element_workflow_parameters = {
            "q0": {"frequencies": np.linspace(6e9, 6.3e9, 101)},
            "q1": {"frequencies": np.linspace(6.1e9, 6.4e9, 101)},
            "q2": {"frequencies": np.linspace(6.2e9, 6.5e9, 101)},
            "q3": {"frequencies": np.linspace(6.3e9, 6.6e9, 101)},
        }

        output = group_element_workflow_parameters(
            element_workflow_parameters, ["q0", "q1", "q2"]
        )
        assert np.array_equal(
            output["frequencies"],
            [
                np.linspace(6e9, 6.3e9, 101),
                np.linspace(6.1e9, 6.4e9, 101),
                np.linspace(6.2e9, 6.5e9, 101),
            ],
        )

        output = group_element_workflow_parameters(
            element_workflow_parameters, ["q0", "q2", "q3"]
        )
        assert np.array_equal(
            output["frequencies"],
            [
                np.linspace(6e9, 6.3e9, 101),
                np.linspace(6.2e9, 6.5e9, 101),
                np.linspace(6.3e9, 6.6e9, 101),
            ],
        )

        element_workflow_parameters_mixed = {
            "q0": {
                "frequencies": np.linspace(6e9, 6.3e9, 101),
                "amplitudes": [0.1, 0.2],
            },
            "q1": {"frequencies": np.linspace(6.1e9, 6.4e9, 101)},
        }
        output = group_element_workflow_parameters(
            element_workflow_parameters_mixed, ["q0", "q1"]
        )
        assert np.array_equal(
            output["frequencies"],
            [np.linspace(6e9, 6.3e9, 101), np.linspace(6.1e9, 6.4e9, 101)],
        )
        assert output["amplitudes"] == [[0.1, 0.2], None]

    def test_get_eval_outputs(self, auto):
        workflow_results = auto["qs1"].run_executable(auto)
        eval_outputs = get_eval_outputs(workflow_results)
        for q in auto.qpu.quantum_elements:
            assert np.array_equal(
                eval_outputs[q.uid], {"success": True, "update": False}
            )
