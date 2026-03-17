# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for the workflow automation logic."""

import numpy as np
import pytest
from laboneq.dsl import Session

from laboneq_applications.automation import WorkflowAutomation, WorkflowLayer
from laboneq_applications.automation.workflow.logic import AdaptFrequencyRange
from laboneq_applications.experiments import qubit_spectroscopy
from laboneq_applications.qpu_types.tunable_transmon import demo_platform

RANGE_THRESHOLDS = {
    0: 1.1,
    200_000_000: 1.2,
    400_000_000: 1.3,
    600_000_000: 1.4,
}


@pytest.fixture
def auto() -> WorkflowAutomation:

    platform = demo_platform(n_qubits=2)
    setup = platform.setup
    qpu = platform.qpu
    session = Session(setup)
    session.connect(do_emulation=True)

    qs1_params = {}
    qs1_params["workflow_parameters"] = {
        "q0": {"frequencies": np.linspace(5.9e9, 6.4e9, 101)},
        "q1": {"frequencies": np.linspace(6.1e9, 6.3e9, 101)},
    }
    qs1_params["options"] = {
        "evaluate": True,
        "update": True,
        "count": 2048,
        "active_reset": True,
    }
    auto_params = {"qs1": qs1_params, "qs2": qs1_params}

    auto = WorkflowAutomation(
        session, qpu, automation_parameters=auto_params, name="example"
    )

    qs1 = WorkflowLayer(
        qubit_spectroscopy.experiment_workflow,
        ["q0", "q1"],
        key="qs1",
        depends_on={"root"},
    )
    auto.add_layer(qs1)

    qs2 = WorkflowLayer(
        qubit_spectroscopy.experiment_workflow,
        ["q0", "q1"],
        key="qs2",
        depends_on={"qs1"},
    )
    auto.add_layer(qs2)

    return auto


class TestAdaptFrequencyRange:
    def test_get_bucket_value(self):
        s = RANGE_THRESHOLDS

        assert AdaptFrequencyRange.get_bucket_value(s, 0) == 1.1
        assert AdaptFrequencyRange.get_bucket_value(s, 100_000_000) == 1.1
        assert AdaptFrequencyRange.get_bucket_value(s, 200_000_000) == 1.2
        assert AdaptFrequencyRange.get_bucket_value(s, 300_000_000) == 1.2
        assert AdaptFrequencyRange.get_bucket_value(s, 400_000_000) == 1.3
        assert AdaptFrequencyRange.get_bucket_value(s, 500_000_000) == 1.3
        assert AdaptFrequencyRange.get_bucket_value(s, 600_000_000) == 1.4
        assert AdaptFrequencyRange.get_bucket_value(s, 800_000_000) == 1.4

        with pytest.raises(ValueError, match="less than all bucket lower bounds"):
            AdaptFrequencyRange.get_bucket_value(s, -1)

    def test_run_executable(self, auto):
        auto.run_layer("qs1")
        freq_logic = AdaptFrequencyRange(
            new_layer_key="qs2",
            range_thresholds=RANGE_THRESHOLDS,
        )
        next_key, updates = freq_logic.run_executable(auto["qs1"])

        assert next_key == "qs2"
        new_params = updates["workflow_parameters"]

        freqs_q0 = auto["qs1"].workflow_parameters["q0"]["frequencies"]
        freqs_q1 = auto["qs1"].workflow_parameters["q1"]["frequencies"]

        # q0: range 500 MHz, multiplier 1.3
        midpoint_q0 = (max(freqs_q0) + min(freqs_q0)) / 2
        expected_q0 = (freqs_q0 - midpoint_q0) * 1.3 + midpoint_q0
        np.testing.assert_allclose(new_params["q0"]["frequencies"], expected_q0)
        assert (
            pytest.approx(midpoint_q0)
            == (
                max(new_params["q0"]["frequencies"])
                + min(new_params["q0"]["frequencies"])
            )
            / 2
        )

        new_range_q0 = max(new_params["q0"]["frequencies"]) - min(
            new_params["q0"]["frequencies"]
        )
        assert pytest.approx(new_range_q0) == (max(freqs_q0) - min(freqs_q0)) * 1.3

        # q1: range 200 MHz, multiplier 1.2
        midpoint_q1 = (max(freqs_q1) + min(freqs_q1)) / 2
        expected_q1 = (freqs_q1 - midpoint_q1) * 1.2 + midpoint_q1
        np.testing.assert_allclose(new_params["q1"]["frequencies"], expected_q1)

        assert (
            pytest.approx(midpoint_q1)
            == (
                max(new_params["q1"]["frequencies"])
                + min(new_params["q1"]["frequencies"])
            )
            / 2
        )

        new_range_q1 = max(new_params["q1"]["frequencies"]) - min(
            new_params["q1"]["frequencies"]
        )
        assert pytest.approx(new_range_q1) == (max(freqs_q1) - min(freqs_q1)) * 1.2
