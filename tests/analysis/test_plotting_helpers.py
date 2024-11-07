# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from laboneq.workflow.tasks import handles
from laboneq.workflow.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)

from laboneq_applications.analysis import plotting_helpers as plt_hlp
from laboneq_applications.experiments.options import TuneupAnalysisOptions


@pytest.fixture
def result():
    """Results from an AmplitudeRabi experiment."""
    data = {}
    data[handles.result_handle("q0")] = AcquiredResult(
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
        )
    )
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=(0.05745863888207082 - 0.13026141779382786j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=(0.1770431406621688 + 0.91612948998106j),
        axis_name=[],
        axis=[],
    )

    sweep_points = np.array(
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
    return RunExperimentResults(data=data), sweep_points


@pytest.fixture
def result_nested_two_qubits():
    """Results from an AmplitudeRabi experiment.

    The same data and sweep points are used for both qubits.
    """
    data = {}
    data[handles.result_handle("q0", suffix="nest")] = AcquiredResult(
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
        )
    )
    data[handles.result_handle("q1", suffix="nest")] = data[
        handles.result_handle("q0", suffix="nest")
    ]

    sweep_points = np.array(
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
    return RunExperimentResults(data=data), [sweep_points, sweep_points]


class TestRawPlotting:
    def test_run(self, single_tunable_transmon_platform, result):
        [q0] = single_tunable_transmon_platform.qpu.qubits

        # plot_raw_complex_data_1d contains is a task that contains a call to
        # save_artifact if options.save_figures == True, and save_artifacts
        # can only be run inside a workflow
        options = TuneupAnalysisOptions()
        options.save_figures = False
        figures = plt_hlp.plot_raw_complex_data_1d(q0, *result, "xlabel", 1, options)

        assert len(figures) == 1
        assert "q0" in figures

    def test_run_nested_two_qubit(
        self, two_tunable_transmon_platform, result_nested_two_qubits
    ):
        qubits = two_tunable_transmon_platform.qpu.qubits

        # plot_raw_complex_data_1d contains is a task that contains a call to
        # save_artifact if options.save_figures == True, and save_artifacts
        # can only be run inside a workflow
        options = TuneupAnalysisOptions()
        options.save_figures = False
        figures = plt_hlp.plot_raw_complex_data_1d(
            qubits, *result_nested_two_qubits, "xlabel", 1, options
        )

        assert len(figures) == 2
        assert "q0" in figures
        assert "q1" in figures
