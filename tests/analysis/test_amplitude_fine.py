import numpy as np
import pytest
from laboneq.workflow.tasks import handles
from laboneq.workflow.tasks.run_experiment import (
    AcquiredResult,
    RunExperimentResults,
)

from laboneq_applications.analysis import amplitude_fine


@pytest.fixture()
def results_single_qubit():
    """Results from an amplitude-fine experiment."""
    data = {}
    data[handles.result_handle("q0")] = AcquiredResult(
        data=np.array(
            [
                0.04101562 + 0.0j,
                0.04101562 + 0.00097656j,
                0.04101562 + 0.00195312j,
                0.04101562 + 0.00292969j,
                0.04101562 + 0.00390625j,
                0.04101562 + 0.00488281j,
                0.04101562 + 0.00585938j,
                0.04101562 + 0.00683594j,
                0.04101562 + 0.0078125j,
                0.04101562 + 0.00878906j,
                0.04101562 + 0.00976562j,
                0.04101562 + 0.01074219j,
                0.04101562 + 0.01171875j,
                0.04101562 + 0.01269531j,
                0.04101562 + 0.01367188j,
                0.04101562 + 0.01464844j,
                0.04101562 + 0.015625j,
                0.04101562 + 0.01660156j,
                0.04101562 + 0.01757812j,
                0.04101562 + 0.01855469j,
                0.04101562 + 0.01953125j,
            ]
        ),
        axis_name=["Repetitions"],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=(0.041015625 + 0.0205078125j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=(0.041015625 + 0.021484375j),
        axis_name=[],
        axis=[],
    )

    sweep_points = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        ]
    )
    return RunExperimentResults(data=data), sweep_points


class TestAmplitudeFineAnalysisSingleQubit:
    def test_create_and_run(
        self, single_tunable_transmon_platform, results_single_qubit
    ):
        [q0] = single_tunable_transmon_platform.qpu.qubits
        options = amplitude_fine.analysis_workflow.options()

        result = amplitude_fine.analysis_workflow(
            result=results_single_qubit[0],
            qubits=q0,
            amplification_qop="x180",
            repetitions=results_single_qubit[1],
            target_angle=np.pi,
            phase_offset=-np.pi / 2,
            options=options,
        ).run()

        assert len(result.tasks) == 7

        task_names = [t.name for t in result.tasks]
        assert "calculate_qubit_population" in task_names
        assert "fit_data" in task_names
        assert "process_fit_results" in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_raw_complex_data_1d" in task_names
        assert "plot_population" in task_names
        for task in result.tasks:
            if task.name in ["calculate_qubit_population", "fit_data"]:
                assert "q0" in task.output
        qb_params = result.tasks["extract_qubit_parameters"].output
        assert "new_parameter_values" in qb_params
        assert "old_parameter_values" in qb_params
        assert len(qb_params["new_parameter_values"]) == 1


@pytest.fixture()
def results_two_qubit():
    """Results from an amplitude-fine experiment."""
    data = {}
    data[handles.result_handle("q0")] = AcquiredResult(
        data=np.array(
            [
                0.04101562 + 0.0j,
                0.04101562 + 0.00097656j,
                0.04101562 + 0.00195312j,
                0.04101562 + 0.00292969j,
                0.04101562 + 0.00390625j,
                0.04101562 + 0.00488281j,
                0.04101562 + 0.00585938j,
                0.04101562 + 0.00683594j,
                0.04101562 + 0.0078125j,
                0.04101562 + 0.00878906j,
                0.04101562 + 0.00976562j,
                0.04101562 + 0.01074219j,
                0.04101562 + 0.01171875j,
                0.04101562 + 0.01269531j,
                0.04101562 + 0.01367188j,
                0.04101562 + 0.01464844j,
                0.04101562 + 0.015625j,
                0.04101562 + 0.01660156j,
                0.04101562 + 0.01757812j,
                0.04101562 + 0.01855469j,
                0.04101562 + 0.01953125j,
            ]
        ),
    )
    data[handles.result_handle("q1")] = AcquiredResult(
        np.array(
            [
                0.04199219 + 0.0j,
                0.04199219 + 0.00097656j,
                0.04199219 + 0.00195312j,
                0.04199219 + 0.00292969j,
                0.04199219 + 0.00390625j,
                0.04199219 + 0.00488281j,
                0.04199219 + 0.00585938j,
                0.04199219 + 0.00683594j,
                0.04199219 + 0.0078125j,
                0.04199219 + 0.00878906j,
                0.04199219 + 0.00976562j,
                0.04199219 + 0.01074219j,
                0.04199219 + 0.01171875j,
                0.04199219 + 0.01269531j,
                0.04199219 + 0.01367188j,
                0.04199219 + 0.01464844j,
                0.04199219 + 0.015625j,
                0.04199219 + 0.01660156j,
                0.04199219 + 0.01757812j,
                0.04199219 + 0.01855469j,
                0.04199219 + 0.01953125j,
            ]
        ),
    )
    data[handles.calibration_trace_handle("q0", "g")] = AcquiredResult(
        data=(0.041015625 + 0.0205078125j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q0", "e")] = AcquiredResult(
        data=(0.041015625 + 0.021484375j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q1", "g")] = AcquiredResult(
        data=(0.0419921875 + 0.0205078125j),
        axis_name=[],
        axis=[],
    )
    data[handles.calibration_trace_handle("q1", "e")] = AcquiredResult(
        data=(0.0419921875 + 0.021484375j),
        axis_name=[],
        axis=[],
    )
    sweep_points = [
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ]
        ),
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ]
        ),
    ]
    return RunExperimentResults(data=data), sweep_points


class TestAmplitudeFineAnalysisTwoQubit:
    def test_create_and_run(self, two_tunable_transmon_platform, results_two_qubit):
        qubits = two_tunable_transmon_platform.qpu.qubits
        options = amplitude_fine.analysis_workflow.options()

        result = amplitude_fine.analysis_workflow(
            result=results_two_qubit[0],
            qubits=qubits,
            amplification_qop="x180",
            repetitions=results_two_qubit[1],
            target_angle=np.pi,
            phase_offset=-np.pi / 2,
            options=options,
        ).run()

        assert len(result.tasks) == 7

        task_names = [t.name for t in result.tasks]
        assert "calculate_qubit_population" in task_names
        assert "fit_data" in task_names
        assert "process_fit_results" in task_names
        assert "extract_qubit_parameters" in task_names
        assert "plot_raw_complex_data_1d" in task_names
        assert "plot_population" in task_names
        for task in result.tasks:
            if task.name in ["calculate_qubit_population", "fit_data"]:
                assert "q0" in task.output
                assert "q1" in task.output
        qb_params = result.tasks["extract_qubit_parameters"].output
        assert "new_parameter_values" in qb_params
        assert "old_parameter_values" in qb_params
        assert len(qb_params["new_parameter_values"]) == 2
