# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from laboneq.simple import Session

from laboneq_applications.experiments import (
    amplitude_fine,
    amplitude_rabi,
    dispersive_shift,
    drag_q_scaling,
    echo,
    iq_blobs,
    lifetime_measurement,
    qubit_spectroscopy,
    qubit_spectroscopy_amplitude,
    ramsey,
    resonator_spectroscopy,
    resonator_spectroscopy_amplitude,
    time_traces,
)
from laboneq_applications.qpu_types.tunable_transmon import demo_platform


@pytest.fixture
def workflow_platform():
    qt_platform = demo_platform(n_qubits=2)
    setup = qt_platform.setup

    qpu = qt_platform.qpu
    qubits = qpu.quantum_elements

    session = Session(setup)
    session.connect(do_emulation=True)

    return session, qpu, qubits


class TestTemporaryParameters:
    def test_amplitude_fine(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = amplitude_fine.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            amplification_qop="x180",
            target_angle=1.0,
            phase_offset=0.0,
            repetitions=[[1, 2, 3, 4], [1, 2, 3, 4]],
        ).run(until="create_experiment")

        result_modified = amplitude_fine.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
                "q1": {"readout_lo_frequency": 8e9},
            },  # pass temporary parameters
            amplification_qop="x180",
            target_angle=1.0,
            phase_offset=0.0,
            repetitions=[[1, 2, 3, 4], [1, 2, 3, 4]],
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )
        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 7e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 8e9
        )

    def test_amplitude_rabi(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = amplitude_rabi.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            amplitudes=[np.linspace(0, 1, 11), np.linspace(0, 0.75, 11)],
        ).run(until="create_experiment")

        result_modified = amplitude_rabi.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
                "q1": {"readout_lo_frequency": 8e9},
            },  # pass temporary parameters
            amplitudes=[np.linspace(0, 1, 11), np.linspace(0, 0.75, 11)],
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )
        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 7e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 8e9
        )

    def test_dispersive_shift(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = dispersive_shift.experiment_workflow(
            session=session,
            qpu=qpu,
            qubit=qubits[0],
            frequencies=np.linspace(1.8e9, 2.2e9, 101),
            states="ge",
        ).run(until="create_experiment")

        result_modified = dispersive_shift.experiment_workflow(
            session=session,
            qpu=qpu,
            qubit=qubits[0],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
            },  # pass temporary parameters
            frequencies=np.linspace(1.8e9, 2.2e9, 101),
            states="ge",
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )

    def test_drag_q_scaling(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = drag_q_scaling.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            q_scalings=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.04, 0.04, 11),
            ],
        ).run(until="create_experiment")

        result_modified = drag_q_scaling.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
                "q1": {"readout_lo_frequency": 8e9},
            },  # pass temporary parameters
            q_scalings=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.04, 0.04, 11),
            ],
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )
        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 7e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 8e9
        )

    def test_echo(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = echo.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            delays=[np.linspace(0, 30e-6, 51), np.linspace(0, 30e-6, 51)],
        ).run(until="create_experiment")

        result_modified = echo.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
                "q1": {"readout_lo_frequency": 8e9},
            },  # pass temporary parameters
            delays=[np.linspace(0, 30e-6, 51), np.linspace(0, 30e-6, 51)],
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )
        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 7e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 8e9
        )

    def test_iq_blobs(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = iq_blobs.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            states="ge",
        ).run(until="create_experiment")

        result_modified = iq_blobs.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
                "q1": {"readout_lo_frequency": 8e9},
            },  # pass temporary parameters
            states="ge",
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )
        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 7e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 8e9
        )

    def test_lifetime_measurement(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = lifetime_measurement.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            delays=[[10e-9, 50e-9, 1], [10e-9, 50e-9, 1]],
        ).run(until="create_experiment")

        result_modified = lifetime_measurement.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
                "q1": {"readout_lo_frequency": 8e9},
            },  # pass temporary parameters
            delays=[[10e-9, 50e-9, 1], [10e-9, 50e-9, 1]],
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )
        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 7e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 8e9
        )

    def test_qubit_spectroscopy(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = qubit_spectroscopy.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            frequencies=[
                np.linspace(6.0e9, 6.3e9, 101),
                np.linspace(5.8e9, 6.2e9, 101),
            ],
        ).run(until="create_experiment")

        result_modified = qubit_spectroscopy.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
                "q1": {"readout_lo_frequency": 8e9},
            },  # pass temporary parameters
            frequencies=[
                np.linspace(6.0e9, 6.3e9, 101),
                np.linspace(5.8e9, 6.2e9, 101),
            ],
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )
        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 7e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 8e9
        )

    def test_qubit_spectroscopy_amplitude(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = qubit_spectroscopy_amplitude.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            frequencies=[
                np.linspace(5.8e9, 6.2e9, 101),
                np.linspace(0.8e9, 1.2e9, 101),
            ],
            amplitudes=[[0.1, 0.5, 1], [0.1, 0.5, 1]],
        ).run(until="create_experiment")

        result_modified = qubit_spectroscopy_amplitude.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
                "q1": {"readout_lo_frequency": 8e9},
            },  # pass temporary parameters
            frequencies=[
                np.linspace(5.8e9, 6.2e9, 101),
                np.linspace(0.8e9, 1.2e9, 101),
            ],
            amplitudes=[[0.1, 0.5, 1], [0.1, 0.5, 1]],
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )
        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 7e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 8e9
        )

    def test_ramsey(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = ramsey.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            delays=[np.linspace(0, 20e-6, 51), np.linspace(0, 20e-6, 51)],
            detunings=[0.67e6, 0.67e6],
        ).run(until="create_experiment")

        result_modified = ramsey.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=[qubits[0], qubits[1]],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
                "q1": {"readout_lo_frequency": 8e9},
            },  # pass temporary parameters
            delays=[np.linspace(0, 10e-6, 51), np.linspace(0, 10e-6, 51)],
            detunings=[1e6, 1e6],
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )
        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 7e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q1/measure"]
            .calibration.local_oscillator.frequency
            == 8e9
        )

    def test_resonator_spectroscopy(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = resonator_spectroscopy.experiment_workflow(
            session=session,
            qpu=qpu,
            qubit=qubits[0],
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
        ).run(until="create_experiment")

        result_modified = resonator_spectroscopy.experiment_workflow(
            session=session,
            qpu=qpu,
            qubit=qubits[0],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
            },  # pass temporary parameters
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )

    def test_resonator_spectroscopy_amplitude(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = resonator_spectroscopy_amplitude.experiment_workflow(
            session=session,
            qpu=qpu,
            qubit=qubits[0],
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
            amplitudes=np.linspace(0.1, 1, 10),
        ).run(until="create_experiment")

        result_modified = resonator_spectroscopy_amplitude.experiment_workflow(
            session=session,
            qpu=qpu,
            qubit=qubits[0],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
            },  # pass temporary parameters
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
            amplitudes=np.linspace(0.1, 1, 10),
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )

    def test_time_traces(self, workflow_platform):
        session, qpu, qubits = workflow_platform

        options = time_traces.experiment_workflow.options()
        options.count(2)
        # TODO: fix tests to work with do_analysis=True when the new options feature is
        #  in
        options.do_analysis(False)

        temporary_parameters_q0 = qubits[0].parameters.copy()
        temporary_parameters_q0.drive_lo_frequency = 1e9

        result_unmodified = time_traces.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubits[0],
            states="gef",
            options=options,
        ).run(until="create_experiment")

        result_modified = time_traces.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubits[0],
            temporary_parameters={
                qubits[0].uid: temporary_parameters_q0,
            },  # pass temporary parameters
            states="gef",
            options=options,
        ).run(until="create_experiment")

        assert (
            result_unmodified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 6.4e9
        )
        assert (
            result_modified.tasks["create_experiment"]
            .output.signals["q0/drive"]
            .calibration.local_oscillator.frequency
            == 1e9
        )
