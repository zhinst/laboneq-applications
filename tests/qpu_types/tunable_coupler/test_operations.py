# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.qpu_types.tunable_coupler.operations."""

from contextlib import nullcontext

import numpy as np
import pytest
from laboneq.dsl.experiment.build_experiment import build
from laboneq.simple import SectionAlignment, Session, SweepParameter, dsl

from laboneq_applications.qpu_types.tunable_transmon import TunableTransmonQubit

import tests.helpers.dsl as tsl


@pytest.fixture
def qops(four_tunable_transmon_cz_platform):
    return four_tunable_transmon_cz_platform.qpu.quantum_operations


@pytest.fixture
def qubits(four_tunable_transmon_cz_platform):
    return [
        q
        for q in four_tunable_transmon_cz_platform.qpu.quantum_elements
        if isinstance(q, TunableTransmonQubit)
    ]


class TestTunableCouplerOperations:
    def check_op_builds_and_compiles(self, section, platform, sweep=None):
        """Check that an operation can be built and compiled successfully."""
        if sweep is not None:
            maybe_sweep = dsl.sweep(uid="sweep", parameter=sweep)
        else:
            maybe_sweep = nullcontext()

        def exp_with_section(qubits):
            with dsl.acquire_loop_rt(count=1):
                with maybe_sweep:
                    dsl.add(section)

        exp = build(exp_with_section, platform.qpu.quantum_elements)

        session = Session(platform.setup)
        session.connect(do_emulation=True)
        session.compile(exp)

    def reserve_ops(self, q):
        """Return the expected reserve operations for the given qubit.

        Note: the four-qubit CZ platform device setup does not configure a
        ``drive_ef`` signal, so only four signals are reserved per qubit.
        """
        return [
            tsl.reserve_op(signal=f"{q.uid}/measure"),
            tsl.reserve_op(signal=f"{q.uid}/acquire"),
            tsl.reserve_op(signal=f"{q.uid}/drive"),
            tsl.reserve_op(signal=f"{q.uid}/flux"),
        ]

    def test_cz_default(self, qops, qubits, four_tunable_transmon_cz_platform):
        q0, q1 = qubits[:2]

        section = qops.cz(q0, q1)

        assert section == tsl.section(
            uid="__cz_q0_q1_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            self.reserve_ops(q1),
            tsl.play_pulse_op(
                signal="c_q0q1/flux",
                amplitude=0.0,
                length=1e-6,
                pulse=tsl.pulse(
                    function="modulated_flux",
                    amplitude=1.0,
                    length=100e-9,
                    pulse_parameters={"frequency": 225e6},
                ),
            ),
            tsl.play_pulse_op(
                signal="q0/drive",
                amplitude=None,
                length=None,
                increment_oscillator_phase=0.0,
                phase=None,
                pulse_parameters=None,
                pulse=None,
            ),
            tsl.play_pulse_op(
                signal="q1/drive",
                amplitude=None,
                length=None,
                increment_oscillator_phase=0.0,
                phase=None,
                pulse_parameters=None,
                pulse=None,
            ),
        )

        self.check_op_builds_and_compiles(section, four_tunable_transmon_cz_platform)

    @pytest.mark.parametrize(
        "tc_amplitude",
        [
            pytest.param(0.8, id="constant"),
            pytest.param(
                SweepParameter(uid="sweep_amp", values=np.linspace(0, 1, 5)),
                id="sweep_parameter",
            ),
        ],
    )
    def test_cz_tc_amplitude(
        self, tc_amplitude, qops, qubits, four_tunable_transmon_cz_platform
    ):
        q0, q1 = qubits[:2]

        section = qops.cz(q0, q1, tc_amplitude=tc_amplitude)

        assert section == tsl.section(uid="__cz_q0_q1_0").children(
            self.reserve_ops(q0),
            self.reserve_ops(q1),
            tsl.play_pulse_op(signal="c_q0q1/flux", amplitude=tc_amplitude),
            tsl.play_pulse_op(signal="q0/drive", increment_oscillator_phase=0.0),
            tsl.play_pulse_op(signal="q1/drive", increment_oscillator_phase=0.0),
        )

        sweep = tc_amplitude if isinstance(tc_amplitude, SweepParameter) else None
        self.check_op_builds_and_compiles(
            section, four_tunable_transmon_cz_platform, sweep=sweep
        )

    def test_cz_reads_parameters_from_edge(
        self, qops, qubits, four_tunable_transmon_cz_platform
    ):
        """Non-default CzParameters on the edge are picked up by the cz operation."""
        q0, q1 = qubits[:2]

        edge = four_tunable_transmon_cz_platform.qpu.topology["cz", q0.uid, q1.uid]
        edge.parameters.coupler_pulse["amplitude"] = 0.6
        edge.parameters.coupler_pulse["length"] = 200e-9
        edge.parameters.coupler_pulse["frequency"] = 350e6
        edge.parameters.control_angle = 0.3
        edge.parameters.target_angle = 0.7

        section = qops.cz(q0, q1)

        assert section == tsl.section(uid="__cz_q0_q1_0").children(
            self.reserve_ops(q0),
            self.reserve_ops(q1),
            tsl.play_pulse_op(
                signal="c_q0q1/flux",
                amplitude=0.6,
                length=200e-9,
                pulse=tsl.pulse(
                    function="modulated_flux",
                    pulse_parameters={"frequency": 350e6},
                ),
            ),
            tsl.play_pulse_op(signal="q0/drive", increment_oscillator_phase=0.3),
            tsl.play_pulse_op(signal="q1/drive", increment_oscillator_phase=0.7),
        )

        self.check_op_builds_and_compiles(section, four_tunable_transmon_cz_platform)
