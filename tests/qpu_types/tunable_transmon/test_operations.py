"""Tests for laboneq_applications.qpu_types.tunable_transmon.operations."""

import copy
from contextlib import nullcontext

import numpy as np
import pytest
from laboneq.simple import SectionAlignment, Session, SweepParameter

import tests.helpers.dsl as tsl
from laboneq_applications import dsl
from laboneq_applications.core.build_experiment import build
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonOperations,
    TunableTransmonQubit,
)


@pytest.fixture()
def qops():
    """Return TunableTransmonQubit operations."""
    return TunableTransmonOperations()


@pytest.fixture()
def q0(single_tunable_transmon):
    return single_tunable_transmon.qubits[0]


class TestTunableTransmonOperations:
    def check_op_builds_and_compiles(self, section, device, sweep=None):
        """Check that an operation can be built and compiled successfully."""

        if sweep is not None:
            maybe_sweep = dsl.sweep(uid="sweep", parameter=sweep)
        else:
            maybe_sweep = nullcontext()

        def exp_with_section(qubits):
            with dsl.acquire_loop_rt(count=1):
                with maybe_sweep:
                    dsl.add(section)

        exp = build(exp_with_section, device.qubits)

        session = Session(device.setup)
        session.connect(do_emulation=True)
        session.compile(exp)

    def check_exp_compiles(self, exp, device):
        session = Session(device.setup)
        session.connect(do_emulation=True)
        session.compile(exp)

    def reserve_ops(self, q):
        """Return the expected reserve operations for the given qubit."""
        return [
            tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/drive"),
            tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/drive_ef"),
            tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/measure"),
            tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/acquire"),
            tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/flux"),
        ]

    def test_create(self):
        qops = TunableTransmonOperations()

        assert qops.QUBIT_TYPES is TunableTransmonQubit
        assert qops.QUBIT_TYPES.TRANSITIONS == ("ge", "ef")

    def test_operation_docstring(self):
        qops = TunableTransmonOperations()

        assert qops.x180.__doc__.startswith(
            "Rotate the qubit by 180 degrees about the X axis.",
        )

    def test_barrier(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.barrier(q0)

        assert section == tsl.section(
            uid="__barrier_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_delay(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.delay(q0, 1e-6)

        assert section == tsl.section(
            uid="__delay_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.delay_op(
                signal="/logical_signal_groups/q0/drive",
                time=1e-06,
                precompensation_clear=None,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    @pytest.mark.parametrize(
        ("rf", "freq", "oscillator_freq"),
        [
            pytest.param(True, 1.6e9, 0.1e9, id="rf-positive"),
            pytest.param(True, 1.4e9, -0.1e9, id="rf-negative"),
            pytest.param(False, 0.1e9, 0.1e9, id="oscillator-positive"),
            pytest.param(False, -0.1e9, -0.1e9, id="oscillator-negative"),
        ],
    )
    def test_set_frequency(
        self,
        qops,
        rf,
        freq,
        oscillator_freq,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits

        @dsl.qubit_experiment
        def exp_set_freq(q):
            # set_frequency requires an experiment context to access the calibration
            with dsl.acquire_loop_rt(count=10):
                qops.set_frequency(q, freq, rf=rf)
                qops.x90(q)

        exp = exp_set_freq(q0)

        calibration = exp.get_calibration()
        signal_calibration = calibration[q0.signals["drive"]]
        assert signal_calibration.oscillator.frequency == oscillator_freq

        self.check_exp_compiles(exp, single_tunable_transmon)

    def test_set_frequency_twice(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits

        @dsl.qubit_experiment
        def exp_set_freq(q):
            # set_frequency requires an experiment context to access the calibration
            with dsl.acquire_loop_rt(count=10):
                qops.set_frequency(q, 1.5e9)
                qops.set_frequency(q, 1.7e9)

        with pytest.raises(RuntimeError) as err:
            exp_set_freq(q0)

        assert str(err.value) == (
            "Frequency of qubit q0 drive line was set multiple times"
            " using the set_frequency operation."
        )

    def test_set_frequency_transition(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits

        @dsl.qubit_experiment
        def exp_set_freq(q):
            # set_frequency requires an experiment context to access the calibration
            with dsl.acquire_loop_rt(count=10):
                qops.set_frequency(q, 1.8e9, transition="ef")
                qops.x90(q, transition="ef")

        exp = exp_set_freq(q0)

        calibration = exp.get_calibration()
        signal_calibration = calibration[q0.signals["drive_ef"]]
        assert signal_calibration.oscillator.frequency == 0.3e9

        self.check_exp_compiles(exp, single_tunable_transmon)

    @pytest.mark.parametrize(
        ("rf", "freq", "oscillator_freq"),
        [
            pytest.param(True, 2.1e9, 0.1e9, id="rf"),
            pytest.param(False, 1.5e9, 1.5e9, id="oscillator"),
        ],
    )
    def test_set_frequency_readout(
        self,
        qops,
        rf,
        freq,
        oscillator_freq,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits

        @dsl.qubit_experiment
        def exp_set_freq(q):
            # set_frequency requires an experiment context to access the calibration
            with dsl.acquire_loop_rt(count=10):
                qops.set_frequency(q, freq, readout=True, rf=rf)
                qops.measure(q, handle="measure")

        exp = exp_set_freq(q0)

        calibration = exp.get_calibration()
        signal_calibration = calibration[q0.signals["measure"]]
        assert signal_calibration.oscillator.frequency == oscillator_freq

        self.check_exp_compiles(exp, single_tunable_transmon)

    @pytest.mark.parametrize(
        ("rf", "freqs", "oscillator_freqs"),
        [
            pytest.param(
                True,
                [1.5e9, 1.6e9, 1.7e9],
                [0.0e9, 0.1e9, 0.2e9],
                id="rf-positive",
            ),
            pytest.param(
                True,
                [1.6e9, 1.4e9, 1.2e9],
                [0.1e9, -0.1e9, -0.3e9],
                id="rf-negative",
            ),
            pytest.param(
                False,
                [1.5e9, 1.6e9, 1.7e9],
                [1.5e9, 1.6e9, 1.7e9],
                id="oscillator-positive",
            ),
            pytest.param(
                False,
                [-0.1e9, -0.2e9, -0.3e9],
                [-0.1e9, -0.2e9, -0.3e9],
                id="oscillator-negative",
            ),
        ],
    )
    def test_set_frequency_sweep(
        self,
        qops,
        rf,
        freqs,
        oscillator_freqs,
        single_tunable_transmon,
    ):
        # TODO: Why do the rf=True cases not fail here?
        [q0] = single_tunable_transmon.qubits

        @dsl.qubit_experiment
        def exp_set_freq(q, frequencies):
            # set_frequency requires an experiment context to access the calibration
            with dsl.acquire_loop_rt(count=10):
                with dsl.sweep(
                    parameter=SweepParameter(values=frequencies),
                ) as frequency:
                    qops.set_frequency(q, frequency, rf=rf)
                    qops.x90(q)

        exp = exp_set_freq(q0, freqs)

        calibration = exp.get_calibration()
        signal_calibration = calibration[q0.signals["drive"]]
        frequency = signal_calibration.oscillator.frequency
        assert isinstance(frequency, SweepParameter)
        np.testing.assert_equal(frequency.values, oscillator_freqs)

        self.check_exp_compiles(exp, single_tunable_transmon)

    @pytest.mark.parametrize(
        ("rf", "freqs", "oscillator_freqs"),
        [
            pytest.param(
                True,
                [2.0e9, 2.1e9, 2.2e9],
                [0.0e9, 0.1e9, 0.2e9],
                id="rf-positive",
            ),
            pytest.param(
                True,
                [2.1e9, 1.9e9, 2.1e9],
                [0.1e9, -0.1e9, 0.1e9],
                id="rf-negative",
            ),
            pytest.param(
                False,
                [1.5e9, 1.6e9, 1.7e9],
                [1.5e9, 1.6e9, 1.7e9],
                id="oscillator-positive",
            ),
            pytest.param(
                False,
                [-0.1e9, -0.2e9, -0.3e9],
                [-0.1e9, -0.2e9, -0.3e9],
                id="oscillator-positive",
            ),
        ],
    )
    def test_set_frequency_readout_sweep(
        self,
        qops,
        rf,
        freqs,
        oscillator_freqs,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits

        @dsl.qubit_experiment
        def exp_set_freq(q):
            # set_frequency requires an experiment context to access the calibration
            with dsl.acquire_loop_rt(count=10):
                with dsl.sweep(
                    parameter=SweepParameter(values=freqs),
                ) as frequency:
                    qops.set_frequency(q, frequency, readout=True, rf=rf)
                    qops.measure(q, handle="measure")

        exp = exp_set_freq(q0)

        calibration = exp.get_calibration()
        signal_calibration = calibration[q0.signals["measure"]]
        frequency = signal_calibration.oscillator.frequency
        assert isinstance(frequency, SweepParameter)
        np.testing.assert_equal(frequency.values, oscillator_freqs)

        self.check_exp_compiles(exp, single_tunable_transmon)

    def test_measure(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.measure(q0, "result")

        assert section == tsl.section(
            uid="__measure_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/measure",
                amplitude=1.0,
                length=2e-6,
                increment_oscillator_phase=None,
                phase=None,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    uid="__readout_pulse_0",
                    function="const",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters=None,
                ),
            ),
            tsl.acquire_op(
                signal="/logical_signal_groups/q0/acquire",
                handle="result",
                kernel=[
                    tsl.pulse(
                        function="const",
                        uid="__integration_kernel_q0_0",
                        amplitude=1.0,
                        length=2e-6,
                        pulse_parameters=None,
                    ),
                ],
                length=2e-6,
                pulse_parameters=None,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_measure_with_readout_pulse(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.measure(q0, "result", readout_pulse={"amplitude": 0.5})

        assert section == tsl.section(
            uid="__measure_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/measure",
                amplitude=1.0,
                length=2e-6,
                increment_oscillator_phase=None,
                phase=None,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    function="const",
                    amplitude=0.5,
                    length=1e-7,
                    pulse_parameters=None,
                ),
            ),
            tsl.acquire_op(),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_measure_with_kernel_pulses(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        kernel_pulses = [
            {"function": "const", "amplitude": 0.5},
            {"function": "const", "amplitude": 0.6},
        ]
        section = qops.measure(q0, "result", kernel_pulses=kernel_pulses)

        assert section == tsl.section(
            uid="__measure_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(),
            tsl.acquire_op(
                signal="/logical_signal_groups/q0/acquire",
                handle="result",
                kernel=[
                    tsl.pulse(
                        function="const",
                        uid="__integration_kernel_q0_0",
                        amplitude=0.5,
                        pulse_parameters=None,
                    ),
                    tsl.pulse(
                        function="const",
                        uid="__integration_kernel_q0_1",
                        amplitude=0.6,
                        pulse_parameters=None,
                    ),
                ],
                length=2e-6,
                pulse_parameters=None,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_measure_twice(self, qops, single_tunable_transmon):
        # if the integration kernels are created twice, this will fail to compile.
        [q0] = single_tunable_transmon.qubits
        with dsl.section(name="measure_twice") as section:
            qops.measure(q0, "result_1")
            qops.measure(q0, "result_2")

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_acquire(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.acquire(q0, "result")

        assert section == tsl.section(
            uid="__acquire_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.acquire_op(
                signal="/logical_signal_groups/q0/acquire",
                handle="result",
                kernel=[
                    tsl.pulse(
                        function="const",
                        uid="__integration_kernel_q0_0",
                        amplitude=1.0,
                        length=2e-6,
                        pulse_parameters=None,
                    ),
                ],
                length=2e-6,
                pulse_parameters=None,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_acquire_with_kernel_pulses(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        kernel_pulses = [
            {"function": "const", "amplitude": 0.5},
            {"function": "const", "amplitude": 0.6},
        ]
        section = qops.acquire(q0, "result", kernel_pulses=kernel_pulses)

        assert section == tsl.section(
            uid="__acquire_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.acquire_op(
                signal="/logical_signal_groups/q0/acquire",
                handle="result",
                kernel=[
                    tsl.pulse(
                        function="const",
                        uid="__integration_kernel_q0_0",
                        amplitude=0.5,
                        pulse_parameters=None,
                    ),
                    tsl.pulse(
                        function="const",
                        uid="__integration_kernel_q0_1",
                        amplitude=0.6,
                        pulse_parameters=None,
                    ),
                ],
                length=2e-6,
                pulse_parameters=None,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_acquire_twice(self, qops, single_tunable_transmon):
        # if the integration kernels are created twice, this will fail to compile.
        [q0] = single_tunable_transmon.qubits
        with dsl.section(name="measure_twice") as section:
            qops.acquire(q0, "result_1")
            qops.acquire(q0, "result_2")

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_prepare_state_g(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.prepare_state(q0, "g")

        assert section == tsl.section(
            uid="__prepare_state_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_prepare_state_e(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.prepare_state(q0, "e")

        assert section == tsl.section(
            uid="__prepare_state_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.section(uid="__x180_q0_0").children(
                self.reserve_ops(q0),
                tsl.play_pulse_op(
                    signal="/logical_signal_groups/q0/drive",
                    amplitude=0.8,
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_prepare_state_f(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.prepare_state(q0, "f")

        assert section == tsl.section(
            uid="__prepare_state_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.section(uid="__x180_q0_0").children(
                self.reserve_ops(q0),
                tsl.play_pulse_op(
                    signal="/logical_signal_groups/q0/drive",
                    amplitude=0.8,
                ),
            ),
            tsl.section(uid="__x180_q0_1").children(
                self.reserve_ops(q0),
                tsl.play_pulse_op(
                    signal="/logical_signal_groups/q0/drive_ef",
                    amplitude=0.7,
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_prepare_state_passive_reset(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.prepare_state(q0, "g", reset="passive")

        assert section == tsl.section(
            uid="__prepare_state_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.section(uid="__passive_reset_q0_0").children(
                self.reserve_ops(q0),
                tsl.delay_op(
                    signal="/logical_signal_groups/q0/drive",
                    time=1e-06,
                    precompensation_clear=None,
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_prepare_state_active_reset(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        with pytest.raises(ValueError) as err:
            qops.prepare_state(q0, "g", reset="active")

        assert str(err.value) == "The active reset operation is not yet implemented."

    def test_prepare_state_invalid_reset(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        with pytest.raises(ValueError) as err:
            qops.prepare_state(q0, "g", reset="moo")

        assert str(err.value) == (
            "The reset parameter to prepare_state must be 'active', 'passive',"
            " or None, not: 'moo'"
        )

    def test_prepare_state_invalid(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        with pytest.raises(ValueError) as err:
            qops.prepare_state(q0, "z")
        assert str(err.value) == "Only states g, e and f can be prepared, not 'z'"

    def test_passive_reset(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.passive_reset(q0)

        assert section == tsl.section(
            uid="__passive_reset_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.delay_op(
                signal="/logical_signal_groups/q0/drive",
                time=1e-06,
                precompensation_clear=None,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    @pytest.mark.parametrize(
        ("delay"),
        [
            pytest.param(2e-5, id="constant"),
            pytest.param(
                SweepParameter(uid="sweep", values=[2e-5, 3e-5, 4e-5]),
                id="sweep_parameter",
            ),
        ],
    )
    def test_passive_reset_delay(self, delay, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.passive_reset(q0, delay=delay)

        assert section == tsl.section(
            uid="__passive_reset_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.delay_op(
                signal="/logical_signal_groups/q0/drive",
                time=delay,
                precompensation_clear=None,
            ),
        )

        sweep = delay if isinstance(delay, SweepParameter) else None
        self.check_op_builds_and_compiles(section, single_tunable_transmon, sweep=sweep)

    @pytest.mark.parametrize(
        ("angle", "expected_amplitude"),
        [
            pytest.param(np.pi, 0.8, id="pi"),
            pytest.param(np.pi / 2, 0.4, id="pi_by_2"),
            pytest.param(np.pi / 3, 0.8 / 3, id="pi_by_3"),
        ],
    )
    def test_rx(self, angle, expected_amplitude, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.rx(q0, angle)

        assert section == tsl.section(
            uid="__rx_q0_0",
            alignment=SectionAlignment.LEFT,
            on_system_grid=False,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=expected_amplitude,
                length=51e-9,
                increment_oscillator_phase=None,
                phase=0.0,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    uid="__rx_pulse_0",
                    function="drag",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters={"beta": 0.01, "sigma": 0.21},
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    @pytest.mark.parametrize("pi2", [1, 2])
    def test_interpolation(self, pi2, qops, single_tunable_transmon):
        def amp(section):
            return section.children[-1].amplitude

        [q0] = single_tunable_transmon.qubits
        my_qubit: TunableTransmonQubit = copy.deepcopy(q0)
        my_qubit.update(
            {
                "ge_drive_amplitude_pi": 4,
                "ge_drive_amplitude_pi2": pi2,
            },
        )
        for angle in np.linspace(0, 1, 9):
            section = qops.rx(my_qubit, np.pi * angle)
            assert amp(section) / 4 == angle
            section = qops.ry(my_qubit, np.pi * angle)
            assert amp(section) / 4 == angle
        section = qops.rx(my_qubit, np.pi / 2)
        section2 = qops.rx(my_qubit, np.pi / 2 + 0.000001)
        assert abs(amp(section) - amp(section2)) < 0.0001
        section = qops.x90(my_qubit)
        assert amp(section) == pi2
        section = qops.x180(my_qubit)
        assert amp(section) == 4

        section = qops.ry(my_qubit, np.pi / 2)
        section2 = qops.ry(my_qubit, np.pi / 2 + 0.000001)
        assert abs(amp(section) - amp(section2)) < 0.0001
        section = qops.y90(my_qubit)
        assert amp(section) == pi2
        section = qops.y180(my_qubit)
        assert amp(section) == 4

    @pytest.mark.parametrize(
        ("transition", "expected_signal"),
        [
            pytest.param("ge", "drive", id="ge"),
            pytest.param("ef", "drive_ef", id="ef"),
        ],
    )
    def test_rx_transitions(
        self,
        transition,
        expected_signal,
        qops,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits
        section = qops.rx(q0, np.pi / 2, transition=transition)

        on_system_grid = transition == "ef"
        assert section == tsl.section(
            uid="__rx_q0_0",
            on_system_grid=on_system_grid,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(signal=f"/logical_signal_groups/q0/{expected_signal}"),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    @pytest.mark.parametrize(
        ("amplitude"),
        [
            pytest.param(0.8, id="constant"),
            pytest.param(
                SweepParameter(uid="sweep", values=[0, 0.1, 0.2]),
                id="sweep_parameter",
            ),
        ],
    )
    def test_rx_amplitude(
        self,
        amplitude,
        qops,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits
        section = qops.rx(q0, np.pi / 2, amplitude=amplitude)

        assert section == tsl.section(uid="__rx_q0_0").children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=amplitude,
            ),
        )

        sweep = amplitude if isinstance(amplitude, SweepParameter) else None
        self.check_op_builds_and_compiles(section, single_tunable_transmon, sweep=sweep)

    @pytest.mark.parametrize(
        ("phase"),
        [
            pytest.param(np.pi / 4, id="constant"),
            pytest.param(
                SweepParameter(uid="sweep", values=[np.pi / 4, np.pi / 2, np.pi]),
                id="sweep_parameter",
            ),
        ],
    )
    def test_rx_phase(
        self,
        phase,
        qops,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits
        section = qops.rx(q0, np.pi / 2, phase=phase)

        assert section == tsl.section(uid="__rx_q0_0").children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                phase=phase,
            ),
        )

        sweep = phase if isinstance(phase, SweepParameter) else None
        self.check_op_builds_and_compiles(section, single_tunable_transmon, sweep=sweep)

    @pytest.mark.parametrize(
        ("length"),
        [
            pytest.param(100e-9, id="constant"),
            pytest.param(
                SweepParameter(uid="sweep", values=[100e-9, 101e-9, 102e-9]),
                id="sweep_parameter",
            ),
        ],
    )
    def test_rx_length(
        self,
        length,
        qops,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits
        section = qops.rx(q0, np.pi / 2, length=length)

        assert section == tsl.section(uid="__rx_q0_0").children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                length=length,
            ),
        )

        sweep = length if isinstance(length, SweepParameter) else None
        self.check_op_builds_and_compiles(section, single_tunable_transmon, sweep=sweep)

    def test_x90(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.x90(q0)

        assert section == tsl.section(
            uid="__x90_q0_0",
            alignment=SectionAlignment.LEFT,
            on_system_grid=False,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=0.4,
                length=51e-9,
                increment_oscillator_phase=None,
                phase=0.0,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    function="drag",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters={"beta": 0.01, "sigma": 0.21},
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_x90_overrides(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.x90(
            q0,
            transition="ef",
            amplitude=0.1,
            phase=np.pi / 4,
            length=30e-9,
            pulse={"beta": 0.05},
        )

        assert section == tsl.section(
            uid="__x90_q0_0",
            alignment=SectionAlignment.LEFT,
            on_system_grid=True,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive_ef",
                amplitude=0.1,
                length=30e-9,
                increment_oscillator_phase=None,
                phase=np.pi / 4,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    function="drag",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters={"beta": 0.05, "sigma": 0.21},
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_x180(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.x180(q0)

        assert section == tsl.section(
            uid="__x180_q0_0",
            alignment=SectionAlignment.LEFT,
            on_system_grid=False,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=0.8,
                length=51e-9,
                increment_oscillator_phase=None,
                phase=0.0,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    function="drag",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters={"beta": 0.01, "sigma": 0.21},
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_x180_overrides(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.x180(
            q0,
            transition="ef",
            amplitude=0.1,
            phase=np.pi / 4,
            length=30e-9,
            pulse={"beta": 0.05},
        )

        assert section == tsl.section(
            uid="__x180_q0_0",
            alignment=SectionAlignment.LEFT,
            on_system_grid=True,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive_ef",
                amplitude=0.1,
                length=30e-9,
                increment_oscillator_phase=None,
                phase=np.pi / 4,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    function="drag",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters={"beta": 0.05, "sigma": 0.21},
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    @pytest.mark.parametrize(
        ("angle", "expected_amplitude"),
        [
            pytest.param(np.pi, 0.8, id="pi"),
            pytest.param(np.pi / 2, 0.4, id="pi_by_2"),
            pytest.param(np.pi / 3, 0.8 / 3, id="pi_by_3"),
        ],
    )
    def test_ry(self, angle, expected_amplitude, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.ry(q0, angle)

        assert section == tsl.section(
            uid="__ry_q0_0",
            alignment=SectionAlignment.LEFT,
            on_system_grid=False,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=expected_amplitude,
                length=51e-9,
                increment_oscillator_phase=None,
                phase=np.pi / 2,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    uid="__ry_pulse_0",
                    function="drag",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters={"beta": 0.01, "sigma": 0.21},
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    @pytest.mark.parametrize(
        ("transition", "expected_signal"),
        [
            pytest.param("ge", "drive", id="ge"),
            pytest.param("ef", "drive_ef", id="ef"),
        ],
    )
    def test_ry_transitions(
        self,
        transition,
        expected_signal,
        qops,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits
        section = qops.ry(q0, np.pi / 2, transition=transition)

        on_system_grid = transition == "ef"
        assert section == tsl.section(
            uid="__ry_q0_0",
            on_system_grid=on_system_grid,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal=f"/logical_signal_groups/q0/{expected_signal}",
                phase=np.pi / 2,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    @pytest.mark.parametrize(
        ("amplitude"),
        [
            pytest.param(0.8, id="constant"),
            pytest.param(
                SweepParameter(uid="sweep", values=[0, 0.1, 0.2]),
                id="sweep_parameter",
            ),
        ],
    )
    def test_ry_amplitude(
        self,
        amplitude,
        qops,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits
        section = qops.ry(q0, np.pi / 2, amplitude=amplitude)

        assert section == tsl.section(uid="__ry_q0_0").children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=amplitude,
                phase=np.pi / 2,
            ),
        )

        sweep = amplitude if isinstance(amplitude, SweepParameter) else None
        self.check_op_builds_and_compiles(section, single_tunable_transmon, sweep=sweep)

    @pytest.mark.parametrize(
        ("phase"),
        [
            pytest.param(np.pi / 4, id="constant"),
            pytest.param(
                SweepParameter(uid="sweep", values=[np.pi / 4, np.pi / 2, np.pi]),
                id="sweep_parameter",
            ),
        ],
    )
    def test_ry_phase(
        self,
        phase,
        qops,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits
        section = qops.ry(q0, np.pi / 2, phase=phase)

        assert section == tsl.section(uid="__ry_q0_0").children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                phase=phase,
            ),
        )

        sweep = phase if isinstance(phase, SweepParameter) else None
        self.check_op_builds_and_compiles(section, single_tunable_transmon, sweep=sweep)

    @pytest.mark.parametrize(
        ("length"),
        [
            pytest.param(100e-9, id="constant"),
            pytest.param(
                SweepParameter(uid="sweep", values=[100e-9, 101e-9, 102e-9]),
                id="sweep_parameter",
            ),
        ],
    )
    def test_ry_length(
        self,
        length,
        qops,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits
        section = qops.ry(q0, np.pi / 2, length=length)

        assert section == tsl.section(uid="__ry_q0_0").children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                length=length,
                phase=np.pi / 2,
            ),
        )

        sweep = length if isinstance(length, SweepParameter) else None
        self.check_op_builds_and_compiles(section, single_tunable_transmon, sweep=sweep)

    def test_y90(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.y90(q0)

        assert section == tsl.section(
            uid="__y90_q0_0",
            alignment=SectionAlignment.LEFT,
            on_system_grid=False,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=0.4,
                length=51e-9,
                increment_oscillator_phase=None,
                phase=np.pi / 2,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    function="drag",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters={"beta": 0.01, "sigma": 0.21},
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_y90_overrides(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.y90(
            q0,
            transition="ef",
            amplitude=0.1,
            phase=np.pi / 4,
            length=30e-9,
            pulse={"beta": 0.05},
        )

        assert section == tsl.section(
            uid="__y90_q0_0",
            alignment=SectionAlignment.LEFT,
            on_system_grid=True,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive_ef",
                amplitude=0.1,
                length=30e-9,
                increment_oscillator_phase=None,
                phase=np.pi / 4,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    function="drag",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters={"beta": 0.05, "sigma": 0.21},
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_y180(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.y180(q0)

        assert section == tsl.section(
            uid="__y180_q0_0",
            alignment=SectionAlignment.LEFT,
            on_system_grid=False,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=0.8,
                length=51e-9,
                increment_oscillator_phase=None,
                phase=np.pi / 2,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    function="drag",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters={"beta": 0.01, "sigma": 0.21},
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_y180_overrides(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.y180(
            q0,
            transition="ef",
            amplitude=0.1,
            phase=np.pi / 4,
            length=30e-9,
            pulse={"beta": 0.05},
        )

        assert section == tsl.section(
            uid="__y180_q0_0",
            alignment=SectionAlignment.LEFT,
            on_system_grid=True,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive_ef",
                amplitude=0.1,
                length=30e-9,
                increment_oscillator_phase=None,
                phase=np.pi / 4,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    function="drag",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters={"beta": 0.05, "sigma": 0.21},
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    @pytest.mark.parametrize(
        "angle",
        [
            pytest.param(np.pi, id="pi"),
            pytest.param(np.pi / 2, id="pi_by_2"),
            pytest.param(np.pi / 3, id="pi_by_3"),
        ],
    )
    def test_rz(self, angle, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.rz(q0, angle)

        assert section == tsl.section(
            uid="__rz_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=None,
                length=None,
                increment_oscillator_phase=angle,
                phase=None,
                pulse_parameters=None,
                pulse=None,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    @pytest.mark.parametrize(
        ("transition", "expected_signal"),
        [
            pytest.param("ge", "drive", id="ge"),
            pytest.param("ef", "drive_ef", id="ef"),
        ],
    )
    def test_rz_transitions(
        self,
        transition,
        expected_signal,
        qops,
        single_tunable_transmon,
    ):
        [q0] = single_tunable_transmon.qubits
        section = qops.rz(q0, np.pi / 2, transition=transition)

        assert section == tsl.section(uid="__rz_q0_0").children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal=f"/logical_signal_groups/q0/{expected_signal}",
                increment_oscillator_phase=np.pi / 2,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_z90(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.z90(q0)

        assert section == tsl.section(
            uid="__z90_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=None,
                length=None,
                increment_oscillator_phase=np.pi / 2,
                phase=None,
                pulse_parameters=None,
                pulse=None,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_z180(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.z180(q0)

        assert section == tsl.section(
            uid="__z180_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.play_pulse_op(
                signal="/logical_signal_groups/q0/drive",
                amplitude=None,
                length=None,
                increment_oscillator_phase=np.pi,
                phase=None,
                pulse_parameters=None,
                pulse=None,
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)
