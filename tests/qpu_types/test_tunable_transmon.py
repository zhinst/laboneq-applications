"""Tests for laboneq_applications.qpu_types.tunable_transmon."""

import copy
from contextlib import nullcontext

import numpy as np
import pytest
from laboneq.simple import SectionAlignment, Session, SweepParameter

import tests.helpers.dsl as tsl
from laboneq_applications.core.build_experiment import build
from laboneq_applications.core.quantum_operations import dsl
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonOperations,
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
    modify_qubits,
    modify_qubits_context,
)


@pytest.fixture()
def qops():
    """Return TunableTransmonQubit operations."""
    return TunableTransmonOperations()


@pytest.fixture()
def q0(single_tunable_transmon):
    return single_tunable_transmon.qubits[0]


@pytest.fixture()
def multi_qubits(two_tunable_transmon):
    return two_tunable_transmon.qubits


class TestTunableTransmonQubit:
    def test_create(self):
        q = TunableTransmonQubit()

        assert isinstance(q.parameters, TunableTransmonQubitParameters)

    def test_transition_parameters_default(self, q0):
        drive_line, params = q0.transition_parameters()
        assert drive_line == "drive"
        assert params["amplitude_pi"] == 0.4

    def test_transition_parameters_ge(self, q0):
        drive_line, params = q0.transition_parameters("ge")
        assert drive_line == "drive"
        assert params["amplitude_pi"] == 0.4

    def test_transition_parameters_ef(self, q0):
        drive_line, params = q0.transition_parameters("ef")
        assert drive_line == "drive_ef"
        assert params["amplitude_pi"] == 0.35

    def test_transition_parameters_error(self, q0):
        with pytest.raises(ValueError) as err:
            q0.transition_parameters("gef")
        assert str(err.value) == "Transition 'gef' is not one of None, 'ge' or 'ef'."

    def test_default_integration_kernels(self, q0):
        assert q0.default_integration_kernels() == [
            tsl.pulse(function="const", amplitude=1, length=2e-6),
        ]

    def test_get_integration_kernels_default(self, q0):
        q0.parameters.readout_integration_parameters["kernels"] = "default"
        assert q0.get_integration_kernels() == [
            tsl.pulse(
                uid="__integration_kernel_q0_0",
                function="const",
                amplitude=1,
                length=2e-6,
            ),
        ]

    def test_get_integration_kernels_pulses(self, q0):
        q0.parameters.readout_integration_parameters["kernels"] = [
            {"function": "const", "amplitude": 2.0},
        ]
        assert q0.get_integration_kernels() == [
            tsl.pulse(
                uid="__integration_kernel_q0_0",
                function="const",
                amplitude=2.0,
                length=1e-7,
            ),
        ]

    def test_get_integration_kernel_overrides(self, q0):
        assert q0.get_integration_kernels([{"function": "const"}]) == [
            tsl.pulse(
                uid="__integration_kernel_q0_0",
                function="const",
                amplitude=1,
                length=1e-7,
            ),
        ]

    def test_get_integration_kernels_empty_list(self, q0):
        q0.parameters.readout_integration_parameters["kernels"] = []
        with pytest.raises(TypeError) as err:
            q0.get_integration_kernels()

        assert str(err.value) == (
            "TunableTransmonQubit readout integration kernels"
            " should be either 'default' or a list of pulse dictionaries."
        )

    def test_get_integration_kernels_invalid_type(self, q0):
        q0.parameters.readout_integration_parameters["kernels"] = "zoo"
        with pytest.raises(TypeError) as err:
            q0.get_integration_kernels()

        assert str(err.value) == (
            "TunableTransmonQubit readout integration kernels"
            " should be either 'default' or a list of pulse dictionaries."
        )


class TestOverrideParameters:
    def _equal_except(self, q0, q0_temp, key):
        for k, v in q0.__dict__.items():
            if k not in ("parameters"):
                assert getattr(q0_temp, k) == v
        for k, v in q0.parameters.__dict__.items():
            if k == key:
                continue
            assert getattr(q0_temp.parameters, k) == v

    def test_create_temp_single_qubit(self, q0):
        original_q0 = copy.deepcopy(q0)
        [q0_temp] = modify_qubits([(q0, {"readout_range_out": 10})])
        assert q0_temp.parameters.readout_range_out == 10
        assert original_q0 == q0
        self._equal_except(q0, q0_temp, "readout_range_out")

        q0.parameters.nested_params = {"a": {"a": 2}, "b": 1}
        # replacing the nested dictionary
        [q0_temp] = modify_qubits([(q0, {"nested_params": {"a": 2}})])
        assert q0_temp.parameters.nested_params == {"a": 2}
        assert original_q0 == q0
        self._equal_except(q0, q0_temp, "nested_params")

        # replacing partially the nested dictionary
        [q0_temp] = modify_qubits([(q0, {"nested_params.a.a": 3})])
        assert q0_temp.parameters.nested_params == {"a": {"a": 3}, "b": 1}
        assert original_q0 == q0
        self._equal_except(q0, q0_temp, "nested_params")

        # replacing a real nested parameter of qubits
        [q0_temp] = modify_qubits(
            [(q0, {"drive_parameters_ge.pulse.beta": 0.1})],
        )
        assert q0_temp.parameters.drive_parameters_ge["pulse"]["beta"] == 0.1
        assert original_q0 == q0
        self._equal_except(q0, q0_temp, "drive_parameters_ge")

    def test_override_multiple_qubits(self, multi_qubits):
        q0, q1 = multi_qubits
        original_q0 = copy.deepcopy(q0)
        original_q1 = copy.deepcopy(q1)

        [q0_temp, q1_temp] = modify_qubits(
            zip(multi_qubits, [{"readout_range_out": 10}, {"readout_range_out": 20}]),
        )
        assert q0_temp.parameters.readout_range_out == 10
        assert q1_temp.parameters.readout_range_out == 20
        assert original_q0 == q0
        assert original_q1 == q1
        self._equal_except(q0, q0_temp, "readout_range_out")
        self._equal_except(q1, q1_temp, "readout_range_out")

        q0.parameters.nested_params = {"a": {"a": 2}, "b": 1}
        q1.parameters.nested_params = {"a": {"a": 3}, "b": 2}
        [q0_temp, q1_temp] = modify_qubits(
            zip(multi_qubits, [{"nested_params": {"a": 2}}, {"nested_params.a.a": 3}]),
        )
        assert q0_temp.parameters.nested_params == {"a": 2}
        assert q1_temp.parameters.nested_params == {"a": {"a": 3}, "b": 2}
        assert original_q0 == q0
        assert original_q1 == q1
        self._equal_except(q0, q0_temp, "nested_params")
        self._equal_except(q1, q1_temp, "nested_params")

        [q0_temp, q1_temp] = modify_qubits(
            [
                (q0, {"drive_parameters_ge.pulse.beta": 0.1}),
                (q1, {"drive_parameters_ef.pulse.sigma": 0.1}),
            ],
        )
        assert q0_temp.parameters.drive_parameters_ge["pulse"]["beta"] == 0.1
        assert q1_temp.parameters.drive_parameters_ef["pulse"]["sigma"] == 0.1
        assert original_q0 == q0
        assert original_q1 == q1
        self._equal_except(q0, q0_temp, "drive_parameters_ge")
        self._equal_except(q1, q1_temp, "drive_parameters_ef")

    def test_temporary_qubits_context(self, multi_qubits):
        q0, q1 = multi_qubits
        with modify_qubits_context(
            zip(multi_qubits, [{"readout_range_out": 10}, {"readout_range_out": 20}]),
        ) as temp_qubits:
            q0_temp, q1_temp = temp_qubits
            assert q0_temp.parameters.readout_range_out == 10
            assert q1_temp.parameters.readout_range_out == 20
            assert q0 == q0_temp
            assert q1 == q1_temp
            self._equal_except(q0, q0_temp, "readout_range_out")
            self._equal_except(q1, q1_temp, "readout_range_out")


class TestTunableTransmonParameters:
    def test_create(self):
        p = TunableTransmonQubitParameters()

        assert p.readout_range_out == 5
        assert p.readout_range_in == 10


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

        assert qops.QUBIT_TYPE is TunableTransmonQubit
        assert qops.QUBIT_TYPE.TRANSITIONS == ("ge", "ef")

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

    def test_prep_g(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.prep(q0, "g")

        assert section == tsl.section(
            uid="__prep_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.section(uid="__reset_q0_0").children(
                self.reserve_ops(q0),
                tsl.delay_op(
                    signal="/logical_signal_groups/q0/drive",
                    time=1e-6,
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_prep_e(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.prep(q0, "e")

        assert section == tsl.section(
            uid="__prep_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.section(uid="__reset_q0_0").children(
                self.reserve_ops(q0),
                tsl.delay_op(
                    signal="/logical_signal_groups/q0/drive",
                    time=1e-6,
                ),
            ),
            tsl.section(uid="__x180_q0_0").children(
                self.reserve_ops(q0),
                tsl.play_pulse_op(
                    signal="/logical_signal_groups/q0/drive",
                    amplitude=0.4,
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_prep_f(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.prep(q0, "f")

        assert section == tsl.section(
            uid="__prep_q0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            self.reserve_ops(q0),
            tsl.section(uid="__reset_q0_0").children(
                self.reserve_ops(q0),
                tsl.delay_op(
                    signal="/logical_signal_groups/q0/drive",
                    time=1e-6,
                ),
            ),
            tsl.section(uid="__x180_q0_0").children(
                self.reserve_ops(q0),
                tsl.play_pulse_op(
                    signal="/logical_signal_groups/q0/drive",
                    amplitude=0.4,
                ),
            ),
            tsl.section(uid="__x180_q0_1").children(
                self.reserve_ops(q0),
                tsl.play_pulse_op(
                    signal="/logical_signal_groups/q0/drive_ef",
                    amplitude=0.35,
                ),
            ),
        )

        self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_prep_invalid(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        with pytest.raises(ValueError) as err:
            qops.prep(q0, "z")
        assert str(err.value) == "Only states g, e and f can be prepared, not 'z'"

    def test_reset(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.reset(q0)

        assert section == tsl.section(
            uid="__reset_q0_0",
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
        ("angle", "expected_amplitude"),
        [
            pytest.param(np.pi, 0.4, id="pi"),
            pytest.param(np.pi / 2, 0.8, id="pi_by_2"),
            pytest.param(np.pi / 3, 0.4 / 3, id="pi_by_3"),
        ],
    )
    def test_rx(self, angle, expected_amplitude, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.rx(q0, angle)

        assert section == tsl.section(
            uid="__rx_q0_0",
            alignment=SectionAlignment.LEFT,
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

        assert section == tsl.section(uid="__rx_q0_0").children(
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

        if isinstance(amplitude, SweepParameter):
            self.check_op_builds_and_compiles(
                section,
                single_tunable_transmon,
                sweep=amplitude,
            )
        else:
            self.check_op_builds_and_compiles(section, single_tunable_transmon)

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

        if isinstance(phase, SweepParameter):
            self.check_op_builds_and_compiles(
                section,
                single_tunable_transmon,
                sweep=phase,
            )
        else:
            self.check_op_builds_and_compiles(section, single_tunable_transmon)

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

        if isinstance(length, SweepParameter):
            self.check_op_builds_and_compiles(
                section,
                single_tunable_transmon,
                sweep=length,
            )
        else:
            self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_x90(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.x90(q0)

        assert section == tsl.section(
            uid="__x90_q0_0",
            alignment=SectionAlignment.LEFT,
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
            pytest.param(np.pi, 0.4, id="pi"),
            pytest.param(np.pi / 2, 0.8, id="pi_by_2"),
            pytest.param(np.pi / 3, 0.4 / 3, id="pi_by_3"),
        ],
    )
    def test_ry(self, angle, expected_amplitude, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.ry(q0, angle)

        assert section == tsl.section(
            uid="__ry_q0_0",
            alignment=SectionAlignment.LEFT,
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

        assert section == tsl.section(uid="__ry_q0_0").children(
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

        if isinstance(amplitude, SweepParameter):
            self.check_op_builds_and_compiles(
                section,
                single_tunable_transmon,
                sweep=amplitude,
            )
        else:
            self.check_op_builds_and_compiles(section, single_tunable_transmon)

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

        if isinstance(phase, SweepParameter):
            self.check_op_builds_and_compiles(
                section,
                single_tunable_transmon,
                sweep=phase,
            )
        else:
            self.check_op_builds_and_compiles(section, single_tunable_transmon)

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

        if isinstance(length, SweepParameter):
            self.check_op_builds_and_compiles(
                section,
                single_tunable_transmon,
                sweep=length,
            )
        else:
            self.check_op_builds_and_compiles(section, single_tunable_transmon)

    def test_y90(self, qops, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        section = qops.y90(q0)

        assert section == tsl.section(
            uid="__y90_q0_0",
            alignment=SectionAlignment.LEFT,
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
