"""Tests for laboneq_applications.core.quantum_operations."""

import numpy as np
import pytest
from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.experiment import builtins
from laboneq.simple import (
    ExecutionType,
    QuantumElement,
    Section,
    SectionAlignment,
    SweepParameter,
    Transmon,
)

import tests.helpers.dsl as tsl
from laboneq_applications import dsl
from laboneq_applications.core.quantum_operations import (
    QuantumOperations,
    _PulseCache,
    create_pulse,
    quantum_operation,
)


class DummyOperations(QuantumOperations):
    """Dummy operations for testing."""

    QUBIT_TYPE = Transmon

    @quantum_operation
    def x(self, q, amplitude):
        """A dummy quantum operation."""
        pulse = dsl.pulse_library.const()
        dsl.play(q.signals["drive"], pulse, amplitude=amplitude)


class ForeignQubit(QuantumElement):
    """A non-Transmon qubit to test passing qubits of incorrect type."""

    def calibration(self):
        """Dummy calibration method."""
        return


@pytest.fixture()
def dummy_ops():
    return DummyOperations()


@pytest.fixture()
def dummy_q():
    return Transmon(
        uid="q0",
        signals={
            "drive": "/lsg/q0/drive",
            "measure": "/lsg/q0/measure",
            "acquire": "/lsg/q0/acquire",
        },
    )


def reserve_ops_for_dummy_q():
    return [
        tsl.reserve_op(signal="/lsg/q0/drive"),
        tsl.reserve_op(signal="/lsg/q0/measure"),
        tsl.reserve_op(signal="/lsg/q0/acquire"),
    ]


class TestQuantumOperationDecorator:
    def test_decorator(self):
        @quantum_operation
        def x(self, q, amplitude):
            pass

        assert x._quantum_op


class TestPulseCache:
    def test_global_cache(self):
        cache = _PulseCache.experiment_or_global_cache()
        assert cache.cache is _PulseCache.GLOBAL_CACHE

    def test_experiment_cache(self):
        with dsl.experiment():
            cache = _PulseCache.experiment_or_global_cache()
            ctx = builtins.current_experiment_context()
        assert cache.cache is not _PulseCache.GLOBAL_CACHE
        assert cache is ctx._pulse_cache

    def test_reset_global_cache(self):
        cache = _PulseCache.experiment_or_global_cache()
        pulse = object()  # dummy pulse

        cache.store(pulse, "name", "const", {"amplitude": 1.0})
        assert cache.get("name", "const", {"amplitude": 1.0}) is pulse
        _PulseCache.reset_global_cache()
        assert cache.get("name", "const", {"amplitude": 1.0}) is None

    def test_get_and_store(self):
        cache = _PulseCache()
        pulse = object()  # dummy pulse

        assert cache.get("name", "const", {"amplitude": 1.0}) is None
        cache.store(pulse, "name", "const", {"amplitude": 1.0})
        assert cache.get("name", "const", {"amplitude": 1.0}) is pulse


class TestCreatePulse:
    def test_no_overrides(self):
        pulse = create_pulse({"function": "const", "amplitude": 0.2})
        assert pulse == tsl.pulse(function="const", amplitude=0.2)

    def test_with_overrides(self):
        pulse = create_pulse(
            {"function": "const", "amplitude": 0.2},
            {"amplitude": 0.1},
        )
        assert pulse == tsl.pulse(function="const", amplitude=0.1)

    def test_overridden_function(self):
        pulse = create_pulse(
            {"function": "const", "amplitude": 0.2},
            {"function": "gaussian"},
        )
        assert pulse == tsl.pulse(function="gaussian", amplitude=1.0)

    def test_invalid_function(self):
        with pytest.raises(ValueError) as err:
            create_pulse({"function": "__moo__"})
        assert str(err.value) == "Unsupported pulse function '__moo__'."

    def test_no_function(self):
        with pytest.raises(KeyError) as err:
            create_pulse({})
        assert str(err.value) == "'function'"

    def test_uid_default(self):
        pulse = create_pulse({"function": "const", "amplitude": 0.2})
        assert pulse == tsl.pulse(uid="__unnamed_0")

    def test_uid_from_name(self):
        pulse = create_pulse({"function": "const", "amplitude": 0.2}, name="rx_pulse")
        assert pulse == tsl.pulse(uid="__rx_pulse_0")

    def test_pulse_from_global_cache(self):
        pulse_1 = create_pulse({"function": "const", "amplitude": 0.2}, name="rx_pulse")
        pulse_2 = create_pulse({"function": "const", "amplitude": 0.2}, name="rx_pulse")
        assert pulse_1 is pulse_2
        assert pulse_1.uid == "__rx_pulse_0"

    def test_pulse_from_experiment_cache(self):
        with dsl.experiment():
            pulse_1 = create_pulse(
                {"function": "const", "amplitude": 0.2},
                name="rx_pulse",
            )
            pulse_2 = create_pulse(
                {"function": "const", "amplitude": 0.2},
                name="rx_pulse",
            )
        assert pulse_1 is pulse_2
        assert pulse_1.uid == "rx_pulse_0"


class TestDsl:
    def test_dsl_attributes(self):
        assert dsl.acquire_loop_rt is builtins.acquire_loop_rt
        assert dsl.delay is builtins.delay
        assert dsl.measure is builtins.measure
        assert dsl.play is builtins.play
        assert dsl.reserve is builtins.reserve
        assert dsl.section is builtins.section
        assert dsl.sweep is builtins.sweep


class TestQuantumOperations:
    @staticmethod
    @dsl.qubit_experiment
    def simple_exp(qop, q, amplitudes, shots, prep=None):
        with dsl.acquire_loop_rt(count=shots):
            with dsl.sweep(parameter=SweepParameter(values=amplitudes)) as amplitude:
                if prep:
                    prep(q)
                qop.x(q, amplitude)

    @staticmethod
    @dsl.qubit_experiment
    def single_op_exp(qop, q, op_name):
        qop[op_name](q)

    def test_create(self, dummy_ops):
        assert dummy_ops.QUBIT_TYPE is Transmon

    def test_quantum_operation(self, dummy_ops, dummy_q):
        section = dummy_ops.x(dummy_q, amplitude=1.5)
        assert type(section) is Section

    def test_quantum_operation_docstring(self, dummy_ops):
        assert dummy_ops.x.__doc__ == "A dummy quantum operation."

    def test_getattr(self, dummy_ops):
        assert dummy_ops.x.op is dummy_ops.BASE_OPS["x"]
        with pytest.raises(AttributeError) as exc:
            dummy_ops.y  # noqa: B018
        assert str(exc.value) == "'DummyOperations' object has no attribute 'y'"

    def test_getitem(self, dummy_ops):
        assert dummy_ops["x"].op is dummy_ops.BASE_OPS["x"]
        with pytest.raises(KeyError) as exc:
            dummy_ops["y"]
        assert str(exc.value) == "'y'"

    def test_setitem_callable(self, dummy_ops):
        def f():
            pass

        dummy_ops["y"] = f
        assert dummy_ops["y"].op is f

        dummy_ops["x"] = f
        assert dummy_ops["x"].op is f

    def test_setitem_op(self, dummy_ops):
        def f():
            pass

        dummy_ops["y"] = f
        dummy_ops["x"] = dummy_ops.y
        assert dummy_ops["x"].op is f
        assert dummy_ops["x"] is dummy_ops["y"]

    def test_contains(self, dummy_ops):
        assert "x" in dummy_ops
        assert "y" not in dummy_ops

    def test_dir(self, dummy_ops):
        public_attrs = [attr for attr in dir(dummy_ops) if not attr.startswith("_")]
        assert public_attrs == [
            "BASE_OPS",
            "QUBIT_TYPE",
            "register",
            "x",
        ]

    def test_register(self, dummy_ops):
        def y(q):
            pass

        assert "y" not in dummy_ops
        dummy_ops.register(y)

        assert dummy_ops.y.op is y

    def test_register_with_name(self, dummy_ops):
        def y(q):
            pass

        assert "y" not in dummy_ops
        dummy_ops.register(y, name="yo")

        assert "y" not in dummy_ops
        assert dummy_ops.yo.op is y

    def test_build(self, dummy_ops, dummy_q):
        amplitudes = np.linspace(0.1, 1, 10)
        exp = self.simple_exp(dummy_ops, dummy_q, amplitudes, 5)

        assert exp == tsl.experiment(uid="simple_exp").children(
            tsl.acquire_loop_rt().children(
                tsl.sweep(parameters=[tsl.sweep_parameter(values=amplitudes)]).children(
                    tsl.section(uid="x_q0_0").children(
                        reserve_ops_for_dummy_q(),
                        tsl.play_pulse_op(
                            signal="/lsg/q0/drive",
                            amplitude=tsl.sweep_parameter(values=amplitudes),
                        ),
                    ),
                ),
            ),
        )

    def test_build_with_prep(self, dummy_ops, dummy_q):
        prep = dummy_ops.x.partial(amplitude=2)
        amplitudes = np.linspace(0.1, 1, 10)

        exp = self.simple_exp(dummy_ops, dummy_q, amplitudes, 5, prep=prep)

        assert exp == tsl.experiment(uid="simple_exp").children(
            tsl.acquire_loop_rt().children(
                tsl.sweep(parameters=[tsl.sweep_parameter(values=amplitudes)]).children(
                    tsl.section(uid="x_q0_0").children(
                        reserve_ops_for_dummy_q(),
                        tsl.play_pulse_op(signal="/lsg/q0/drive", amplitude=2.0),
                    ),
                    tsl.section(uid="x_q0_1").children(
                        reserve_ops_for_dummy_q(),
                        tsl.play_pulse_op(
                            signal="/lsg/q0/drive",
                            amplitude=tsl.sweep_parameter(values=amplitudes),
                        ),
                    ),
                ),
            ),
        )

    def test_build_with_nested_ops(self, dummy_ops, dummy_q):
        def x_amp_2(self, q):
            self.x(q, amplitude=2.0)

        dummy_ops.register(x_amp_2)

        exp = self.single_op_exp(dummy_ops, dummy_q, "x_amp_2")

        assert exp == tsl.experiment(uid="single_op_exp").children(
            tsl.section(uid="x_amp_2_q0_0").children(
                reserve_ops_for_dummy_q(),
                tsl.section(uid="x_q0_0").children(
                    reserve_ops_for_dummy_q(),
                    tsl.play_pulse_op(signal="/lsg/q0/drive", amplitude=2.0),
                ),
            ),
        )


class TestOperation:
    def test_call(self, dummy_ops, dummy_q):
        section_1 = dummy_ops.x(dummy_q, amplitude=2.0)
        section_2 = dummy_ops.x(dummy_q, amplitude=3.0)

        assert section_1 == tsl.section(uid="__x_q0_0").children(
            tsl.reserve_op(signal="/lsg/q0/drive"),
            tsl.reserve_op(signal="/lsg/q0/measure"),
            tsl.reserve_op(signal="/lsg/q0/acquire"),
            tsl.play_pulse_op(signal="/lsg/q0/drive", amplitude=2.0),
        )

        assert section_2 == tsl.section(uid="__x_q0_1").children(
            tsl.reserve_op(signal="/lsg/q0/drive"),
            tsl.reserve_op(signal="/lsg/q0/measure"),
            tsl.reserve_op(signal="/lsg/q0/acquire"),
            tsl.play_pulse_op(signal="/lsg/q0/drive", amplitude=3.0),
        )

    def test_call_with_foreign_qubits(self, dummy_ops):
        q = ForeignQubit(uid="q0")
        with pytest.raises(TypeError) as err:
            dummy_ops.x(q, amplitude=1.0)
        assert str(err.value) == (
            "Quantum operation 'x' was passed the following qubits"
            " that are not of type Transmon: q0"
        )

    def test_partial(self, dummy_ops, dummy_q):
        x_with_amp = dummy_ops.x.partial(amplitude=1.5)

        section = x_with_amp(dummy_q)

        assert section == tsl.section(uid="__x_q0_0").children(
            tsl.reserve_op(signal="/lsg/q0/drive"),
            tsl.reserve_op(signal="/lsg/q0/measure"),
            tsl.reserve_op(signal="/lsg/q0/acquire"),
            tsl.play_pulse_op(signal="/lsg/q0/drive", amplitude=1.5),
        )

    def test_section_omit(self, dummy_ops, dummy_q):
        x = dummy_ops.x.partial(amplitude=1.0)
        q = dummy_q

        with dsl.section(uid="top") as section:
            result = x.section(omit=True)(q)

        assert section == tsl.section(uid="top").children(
            tsl.play_pulse_op(),
        )
        assert result is None

        with pytest.raises(LabOneQException) as err:
            x.section(omit=True)(q)
        assert str(err.value) == "Must be in a section context"

    def test_section_alignment(self, dummy_ops, dummy_q):
        x = dummy_ops.x.partial(amplitude=1.0)
        q = dummy_q

        section_default = x(q)
        assert section_default.alignment == SectionAlignment.LEFT

        section_right = x.section(alignment=SectionAlignment.RIGHT)(q)
        assert section_right.alignment == SectionAlignment.RIGHT

        section_left = x.section(alignment=SectionAlignment.LEFT)(q)
        assert section_left.alignment == SectionAlignment.LEFT

        section_right_left = x.section(alignment=SectionAlignment.RIGHT).section(
            alignment=SectionAlignment.LEFT,
        )(q)
        assert section_right_left.alignment == SectionAlignment.LEFT

        section_restored_default = x.section(alignment=SectionAlignment.RIGHT).section(
            alignment=None,
        )(q)
        assert section_restored_default.alignment == SectionAlignment.LEFT

    def test_section_execution_type(self, dummy_ops, dummy_q):
        x = dummy_ops.x.partial(amplitude=1.0)
        q = dummy_q

        section_default = x(q)
        assert section_default.execution_type is None

        section_real_time = x.section(execution_type=ExecutionType.REAL_TIME)(q)
        assert section_real_time.execution_type == ExecutionType.REAL_TIME

        section_near_time = x.section(execution_type=ExecutionType.NEAR_TIME)(q)
        assert section_near_time.execution_type == ExecutionType.NEAR_TIME

        section_restored_default = x.section(
            execution_type=ExecutionType.REAL_TIME,
        ).section(execution_type=None)(q)
        assert section_restored_default.execution_type is None

    def test_section_length(self, dummy_ops, dummy_q):
        x = dummy_ops.x.partial(amplitude=1.0)
        q = dummy_q

        section_default = x(q)
        assert section_default.length is None

        section_altered = x.section(length=1e-6)(q)
        assert section_altered.length == 1e-6

        section_restored_default = x.section(length=1e-6).section(length=None)(q)
        assert section_restored_default.length is None

    def test_section_play_after(self, dummy_ops, dummy_q):
        x = dummy_ops.x.partial(amplitude=1.0)
        q = dummy_q

        section_default = x(q)
        assert section_default.play_after is None

        section_altered = x.section(play_after=["x180_q0_0"])(q)
        assert section_altered.play_after == ["x180_q0_0"]

        section_restored_default = x.section(play_after=["x180_q0_0"]).section(
            play_after=None,
        )(q)
        assert section_restored_default.play_after is None

    def test_section_on_system_grid(self, dummy_ops, dummy_q):
        x = dummy_ops.x.partial(amplitude=1.0)
        q = dummy_q

        section_default = x(q)
        assert section_default.on_system_grid is False

        section_altered = x.section(on_system_grid=True)(q)
        assert section_altered.on_system_grid is True

        section_restored_default = x.section(on_system_grid=True).section(
            on_system_grid=None,
        )(q)
        assert section_restored_default.on_system_grid is False

    def test_section_mixed_parameters(self, dummy_ops, dummy_q):
        x = dummy_ops.x.partial(amplitude=1.0)
        q = dummy_q

        section = (
            x.section(alignment=SectionAlignment.RIGHT, on_system_grid=True)
            .section(length=1e-6)
            .section(play_after=["x180_q0_0"])(q)
        )

        assert section.alignment == SectionAlignment.RIGHT
        assert section.on_system_grid is True
        assert section.length == 1e-6
        assert section.play_after == ["x180_q0_0"]

    def test_op(self, dummy_ops):
        assert dummy_ops.x.op is DummyOperations.BASE_OPS["x"]

    def test_src(self, dummy_ops):
        assert dummy_ops.x.src == "\n".join(
            [
                "@quantum_operation",
                "def x(self, q, amplitude):",
                '    """A dummy quantum operation."""',
                "    pulse = dsl.pulse_library.const()",
                '    dsl.play(q.signals["drive"], pulse, amplitude=amplitude)',
                "",
            ],
        )
