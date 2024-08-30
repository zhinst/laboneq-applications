"""Tests for laboneq_applications.core.quantum_operations."""

import functools

import numpy as np
import pytest
from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.experiment import builtins, pulse_library
from laboneq.simple import (
    QuantumElement,
    Section,
    SweepParameter,
    Transmon,
)

from laboneq_applications import dsl
from laboneq_applications.core.quantum_operations import (
    QuantumOperations,
    _PulseCache,
    create_pulse,
    quantum_operation,
)

import tests.helpers.dsl as tsl


class DummyOperations(QuantumOperations):
    """Dummy operations for testing."""

    QUBIT_TYPES = Transmon

    @quantum_operation
    def x(self, q, amplitude):
        """A dummy quantum operation."""
        pulse = dsl.pulse_library.const()
        dsl.play(q.signals["drive"], pulse, amplitude=amplitude)


class DummyCoupler(Transmon):
    """Dummy coupler qubit for testing."""


class DummyMultiTypeOperations(DummyOperations):
    """Dummy operations for testing multi qubit-type support."""

    QUBIT_TYPES = (DummyCoupler, Transmon)

    @quantum_operation
    def cz(self, q0, q1, amplitude=1.0):
        """A dummy two qubit quantum operation."""
        pulse = dsl.pulse_library.const()
        dsl.play(q0.signals["drive"], pulse, amplitude=amplitude)
        dsl.play(q1.signals["drive"], pulse, amplitude=amplitude)


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


@pytest.fixture()
def dummy_multi_type_ops():
    return DummyMultiTypeOperations()


@pytest.fixture()
def dummy_coupler_q():
    return DummyCoupler(
        uid="q1",
        signals={
            "drive": "/lsg/q1/drive",
            "measure": "/lsg/q1/measure",
            "acquire": "/lsg/q1/acquire",
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

    def test_pulse_from_user_cache(self):
        @pulse_library.register_pulse_functional
        def test_pulse(x, **_):
            return np.ones(len(x))

        with dsl.experiment():
            pulse_1 = create_pulse(
                {"function": "test_pulse", "amplitude": 0.2, "length": 1e-6},
                name="some_pulse",
            )
            pulse_2 = create_pulse(
                {"function": "test_pulse", "amplitude": 0.2, "length": 1e-6},
                name="some_pulse",
            )
        assert pulse_1 is pulse_2
        assert pulse_1.uid == "some_pulse_0"
        assert pulse_1.amplitude == 0.2
        assert pulse_1.function == "test_pulse"

    def test_unsupported_pulse_function(self):
        with pytest.raises(ValueError) as err:
            create_pulse({"function": "missing"}, name="rx")
        assert str(err.value) == "Unsupported pulse function 'missing'."


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
        assert dummy_ops.QUBIT_TYPES is Transmon

    def test_create_multi_type(self, dummy_multi_type_ops):
        assert (DummyCoupler, Transmon) == dummy_multi_type_ops.QUBIT_TYPES

    def test_quantum_operation(self, dummy_ops, dummy_q):
        section = dummy_ops.x(dummy_q, amplitude=1.5)
        assert type(section) is Section

    def test_quantum_operation_multi_type(
        self,
        dummy_multi_type_ops,
        dummy_q,
        dummy_coupler_q,
    ):
        section = dummy_multi_type_ops.x(dummy_coupler_q, amplitude=1.5)
        assert type(section) is Section

        section = dummy_multi_type_ops.cz(dummy_coupler_q, dummy_q)
        assert type(section) is Section

    def test_quantum_operation_docstring(self, dummy_ops):
        assert dummy_ops.x.__doc__ == "A dummy quantum operation."

    def test_getattr(self, dummy_ops):
        assert dummy_ops.x.op is dummy_ops.BASE_OPS["x"]
        with pytest.raises(AttributeError) as exc:
            dummy_ops.y #noqa: B018
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
            "QUBIT_TYPES",
            "keys",
            "register",
            "x",
        ]

    def test_keys(self, dummy_ops):
        assert dummy_ops.keys() == ["x"]
        dummy_ops.register(dummy_ops.x, "a")
        assert dummy_ops.keys() == ["a", "x"]

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
        prep = functools.partial(dummy_ops.x, amplitude=2)
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

    def test_call_multi_type(self, dummy_multi_type_ops, dummy_q, dummy_coupler_q):
        section_1 = dummy_multi_type_ops.x(dummy_q, amplitude=2.0)
        section_2 = dummy_multi_type_ops.x(dummy_coupler_q, amplitude=3.0)
        section_3 = dummy_multi_type_ops.cz(dummy_coupler_q, dummy_q, amplitude=4.0)

        assert section_1 == tsl.section(uid="__x_q0_0").children(
            tsl.reserve_op(signal="/lsg/q0/drive"),
            tsl.reserve_op(signal="/lsg/q0/measure"),
            tsl.reserve_op(signal="/lsg/q0/acquire"),
            tsl.play_pulse_op(signal="/lsg/q0/drive", amplitude=2.0),
        )

        assert section_2 == tsl.section(uid="__x_q1_0").children(
            tsl.reserve_op(signal="/lsg/q1/drive"),
            tsl.reserve_op(signal="/lsg/q1/measure"),
            tsl.reserve_op(signal="/lsg/q1/acquire"),
            tsl.play_pulse_op(signal="/lsg/q1/drive", amplitude=3.0),
        )

        assert section_3 == tsl.section(uid="__cz_q1_q0_0").children(
            tsl.reserve_op(signal="/lsg/q1/drive"),
            tsl.reserve_op(signal="/lsg/q1/measure"),
            tsl.reserve_op(signal="/lsg/q1/acquire"),
            tsl.reserve_op(signal="/lsg/q0/drive"),
            tsl.reserve_op(signal="/lsg/q0/measure"),
            tsl.reserve_op(signal="/lsg/q0/acquire"),
            tsl.play_pulse_op(signal="/lsg/q1/drive", amplitude=4.0),
            tsl.play_pulse_op(signal="/lsg/q0/drive", amplitude=4.0),
        )

    def test_call_with_foreign_qubits(self, dummy_ops):
        q = ForeignQubit(uid="q0")
        with pytest.raises(TypeError) as err:
            dummy_ops.x(q, amplitude=1.0)
        assert str(err.value) == (
            "Quantum operation 'x' was passed the following qubits that are"
            " not of a supported qubit type: q0. The supported qubit types are:"
            " Transmon."
        )

    def test_call_with_foreign_qubits_for_multi_type(
        self,
        dummy_multi_type_ops,
        dummy_q,
    ):
        q0 = ForeignQubit(uid="q0")
        q1 = ForeignQubit(uid="q1")

        with pytest.raises(TypeError) as err:
            dummy_multi_type_ops.x(q0, amplitude=1.0)
        assert str(err.value) == (
            "Quantum operation 'x' was passed the following qubits that are"
            " not of a supported qubit type: q0. The supported qubit types are:"
            " DummyCoupler, Transmon."
        )

        with pytest.raises(TypeError) as err:
            dummy_multi_type_ops.cz(q0, dummy_q)
        assert str(err.value) == (
            "Quantum operation 'cz' was passed the following qubits that are"
            " not of a supported qubit type: q0. The supported qubit types are:"
            " DummyCoupler, Transmon."
        )

        with pytest.raises(TypeError) as err:
            dummy_multi_type_ops.cz(q0, q1)
        assert str(err.value) == (
            "Quantum operation 'cz' was passed the following qubits that are"
            " not of a supported qubit type: q0, q1. The supported qubit types are:"
            " DummyCoupler, Transmon."
        )

    def test_functools_partial(self, dummy_ops, dummy_q):
        x_with_amp = functools.partial(dummy_ops.x, amplitude=1.5)

        section = x_with_amp(dummy_q)

        assert section == tsl.section(uid="__x_q0_0").children(
            tsl.reserve_op(signal="/lsg/q0/drive"),
            tsl.reserve_op(signal="/lsg/q0/measure"),
            tsl.reserve_op(signal="/lsg/q0/acquire"),
            tsl.play_pulse_op(signal="/lsg/q0/drive", amplitude=1.5),
        )

    def test_omit_section(self, dummy_ops, dummy_q):
        q = dummy_q

        with dsl.section(uid="top") as section:
            result = dummy_ops.x.omit_section(q, amplitude=1.0)

        assert section == tsl.section(uid="top").children(
            tsl.play_pulse_op(),
        )
        assert result is None

        with pytest.raises(LabOneQException) as err:
            dummy_ops.x.omit_section(q, amplitude=1.0)
        assert str(err.value) == "Must be in a section context"

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
