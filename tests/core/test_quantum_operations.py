""" Tests for laboneq_library.core.quantum_operations. """


import numpy as np
import numpy.testing
import pytest

from laboneq.simple import (
    Experiment,
    Section,
    SectionAlignment,
    SweepParameter,
    Transmon,
)
from laboneq.dsl.experiment import builtins

from laboneq_library.core.quantum_operations import (
    QuantumOperations,
    current_quantum_operations,
    qop,
    quantum_operation,
    dsl,
)


class DummyOperations(QuantumOperations):
    """Dummy operations for testing."""

    QUBIT_TYPE = Transmon

    @quantum_operation
    def x(q, amplitude):
        """A dummy quantum operation."""
        pulse = dsl.pulse_library.const()
        dsl.play(q.signals["drive"], pulse, amplitude=amplitude)


@pytest.fixture
def dummy_ops():
    return DummyOperations()


@pytest.fixture
def dummy_q():
    return Transmon(
        uid="q0",
        signals={
            "drive": "/lsg/drive",
            "measure": "/lsg/measure",
            "acquire": "/lsg/acquire",
        },
    )


class TestQuantumOperationDecorator:
    def test_decorator(self):
        @quantum_operation
        def x(q, amplitude):
            pass

        assert x._quantum_op


class TestCurrentQuantumOperations:
    def test_current_operations(self):
        qops_1 = DummyOperations()
        qops_2 = DummyOperations()

        with pytest.raises(RuntimeError) as exc:
            current_quantum_operations()
        assert str(exc.value) == "No quantum operations context is currently set."

        with qops_1:
            assert qops_1 is current_quantum_operations()
            with qops_2:
                assert qops_2 is current_quantum_operations()
            assert qops_1 is current_quantum_operations()

        with pytest.raises(RuntimeError) as exc:
            current_quantum_operations()
        assert str(exc.value) == "No quantum operations context is currently set."


class TestQop:
    def test_getattr(self):
        qops_1 = DummyOperations()
        qops_2 = DummyOperations()

        with pytest.raises(RuntimeError) as exc:
            qop.x
        assert str(exc.value) == "No quantum operations context is currently set."

        with qops_1:
            assert qops_1.x is qop.x
            with qops_2:
                assert qops_2.x is qop.x
            assert qops_1.x is qop.x

        with pytest.raises(RuntimeError) as exc:
            qop.x
        assert str(exc.value) == "No quantum operations context is currently set."

    def test_getitem(self):
        qops_1 = DummyOperations()
        qops_2 = DummyOperations()

        with pytest.raises(RuntimeError) as exc:
            qop["x"]
        assert str(exc.value) == "No quantum operations context is currently set."

        with qops_1:
            assert qops_1.x is qop["x"]
            with qops_2:
                assert qops_2.x is qop["x"]
            assert qops_1.x is qop["x"]

        with pytest.raises(RuntimeError) as exc:
            qop["x"]
        assert str(exc.value) == "No quantum operations context is currently set."


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
    def simple_exp(self, q, amplitudes, shots, prep=None):
        with dsl.acquire_loop_rt(count=shots):
            with dsl.sweep(parameter=SweepParameter(values=amplitudes)) as amplitude:
                if prep:
                    prep(q)
                qop.x(q, amplitude)

    def single_op_exp(self, q, op_name):
        qop[op_name](q)

    def test_create(self, dummy_ops):
        assert dummy_ops.QUBIT_TYPE is Transmon

    def test_quantum_operation(self, dummy_ops, dummy_q):
        section = dummy_ops.x(dummy_q, amplitude=1.5)
        assert type(section) is Section

    def test_context_manager(self, dummy_ops):
        with dummy_ops:
            assert current_quantum_operations() is dummy_ops

    def test_getattr(self, dummy_ops):
        assert dummy_ops.x.op is dummy_ops.BASE_OPS["x"]
        with pytest.raises(AttributeError) as exc:
            dummy_ops.y
        assert str(exc.value) == "'DummyOperations' object has no attribute 'y'"

    def test_getitem(self, dummy_ops):
        assert dummy_ops["x"].op is dummy_ops.BASE_OPS["x"]
        with pytest.raises(KeyError) as exc:
            dummy_ops["y"]
        assert str(exc.value) == "'y'"

    def test_contains(self, dummy_ops):
        assert "x" in dummy_ops
        assert "y" not in dummy_ops

    def test_dir(self, dummy_ops):
        public_attrs = [attr for attr in dir(dummy_ops) if not attr.startswith("_")]
        assert public_attrs == [
            "BASE_OPS",
            "QUBIT_TYPE",
            "build",
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
        exp = dummy_ops.build(self.simple_exp, dummy_q, np.linspace(0.1, 1, 10), 5)

        assert isinstance(exp, Experiment)

        [acquire] = exp.sections
        assert acquire.alignment == SectionAlignment.LEFT

        [sweep] = acquire.children
        assert sweep.alignment == SectionAlignment.LEFT

        [param] = sweep.parameters
        np.testing.assert_almost_equal(param.values, np.linspace(0.1, 1, 10))

        [x] = sweep.children
        assert x.uid == "x_0"
        assert x.alignment == SectionAlignment.LEFT

        [pulse] = x.children
        assert pulse.signal == "/lsg/drive"
        assert pulse.amplitude is param

    def test_build_with_nested_ops(self, dummy_ops, dummy_q):
        def x_amp_2(q):
            qop.x(q, amplitude=2.0)

        dummy_ops.register(x_amp_2)

        exp = dummy_ops.build(self.single_op_exp, dummy_q, "x_amp_2")

        [op] = exp.sections
        assert op.uid == "x_amp_2_0"

        [x_0] = op.children
        assert x_0.uid == "x_0"

        [pulse] = x_0.children
        assert pulse.signal == "/lsg/drive"
        assert pulse.amplitude == 2.0


class TestOperation:
    def test_call(self, dummy_ops, dummy_q):
        with dummy_ops:
            section_1 = dummy_ops.x(dummy_q, amplitude=2.0)
            section_2 = dummy_ops.x(dummy_q, amplitude=3.0)

        assert section_1.uid == "x_0"
        [pulse_1] = section_1.children
        assert pulse_1.signal == "/lsg/drive"
        assert pulse_1.amplitude == 2.0

        assert section_2.uid == "x_1"
        [pulse_2] = section_2.children
        assert pulse_2.signal == "/lsg/drive"
        assert pulse_2.amplitude == 3.0

    def test_partial(self, dummy_ops, dummy_q):
        x_with_amp = dummy_ops.x.partial(amplitude=1.5)

        with dummy_ops:
            section = x_with_amp(dummy_q)

        assert section.uid == "x_0"
        [pulse] = section.children
        assert pulse.signal == "/lsg/drive"
        assert pulse.amplitude == 1.5

    def test_op(self, dummy_ops):
        assert dummy_ops.x.op is DummyOperations.BASE_OPS["x"]

    def test_src(self, dummy_ops):
        assert dummy_ops.x.src == "\n".join(
            [
                "@quantum_operation",
                "def x(q, amplitude):",
                '    """A dummy quantum operation."""',
                "    pulse = dsl.pulse_library.const()",
                '    dsl.play(q.signals["drive"], pulse, amplitude=amplitude)',
                "",
            ]
        )
