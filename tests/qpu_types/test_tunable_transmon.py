""" Tests for laboneq_library.qpu_types.tunable_transmon. """

from laboneq_library.qpu_types.tunable_transmon import (
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
    TunableTransmonOperations,
)


class TestTunableTransmonQubit:
    def test_create(self):
        q = TunableTransmonQubit()

        assert isinstance(q.parameters, TunableTransmonQubitParameters)


class TestTunableTransmonParameters:
    def test_create(self):
        p = TunableTransmonQubitParameters()

        assert p.readout_range_out == 5
        assert p.readout_range_in == 10


class TestTunableTransmonOperations:
    def test_create(self):
        qops = TunableTransmonOperations()

        assert qops.QUBIT_TYPE is TunableTransmonQubit
        assert qops.TRANSITIONS == ("ge", "ef")
