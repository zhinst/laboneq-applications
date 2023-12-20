import pytest
from laboneq.simple import Qubit

from laboneq_library.automatic_tuneup.tuneup.analyzer import MockAnalyzer
from laboneq_library.automatic_tuneup.tuneup.params import SweepParams
from laboneq_library.automatic_tuneup.tuneup.qubit_config import (
    QubitConfig,
    QubitConfigs,
)


class TestQubitConfig:
    parameter = SweepParams()
    qubit = Qubit()
    analyzer = MockAnalyzer()
    qubit_config = QubitConfig(
        parameter=parameter,
        qubit=qubit,
        analyzer=analyzer,
        update_key="resonance_frequency",
    )

    def test_init(self):
        assert self.qubit_config.parameter == self.parameter
        assert self.qubit_config.qubit == self.qubit
        assert self.qubit_config.analyzer == self.analyzer

    def test_post_init(self):
        assert self.qubit_config.need_to_verify is True
        assert self.qubit_config.update_key == "resonance_frequency"
        assert self.qubit_config.pulses is None
        assert not self.qubit_config._update_key_in_user_defined
        assert self.qubit_config._analyzed_result is None
        assert self.qubit_config._update_value is None
        assert not self.qubit_config._verified

    def test_invaid_update_key(self):
        with pytest.raises(ValueError):
            self.qubit_config.update_key = "invalid_key"

    def test_valid_update_key(self):
        self.qubit_config.update_key = "resonance_frequency"
        assert self.qubit_config.update_key == "resonance_frequency"

    def test_update_qubit(self):
        self.qubit.parameters.drive_lo_frequency = 1.23e9
        self.qubit_config._analyzed_result = 1e9
        self.qubit_config.update_qubit()
        assert self.qubit_config._update_value == 2.23e9
        assert self.qubit_config.qubit.parameters.resonance_frequency == 2.23e9

    def test_update_qubit_user_defined_parameters(self):
        self.qubit.parameters.user_defined = dict(prop1=0)
        self.qubit_config.update_key = "prop1"
        self.qubit_config._analyzed_result = 1e9
        self.qubit_config.update_qubit()
        assert self.qubit_config._update_value == 1e9
        assert self.qubit_config.qubit.parameters.user_defined["prop1"] == 1e9


class TestQubitConfigs:
    parameter1 = SweepParams()
    parameter2 = SweepParams()
    qubit1 = Qubit()
    qubit2 = Qubit()
    analyzer = MockAnalyzer()
    qubit_config1 = QubitConfig(
        parameter=parameter1,
        qubit=qubit1,
        analyzer=analyzer,
        update_key="resonance_frequency",
    )
    qubit_config2 = QubitConfig(
        parameter=parameter2,
        qubit=qubit2,
        analyzer=analyzer,
        update_key="resonance_frequency",
    )
    qubit_configs = QubitConfigs([qubit_config1, qubit_config2])

    def test_get_qubits(self):
        qubits = self.qubit_configs.get_qubits()
        assert set(qubits) == {self.qubit1, self.qubit2}

    def test_get_parameters(self):
        parameters = self.qubit_configs.get_parameters()
        assert parameters == [self.parameter1, self.parameter2]

    def test_get_need_to_verify(self):
        assert self.qubit_configs.all_verified() is False
        need_to_verify = self.qubit_configs.get_need_to_verify()
        assert need_to_verify == [self.qubit_config1, self.qubit_config2]

        self.qubit_configs[0].need_to_verify = False
        need_to_verify = self.qubit_configs.get_need_to_verify()
        assert need_to_verify == [self.qubit_config2]

        self.qubit_configs[0].need_to_verify = False
        self.qubit_configs[1].need_to_verify = False
        need_to_verify = self.qubit_configs.get_need_to_verify()
        assert not need_to_verify

    def test_all_verified(self):

        self.qubit_configs[0].need_to_verify = True
        self.qubit_configs[1].need_to_verify = True
        self.qubit_configs[0]._verified = True
        self.qubit_configs[1]._verified = True
        assert self.qubit_configs.get_need_to_verify() == [
            self.qubit_configs[0],
            self.qubit_configs[1],
        ]
        assert self.qubit_configs.all_verified() is True

        self.qubit_configs[0].need_to_verify = False
        self.qubit_configs[1]._verified = False
        assert self.qubit_configs.get_need_to_verify() == [self.qubit_configs[1]]
        assert self.qubit_configs.all_verified() is False

        self.qubit_configs[1].need_to_verify = False
        assert self.qubit_configs.get_need_to_verify() == []
        assert self.qubit_configs.all_verified() is True

        self.qubit_configs[1].need_to_verify = True
        self.qubit_configs[0]._verified = False
        self.qubit_configs[1]._verified = True
        assert self.qubit_configs.all_verified() is True

    def test_copy(self):
        qubit_configs_copy = self.qubit_configs.copy()
        assert isinstance(qubit_configs_copy, QubitConfigs)
        assert len(qubit_configs_copy) == 2
        assert isinstance(qubit_configs_copy[0], QubitConfig)
        assert isinstance(qubit_configs_copy[1], QubitConfig)
        assert qubit_configs_copy[0].need_to_verify is False
        assert qubit_configs_copy[1]._verified is False
