import pytest
from laboneq.simple import Qubit

from laboneq_library.automatic_tuneup.tuneup.analyzer import MockAnalyzer
from laboneq_library.automatic_tuneup.tuneup.params import SweepParams
from laboneq_library.automatic_tuneup.tuneup.qubit_config import (
    QubitConfig,
    QubitConfigs,
)


@pytest.fixture(scope="function")
def qubit_config():
    parameter = SweepParams()
    qubit = Qubit()
    analyzer = MockAnalyzer()
    return QubitConfig(parameter=parameter, qubit=qubit, analyzer=analyzer,update_key="resonance_frequency")


def test_qubit_config(qubit_config):
    assert isinstance(qubit_config.parameter, SweepParams)
    assert isinstance(qubit_config.qubit, Qubit)
    assert isinstance(qubit_config.analyzer, MockAnalyzer)
    assert qubit_config.need_to_verify is True
    assert qubit_config.update_key == "resonance_frequency"
    assert qubit_config.pulses is None
    assert qubit_config._update_key_in_user_defined is False
    assert qubit_config._analyzed_result is None
    assert qubit_config._update_value is None
    assert qubit_config._verified is False

    assert qubit_config._update_key_in_user_defined is False

    with pytest.raises(ValueError):
        qubit_config.update_key = "invalid_key"



@pytest.fixture(scope="function")
def qubit_configs():
    parameter = SweepParams()
    qubit1 = Qubit()
    qubit2 = Qubit()
    analyzer = MockAnalyzer()
    qubit_config1 = QubitConfig(parameter=parameter, qubit=qubit1, analyzer=analyzer, update_key="resonance_frequency")
    qubit_config2 = QubitConfig(parameter=parameter, qubit=qubit2, analyzer=analyzer, update_key="resonance_frequency")
    return QubitConfigs([qubit_config1, qubit_config2])


def test_qubit_configs(qubit_configs):
    assert isinstance(qubit_configs, QubitConfigs)
    assert len(qubit_configs) == 2
    assert isinstance(qubit_configs[0], QubitConfig)
    assert isinstance(qubit_configs[1], QubitConfig)

    qubits = qubit_configs.get_qubits()
    assert isinstance(qubits[0], Qubit)
    assert isinstance(qubits[1], Qubit)

    parameters = qubit_configs.get_parameters()
    assert isinstance(parameters[0], SweepParams)
    assert isinstance(parameters[1], SweepParams)

    assert qubit_configs.all_verified() is False

    need_to_verify = qubit_configs.get_need_to_verify()
    assert isinstance(need_to_verify[0], QubitConfig)
    assert isinstance(need_to_verify[1], QubitConfig)

    qubit_configs[0].need_to_verify = True
    qubit_configs[1].need_to_verify = True
    qubit_configs[0]._verified = True
    qubit_configs[1]._verified = True
    assert qubit_configs.get_need_to_verify() == [qubit_configs[0],qubit_configs[1]]
    assert qubit_configs.all_verified() is True

    qubit_configs[0].need_to_verify = False
    qubit_configs[1]._verified = False
    assert qubit_configs.get_need_to_verify() == [qubit_configs[1]]
    assert qubit_configs.all_verified() is False

    qubit_configs[1].need_to_verify = False
    assert qubit_configs.get_need_to_verify() == []
    assert qubit_configs.all_verified() is True

    qubit_configs[1].need_to_verify = True
    qubit_configs[0]._verified = False
    qubit_configs[1]._verified = True
    assert qubit_configs.all_verified() is True

    qubit_configs_copy = qubit_configs.copy()
    assert isinstance(qubit_configs_copy, QubitConfigs)
    assert len(qubit_configs_copy) == 2
    assert isinstance(qubit_configs_copy[0], QubitConfig)
    assert isinstance(qubit_configs_copy[1], QubitConfig)
    assert qubit_configs_copy[0].need_to_verify is False
    assert qubit_configs_copy[1]._verified is False
