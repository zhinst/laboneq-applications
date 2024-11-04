"""Device setups for tests."""

import pytest
from laboneq.dsl.quantum.qpu import QuantumPlatform

from laboneq_applications.qpu_types.tunable_transmon import demo_platform


@pytest.fixture()
def single_tunable_transmon_platform() -> QuantumPlatform:
    """Return a single tunable transmon device setup and its qubits."""
    return demo_platform(1)


@pytest.fixture()
def two_tunable_transmon_platform() -> QuantumPlatform:
    """Return a single tunable transmon device setup and its qubits."""
    return demo_platform(2)
