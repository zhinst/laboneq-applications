"""Device setups for tests."""

import pytest

from laboneq_applications.qpu_types.tunable_transmon import demo_qpu


@pytest.fixture()
def single_tunable_transmon():
    """Return a single tunable transmon device setup and its qubits."""
    return demo_qpu(1)


@pytest.fixture()
def two_tunable_transmon():
    """Return a single tunable transmon device setup and its qubits."""
    return demo_qpu(2)
