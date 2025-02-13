# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Device setups for tests."""

import pytest
from laboneq.dsl.quantum.qpu import QuantumPlatform

from laboneq_applications.qpu_types.tunable_transmon import (
    demo_platform as demo_platform_transmons,
)
from laboneq_applications.qpu_types.twpa import demo_platform as demo_platform_twpas


@pytest.fixture
def single_tunable_transmon_platform() -> QuantumPlatform:
    """Return a single tunable transmon device setup and its qubits."""
    return demo_platform_transmons(1)


@pytest.fixture
def two_tunable_transmon_platform() -> QuantumPlatform:
    """Return a single tunable transmon device setup and its qubits."""
    return demo_platform_transmons(2)


@pytest.fixture
def single_twpa_platform() -> QuantumPlatform:
    """Return a single-TWPA device setup and its TWPA."""
    return demo_platform_twpas(1)
