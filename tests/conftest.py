"""Configuration for pytest."""

__all__ = [
    "reset_uids",  # autouse fixture
    "single_tunable_transmon",  # fixture
    "two_tunable_transmon",  # fixture
    "simple_device_setup",  # fixture
    "simple_experiment",  # fixture
    "simple_session",  # fixture
]

from tests.helpers.device_setups import (
    single_tunable_transmon,
    two_tunable_transmon,
)
from tests.helpers.dsl import ExpectedDSLStructure, reset_uids
from tests.helpers.simple_setup import (
    simple_device_setup,
    simple_experiment,
    simple_session,
)


def pytest_assertrepr_compare(config, op, left, right):
    """Enable friendlier comparison messages for DSL assertions."""
    if isinstance(left, ExpectedDSLStructure):
        return left.compare(right)
    if isinstance(right, ExpectedDSLStructure):
        return right.compare(left)
    return None
