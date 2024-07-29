"""Configuration for pytest."""

__all__ = [
    "reset_uids",  # autouse fixture
    "single_tunable_transmon",  # fixture
    "two_tunable_transmon",  # fixture
]

from tests.helpers.demo_qpus import (
    single_tunable_transmon,
    two_tunable_transmon,
)
from tests.helpers.dsl import ExpectedDSLStructure, reset_uids


def pytest_assertrepr_compare(config, op, left, right):
    """Enable friendlier comparison messages for DSL assertions."""
    if isinstance(left, ExpectedDSLStructure):
        return left.compare(right)
    if isinstance(right, ExpectedDSLStructure):
        return right.compare(left)
    return None
