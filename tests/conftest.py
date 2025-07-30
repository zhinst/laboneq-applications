# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Configuration for pytest."""

from pathlib import Path

import pytest

__all__ = [
    "reset_uids",  # autouse fixture
    "single_tunable_transmon_platform",  # fixture
    "single_twpa_platform",  # fixture
    "two_tunable_transmon_platform",  # fixture
]

from tests.helpers.demo_qpus import (
    single_tunable_transmon_platform,
    single_twpa_platform,
    two_tunable_transmon_platform,
)
from tests.helpers.dsl import ExpectedDSLStructure, reset_uids


def pytest_assertrepr_compare(config, op, left, right):
    """Enable friendlier comparison messages for DSL assertions."""
    if isinstance(left, ExpectedDSLStructure):
        return left.compare(right)
    if isinstance(right, ExpectedDSLStructure):
        return right.compare(left)
    return None


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--allow-external-files",
        action="store_true",
        default=False,
        help="Allow tests to generate external files without failing.",
    )


@pytest.fixture(scope="session", autouse=True)
def _check_generated_files_root(request: pytest.FixtureRequest):
    """A session wide fixture to check that no external files are generated.

    The fixture only checks the root directory.
    """
    if request.config.getoption("--allow-external-files"):
        yield
    else:
        root_path = Path.cwd()
        # NOTE: Any file check failed in CI image, restricted set should be enough for
        # now, As the main culprit for this is LabOne Q pulse sheets and images.
        disallowed_file_extensions = [
            ".html",
            ".svg",
            ".png",
            ".log",
            ".json",
            ".jsonl",
        ]
        before_files = set()
        for extension in disallowed_file_extensions:
            before_files.update(set(root_path.glob(f"*{extension}")))
        yield
        after_files = set()
        for extension in disallowed_file_extensions:
            after_files.update(set(root_path.glob(f"*{extension}")))
        generated_files = after_files - before_files
        if generated_files:
            msg = (
                f"Test generated unexpected files: {[str(f) for f in generated_files]}"
                "To disable the error, run pytest with '--allow-external-files'."
            )
            pytest.fail(msg)
