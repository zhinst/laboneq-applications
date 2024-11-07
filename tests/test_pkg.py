# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Test for the laboneq_applications package API."""

import laboneq_applications


def test_version():
    assert laboneq_applications.__version__ == "1.0.0dev0"
