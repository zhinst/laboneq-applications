# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.typing."""

import laboneq_applications.typing


class TestTypingAPI:
    def test_api(self):
        assert laboneq_applications.typing.Qubits
        assert laboneq_applications.typing.QubitSweepPoints
