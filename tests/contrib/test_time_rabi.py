# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Smoke-tests for the time_rabi experiments."""

import numpy as np
import pytest

from laboneq_applications.contrib.experiments import time_rabi


# The emulated acquisition data is constant, so the cosine fit in the
# analysis yields a zero frequency and no pi-pulse lengths can be extracted.
@pytest.mark.filterwarnings(
    "ignore:The frequency of the cosine function is zero:UserWarning"
)
class TestTimeRabi:
    def test_time_rabi(self, two_tunable_transmon_platform):
        platform = two_tunable_transmon_platform
        qpu = platform.qpu
        qubit_uids = qpu.quantum_element_uids
        options = time_rabi.experiment_workflow.options()
        options.do_analysis(True)
        session = platform.session(do_emulation=True)
        wf = time_rabi.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qubit_uids,
            lengths=[
                np.arange(0.1e-6, 1.05e-6, 0.1e-6),
                np.arange(0.1e-6, 1.05e-6, 0.1e-6),
            ],
            options=options,
        )
        wf.run()
