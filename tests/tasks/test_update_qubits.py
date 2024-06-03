import copy

import pytest

from laboneq_applications.qpu_types.tunable_transmon import TunableTransmonQubit
from laboneq_applications.tasks.update_qubits import update_qubits
from laboneq_applications.workflow.engine import Workflow


class TestUpdateSingleQubits:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.q0 = TunableTransmonQubit("q0")
        self.q1 = TunableTransmonQubit("q1")

    def test_update_standalone(self):
        update_qubits(
            [
                (self.q0, {"drive_parameters_ge": {"amplitude": 1.0}}),
                (self.q1, {"drive_parameters_ef": {"amplitude": 1.0}}),
            ],
        )
        assert self.q0.parameters.drive_parameters_ge == {"amplitude": 1.0}
        assert self.q1.parameters.drive_parameters_ef == {"amplitude": 1.0}

        update_qubits(
            [
                (self.q0, {"readout_parameters.length": 100e-9}),
                (self.q1, {"readout_parameters.length": 100e-9}),
            ],
        )

        assert self.q0.parameters.readout_parameters["length"] == 100e-9
        assert self.q1.parameters.readout_parameters["length"] == 100e-9

        # just to make sure that the previously updated parameters are not changed
        assert self.q0.parameters.drive_parameters_ge == {"amplitude": 1.0}
        assert self.q1.parameters.drive_parameters_ef == {"amplitude": 1.0}

        update_qubits(
            [
                (self.q0, {"readout_amplitude": 10}),
                (self.q1, {"readout_amplitude": 11}),
            ],
        )

        assert self.q0.parameters.readout_amplitude == 10
        assert self.q1.parameters.readout_amplitude == 11

    def test_update_fail(self):
        original_q0_params = copy.deepcopy(self.q0.parameters)
        original_q1_params = copy.deepcopy(self.q1.parameters)
        with pytest.raises(ValueError) as err:
            update_qubits(
                [
                    (
                        self.q0,
                        {
                            "drive_parameters_ge": {"amplitude": 1.0},
                            "non_existing_param": 1.0,
                        },
                    ),
                    (self.q1, {"drive_parameters_ef": {"amplitude": 1.0}}),
                ],
            )
        invalid_params = [(self.q0.uid, ["non_existing_param"])]
        assert str(err.value) == (
            f"Qubits were not updated due to the following parameters "
            f"being invalid: {invalid_params}"
        )

        assert self.q0.parameters == original_q0_params
        assert self.q1.parameters == original_q1_params

        with pytest.raises(ValueError) as err:
            update_qubits(
                [
                    (self.q0, {"drive_parameters_ge": {"amplitude": 1.0}}),
                    (
                        self.q1,
                        {
                            "drive_parameters_ef": {"amplitude": 1.0},
                            "non_existing_param": 1.0,
                        },
                    ),
                ],
            )
        invalid_params = [(self.q1.uid, ["non_existing_param"])]
        assert str(err.value) == (
            f"Qubits were not updated due to the following parameters "
            f"being invalid: {invalid_params}"
        )
        assert self.q0.parameters == original_q0_params
        assert self.q1.parameters == original_q1_params

        with pytest.raises(ValueError) as err:
            update_qubits(
                [
                    (
                        self.q0,
                        {
                            "drive_parameters_ge": {"amplitude": 1.0},
                            "non_existing_param": 1.0,
                        },
                    ),
                    (
                        self.q1,
                        {
                            "drive_parameters_ef": {"amplitude": 1.0},
                            "non_existing_param": 1.0,
                        },
                    ),
                ],
            )
        invalid_params = [
            (self.q0.uid, ["non_existing_param"]),
            (self.q1.uid, ["non_existing_param"]),
        ]
        assert str(err.value) == (
            f"Qubits were not updated due to the following parameters "
            f"being invalid: {invalid_params}"
        )
        assert self.q0.parameters == original_q0_params
        assert self.q1.parameters == original_q1_params

    def test_update_task(self):
        params_update_0 = {
            "drive_parameters_ge": {
                "amplitude_pi": 1.0,
                "amplitude_pi2": 1.0,
                "length": 50e-9,
                "pulse": {"function": "drag", "beta": 0, "sigma": 0.25},
            },
        }
        params_update_1 = {
            "drive_parameters_ef": {
                "amplitude_pi": 1.0,
                "amplitude_pi2": 1.0,
                "length": 50e-9,
                "pulse": {"function": "drag", "beta": 0, "sigma": 0.25},
            },
        }
        with Workflow() as wf:
            update_qubits(
                [(self.q0, params_update_0), (self.q1, params_update_1)],
            )
            update_qubits(
                [
                    (self.q0, {"readout_parameters.length": 100e-9}),
                    (self.q1, {"readout_parameters.length": 100e-9}),
                ],
            )
        run = wf.run()

        assert len(run.tasklog) == 1
        assert "update_qubits" in run.tasklog
        assert (
            self.q0.parameters.drive_parameters_ge
            == params_update_0["drive_parameters_ge"]
        )

        assert (
            self.q1.parameters.drive_parameters_ef
            == params_update_1["drive_parameters_ef"]
        )

        assert self.q0.parameters.readout_parameters["length"] == 100e-9
        assert self.q1.parameters.readout_parameters["length"] == 100e-9
