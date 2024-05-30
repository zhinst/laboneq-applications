""" Tests for laboneq_applications.experiments amplitude Rabi. """


from laboneq.simple import (
    LinearSweepParameter,
    Session,
    SweepParameter,
)

import tests.helpers.dsl as tsl
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.quantum_operations import dsl
from laboneq_applications.experiments_legacy.qubit_calibration_experiments import (
    AmplitudeRabi,
)
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonOperations,
)


class AmplitudeRabiOps(AmplitudeRabi):
    @staticmethod
    @qubit_experiment
    def rabi_exp(qop, qubits, qubit_amplitudes, count=10, transition="ge"):
        with dsl.acquire_loop_rt(count=count):
            for q, q_amplitudes in zip(qubits, qubit_amplitudes):
                # rabi:
                with dsl.sweep(
                    uid=f"amps_{q.uid}",
                    parameter=SweepParameter(f"amplitude_{q.uid}", q_amplitudes),
                ) as amplitude:
                    qop.prepare_state(q, transition[0])
                    qop.x180(q, amplitude=amplitude, transition=transition)
                    qop.measure(q, "result")
                    qop.passive_reset(q)

                # calibration measurements:
                with dsl.section(
                    uid=f"cal_{q.uid}",
                ):
                    for state in transition:
                        qop.prepare_state(q, state)
                        qop.measure(q, f"cal_state_{state}")
                        qop.passive_reset(q)

    def define_experiment(self):
        # extract old parameters
        qubits = self.qubits
        qubit_amplitudes = [
            self.sweep_parameters_dict[q.uid][0].values for q in self.qubits
        ]
        count = self.acquisition_metainfo["count"]
        transition = self.transition_to_calibrate

        # build experiment
        qops = TunableTransmonOperations()
        exp = self.rabi_exp(
            qops,
            qubits,
            qubit_amplitudes,
            count=count,
            transition=transition,
        )

        # return the experiment
        self.experiment = exp


class TestAmplitudeRabi:
    def reserve_ops(self, q):
        """Return the expected reserve operations for the given qubit."""
        return [
            tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/drive"),
            tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/drive_ef"),
            tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/measure"),
            tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/acquire"),
            tsl.reserve_op(signal=f"/logical_signal_groups/{q.uid}/flux"),
        ]

    def test_define_experiment(self, single_tunable_transmon):
        [q0] = single_tunable_transmon.qubits
        q0_setup = single_tunable_transmon.setup

        session = Session(q0_setup)
        session.connect(do_emulation=True)

        qubit_temporary_values = []
        sweep_parameters_dict = {}
        transition_to_calibrate = "ge"  # or "ef"
        preparation_type = "wait"  # or "active_reset"

        for qubit in [q0]:
            qubit_temporary_values += [
                (qubit, "reset_delay_length", 200e-6),
            ]

            pi_amp = (
                qubit.parameters.drive_parameters_ef["amplitude_pi"]
                if transition_to_calibrate == "ef"
                else qubit.parameters.drive_parameters_ge["amplitude_pi"]
            )
            swp_end = pi_amp + 0.1
            sweep_parameters_dict[qubit.uid] = [
                LinearSweepParameter(
                    f"amps_{qubit.uid}",
                    0,
                    swp_end,
                    21,
                    "Amplitude Scaling",
                ),
            ]

        experiment_metainfo = {
            "cal_states": transition_to_calibrate,
            "transition_to_calibrate": transition_to_calibrate,
            "preparation_type": preparation_type,
        }

        rabi = AmplitudeRabiOps(
            [q0],
            session,
            q0_setup,
            experiment_metainfo=experiment_metainfo,
            acquisition_metainfo={"count": 2**0},
            sweep_parameters_dict=sweep_parameters_dict,
            qubit_temporary_values=qubit_temporary_values,
            do_analysis=True,
            update=False,
            save=False,
            run=False,
        )
        rabi.define_experiment()
        rabi.compile_experiment()

        assert rabi.experiment == tsl.experiment().children(
            tsl.acquire_loop_rt().children(
                tsl.sweep(uid="amps_q0").children(
                    tsl.section(uid="prepare_state_q0_0").children(
                        self.reserve_ops(q0),
                    ),
                    tsl.section(uid="x180_q0_0").children(
                        self.reserve_ops(q0),
                        tsl.play_pulse_op(),
                    ),
                    tsl.section(uid="measure_q0_0").children(
                        self.reserve_ops(q0),
                        tsl.play_pulse_op(),
                        tsl.acquire_op(),
                    ),
                    tsl.section(uid="passive_reset_q0_0").children(
                        self.reserve_ops(q0),
                        tsl.delay_op(),
                    ),
                ),
                tsl.section(uid="cal_q0").children(
                    tsl.section(uid="prepare_state_q0_1").children(
                        self.reserve_ops(q0),
                    ),
                    tsl.section(uid="measure_q0_1").children(
                        self.reserve_ops(q0),
                        tsl.play_pulse_op(),
                        tsl.acquire_op(),
                    ),
                    tsl.section(uid="passive_reset_q0_1").children(
                        self.reserve_ops(q0),
                        tsl.delay_op(),
                    ),
                    tsl.section(uid="prepare_state_q0_2").children(
                        self.reserve_ops(q0),
                        tsl.section(uid="x180_q0_1").children(
                            self.reserve_ops(q0),
                            tsl.play_pulse_op(),
                        ),
                    ),
                    tsl.section(uid="measure_q0_2").children(
                        self.reserve_ops(q0),
                        tsl.play_pulse_op(),
                        tsl.acquire_op(),
                    ),
                    tsl.section(uid="passive_reset_q0_2").children(
                        self.reserve_ops(q0),
                        tsl.delay_op(),
                    ),
                ),
            ),
        )
