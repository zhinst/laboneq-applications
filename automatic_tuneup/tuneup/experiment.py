from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_results
from laboneq.dsl.parameter import Parameter
from laboneq.simple import *  # noqa: F403

from .tuneup_logging import initialize_logging

logger = initialize_logging()


class TuneUpExperimentFactory(ABC):
    def __init__(
        self,
        parameters: List[LinearSweepParameter],
        qubit: QuantumElement,
        exp_settings: Optional[dict] = None,
        ext_calls: Optional[Callable] = None,
        pulse_storage: Optional[dict] = None,
    ):
        self.parameters = parameters
        self.qubit = qubit
        self.exp_settings = exp_settings
        self.ex_calls = ext_calls
        self.pulse_storage = pulse_storage

    @abstractmethod
    def _gen_experiment(self, parameters: List[LinearSweepParameter]):
        """
        Return a complete experiment object, that can be run by the Scan object
        """
        pass

    @abstractmethod
    def get_updated_value(self, analyzed_result):
        pass

    def set_extra_calibration(
        self,
        drive_range: Optional[int] = None,
        measure_range: Optional[int] = None,
        acquire_range: Optional[int] = None,
    ):
        raise NotImplementedError("Set_extra_calibration not implemented")

    def plot(self, result):
        raise NotImplementedError("Plot not implemented")

    def add_pulse(self, pulse, pulse_name):
        raise NotImplementedError("Add_pulse not implemented")


class MockExp(TuneUpExperimentFactory):
    def __init__(
        self,
        parameters,
        qubit,
        exp_settings=None,
        ext_calls: Callable = None,
        pulse_storage=None,
    ):
        super().__init__(parameters, qubit, exp_settings, ext_calls, pulse_storage)
        self.exp = self._gen_experiment(self.parameters)

    def _gen_experiment(self, parameters):
        """
        Return a complete experiment object, that can be run by the Scan object
        """
        exp_spec = Experiment(
            uid="Mock exp",
            signals=self.qubit.experiment_signals(with_calibration=True),
        )
        return exp_spec

    def get_updated_value(self, analyzed_result):
        pass

    def set_extra_calibration(
        self, drive_range=None, measure_range=None, acquire_range=None
    ):
        pass

    def plot(self, result):
        pass


class RSpecCwFactory(TuneUpExperimentFactory):
    def __init__(
        self,
        parameters,
        qubit,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5},
        ext_calls=None,
        pulse_storage=None,
    ):
        super().__init__(parameters, qubit, exp_settings, ext_calls, pulse_storage)
        self.exp = self._gen_experiment(self.parameters)

    def _gen_experiment(self, parameters):
        freq_sweep = parameters[0]
        exp_settings = self.exp_settings
        exp_spec = Experiment(
            uid="Resonator Spectroscopy",
            signals=self.qubit.experiment_signals(with_calibration=True),
        )

        exp_spec.signals[self.qubit.signals["measure"]].oscillator = Oscillator(
            "measure_osc",
            frequency=freq_sweep,
            modulation_type=ModulationType.HARDWARE,  # HAS TO USE HARDWARE MODULATION FOR SPECTROSCOPY MODE
        )

        with exp_spec.acquire_loop_rt(
            uid="shots",
            count=exp_settings["num_averages"],
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp_spec.sweep(uid="res_freq", parameter=freq_sweep):
                with exp_spec.section(uid="spectroscopy"):
                    exp_spec.acquire(
                        signal=self.qubit.signals["acquire"],
                        handle="res_spec",
                        length=exp_settings["integration_time"],
                    )
                with exp_spec.section(uid="delay", length=1e-6):
                    exp_spec.reserve(signal=self.qubit.signals["measure"])

        return exp_spec

    def plot(self, result):
        plot_results(result, plot_height=4)

    def set_extra_calibration(
        self, drive_range=None, measure_range=None, acquire_range=None
    ):
        if drive_range is not None:
            self.exp.signals[self.qubit.signals["drive"]].range = drive_range
        if measure_range is not None:
            self.exp.signals[self.qubit.signals["measure"]].range = measure_range
        if acquire_range is not None:
            self.exp.signals[self.qubit.signals["acquire"]].range = acquire_range

    def get_updated_value(self, analyzed_result):
        return self.qubit.parameters.readout_lo_frequency + analyzed_result


class ReadoutSpectroscopyCWBiasSweep(RSpecCwFactory):
    def __init__(
        self,
        parameters: Parameter,
        qubit,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5, "slot": 0},
        ext_calls=None,
        pulse_storage=None,
    ):
        self.parameters = parameters
        self.exp_settings = exp_settings
        self.qubit = qubit
        if ext_calls is None:
            raise ValueError("ext_calls must be defined for this experiment")
        self.ext_call = ext_calls
        self.exp = self._gen_experiment(self.parameters)

    def _gen_experiment(self, parameters):
        freq_sweep, dc_volt_sweep = parameters
        exp_settings = self.exp_settings
        exp_spec = Experiment(
            uid="Resonator Spectroscopy",
            signals=self.qubit.experiment_signals(with_calibration=True),
        )

        exp_spec.signals[self.qubit.signals["measure"]].oscillator = Oscillator(
            "measure_osc",
            frequency=freq_sweep,
            modulation_type=ModulationType.HARDWARE,  # HAS TO USE HARDWARE MODULATION FOR SPECTROSCOPY MODE
        )

        with exp_spec.sweep(uid="dc_volt_sweep", parameter=dc_volt_sweep):
            exp_spec.call(self.ext_call, qubit_uid=0, voltage=dc_volt_sweep)
            with exp_spec.acquire_loop_rt(
                uid="shots",
                count=exp_settings["num_averages"],
                acquisition_type=AcquisitionType.SPECTROSCOPY,
            ):
                with exp_spec.sweep(uid="res_freq", parameter=freq_sweep):
                    with exp_spec.section(uid="spectroscopy"):
                        exp_spec.acquire(
                            signal=self.qubit.signals["acquire"],
                            handle="res_spec",
                            length=exp_settings["integration_time"],
                        )
                    with exp_spec.section(uid="delay", length=1e-6):
                        exp_spec.reserve(signal=self.qubit.signals["measure"])

        return exp_spec


class ReadoutSpectroscopyPulsed(RSpecCwFactory):
    def __init__(
        self,
        parameters,
        qubits,
        exp_settings={
            "num_averages": 2**5,
        },
        ext_calls=None,
        pulse_storage=None,
    ):
        self.parameters = parameters
        self.exp_settings = exp_settings
        self.qubit = qubits

        if pulse_storage is None:
            readout_pulse = pulse_library.const(
                uid="readout_pulse", length=2e-6, amplitude=1.0
            )
            self.pulse_storage = {"readout_pulse": readout_pulse}
        else:
            self.pulse_storage = pulse_storage
        self.exp = self._gen_experiment(self.parameters)

    def _gen_experiment(self, parameters):
        freq_sweep = parameters[0]
        exp_settings = self.exp_settings
        exp_spec = Experiment(
            uid="Resonator Spectroscopy",
            signals=self.qubit.experiment_signals(with_calibration=True),
        )

        exp_spec.signals[self.qubit.signals["measure"]].oscillator = Oscillator(
            "measure_osc",
            frequency=freq_sweep,
            modulation_type=ModulationType.HARDWARE,
        )

        with exp_spec.acquire_loop_rt(
            uid="shots",
            count=exp_settings["num_averages"],
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            averaging_mode=AveragingMode.CYCLIC,
        ):
            with exp_spec.sweep(uid="res_freq", parameter=freq_sweep):
                with exp_spec.section(uid="spectroscopy"):
                    exp_spec.play(
                        signal=self.qubit.signals["measure"],
                        pulse=self.pulse_storage["readout_pulse"],
                    )
                    exp_spec.acquire(
                        signal=self.qubit.signals["acquire"],
                        handle="res_spec",
                        length=self.pulse_storage["readout_pulse"].length,
                    )
                with exp_spec.section(uid="delay", length=10e-6):
                    exp_spec.reserve(signal=self.qubit.signals["measure"])
        return exp_spec


class PulsedQubitSpectroscopy(RSpecCwFactory):
    def __init__(
        self,
        parameters,
        qubits,
        exp_settings={
            "num_averages": 2**5,
        },
        ext_calls=None,
        pulse_storage=None,
    ):
        self.parameters = parameters
        self.exp_settings = exp_settings
        self.qubit = qubits

        if pulse_storage is None:
            readout_pulse = pulse_library.const(
                uid="readout_pulse", length=2e-6, amplitude=1.0
            )
            kernel_pulse = pulse_library.const(
                uid="kernel_pulse", length=2e-6, amplitude=1.0
            )
            self.pulse_storage = {
                "readout_pulse": readout_pulse,
                "kernel_pulse": kernel_pulse,
            }
            logger.info(
                f"Default pulses (const, length: {readout_pulse.length}) are usedfor readout and kernel"
            )
        else:
            self.pulse_storage = pulse_storage
        self.exp = self._gen_experiment(self.parameters)

    def _gen_experiment(self, parameters):
        freq_sweep = parameters[0]
        exp_settings = self.exp_settings
        exp_qspec = Experiment(
            uid="Qubit Spectroscopy",
            signals=self.qubit.experiment_signals(with_calibration=True),
        )

        exp_qspec.signals[self.qubit.signals["drive"]].oscillator = Oscillator(
            "drive_osc",
            frequency=freq_sweep,
            modulation_type=ModulationType.HARDWARE,
        )

        with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=exp_settings["num_averages"],
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_qspec.sweep(uid="qfreq_sweep", parameter=freq_sweep):
                with exp_qspec.section(uid="qubit_excitation"):
                    exp_qspec.play(
                        signal=self.qubit.signals["drive"],
                        pulse=self.pulse_storage["drive_pulse"],
                    )
                with exp_qspec.section(
                    uid="readout_section", play_after="qubit_excitation"
                ):
                    exp_qspec.play(
                        signal=self.qubit.signals["measure"],
                        pulse=self.pulse_storage["readout_pulse"],
                    )
                    exp_qspec.acquire(
                        signal=self.qubit.signals["acquire"],
                        handle="res_spec",
                        kernel=self.pulse_storage["kernel_pulse"],
                    )
                with exp_qspec.section(uid="delay"):
                    exp_qspec.delay(signal=self.qubit.signals["measure"], time=10e-6)
                    exp_qspec.reserve(signal=self.qubit.signals["drive"])

        return exp_qspec

    def get_updated_value(self, analyzed_result):
        return self.qubit.parameters.drive_lo_frequency + analyzed_result


class PulsedQubitSpectroscopySweepRO(RSpecCwFactory):
    def __init__(
        self,
        parameters,
        qubits,
        exp_settings={
            "num_averages": 2**5,
        },
        ext_calls=None,
        pulse_storage=None,
    ):
        super().__init__(
            parameters,
            qubits,
            exp_settings,
            ext_calls=ext_calls,
            pulse_storage=pulse_storage,
        )

    def _gen_experiment(self, parameters):
        freq_sweep = parameters[0]
        exp_settings = self.exp_settings
        exp_qspec = Experiment(
            uid="Qubit Spectroscopy",
            signals=self.qubit.experiment_signals(with_calibration=True),
        )

        osc1 = Oscillator(
            "measure_osc",
            frequency=freq_sweep,
            modulation_type=ModulationType.HARDWARE,
        )
        exp_qspec.signals[self.qubit.signals["measure"]].oscillator = osc1

        with exp_qspec.acquire_loop_rt(
            uid="freq_shots",
            count=exp_settings["num_averages"],
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp_qspec.sweep(uid="qfreq_sweep", parameter=freq_sweep):
                with exp_qspec.section(uid="qubit_excitation"):
                    exp_qspec.play(
                        signal=self.qubit.signals["drive"],
                        pulse=self.pulse_storage["drive_pulse"],
                    )
                with exp_qspec.section(
                    uid="readout_section", play_after="qubit_excitation"
                ):
                    exp_qspec.play(
                        signal=self.qubit.signals["measure"],
                        pulse=self.pulse_storage["readout_pulse"],
                    )
                    exp_qspec.acquire(
                        signal=self.qubit.signals["acquire"],
                        handle="res_spec",
                        length=self.pulse_storage["kernel_pulse"].length,
                    )
                with exp_qspec.section(uid="delay"):
                    exp_qspec.delay(signal=self.qubit.signals["measure"], time=10e-6)
                    exp_qspec.reserve(signal=self.qubit.signals["drive"])

        return exp_qspec


class PulsedQubitSpectroscopyBiasSweep(PulsedQubitSpectroscopy):
    def __init__(
        self,
        parameters,
        qubits,
        exp_settings={
            "num_averages": 2**5,
        },
        ext_calls=None,
        pulse_storage=None,
    ):
        super().__init__(
            parameters,
            qubits,
            exp_settings,
            ext_calls=ext_calls,
            pulse_storage=pulse_storage,
        )
        self.exp = self._gen_experiment(self.parameters)

    def _gen_experiment(self, parameters):
        freq_sweep, dc_volt_sweep = parameters
        exp_settings = self.exp_settings
        exp_qspec = Experiment(
            uid="Qubit Spectroscopy Flux Bias Sweep",
            signals=self.qubit.experiment_signals(with_calibration=True),
        )

        exp_qspec.signals[self.qubit.signals["drive"]].oscillator = Oscillator(
            "drive_osc",
            frequency=freq_sweep,
            modulation_type=ModulationType.HARDWARE,
        )

        with exp_qspec.sweep(uid="dc_volt_sweep", parameter=dc_volt_sweep):
            exp_qspec.call(self.ext_call, qubit_uid=0, voltage=dc_volt_sweep)
            with exp_qspec.acquire_loop_rt(
                uid="freq_shots",
                count=exp_settings["num_averages"],
                acquisition_type=AcquisitionType.INTEGRATION,
            ):
                with exp_qspec.sweep(uid="qfreq_sweep", parameter=freq_sweep):
                    with exp_qspec.section(uid="qubit_excitation"):
                        exp_qspec.play(
                            signal=self.qubit.signals["drive"],
                            pulse=self.pulse_storage["drive_pulse"],
                        )
                    with exp_qspec.section(
                        uid="readout_section", play_after="qubit_excitation"
                    ):
                        exp_qspec.play(
                            signal=self.qubit.signals["measure"],
                            pulse=self.pulse_storage["readout_pulse"],
                        )
                        exp_qspec.acquire(
                            signal=self.qubit.signals["acquire"],
                            handle="res_spec",
                            kernel=self.pulse_storage["kernel_pulse"],
                        )
                    with exp_qspec.section(uid="delay"):
                        exp_qspec.delay(
                            signal=self.qubit.signals["measure"], time=10e-6
                        )
                        exp_qspec.reserve(signal=self.qubit.signals["drive"])

        return exp_qspec


class AmplitudeRabi(TuneUpExperimentFactory):
    def __init__(
        self,
        parameters,
        qubits,
        exp_settings={
            "integration_time": 10e-6,
            "num_averages": 2**5,
            "readout_length": 2e-6,
        },
        ext_calls=None,
        pulse_storage=None,
    ):
        self.parameters = parameters
        self.exp_settings = exp_settings
        self.qubit = qubits

        if pulse_storage is None:
            readout_pulse = pulse_library.const(
                uid="readout_pulse", length=2e-6, amplitude=1.0
            )
            kernel_pulse = pulse_library.const(
                uid="kernel_pulse", length=2e-6, amplitude=1.0
            )
            gaussian_drive = pulse_library.gaussian(
                uid="gaussian_drive",
                length=100e-9,
                amplitude=1.0,
            )
            self.pulse_storage = {
                "readout_pulse": readout_pulse,
                "drive_pulse": gaussian_drive,
                "kernel_pulse": kernel_pulse,
            }
        else:
            self.pulse_storage = pulse_storage

        pi_pulse_amplitude = self.qubit.parameters.user_defined.get(
            "pi_pulse_amplitude"
        )
        if pi_pulse_amplitude is not None:
            logger.info(f"Setting amplitude of drive pulse to {pi_pulse_amplitude}")
            self.pulse_storage["drive_pulse"].amplitude = pi_pulse_amplitude

        self.exp = self._gen_experiment(self.parameters)

    def _gen_experiment(self, parameters):
        amplitude_sweep = parameters[0]
        exp_settings = self.exp_settings
        exp_rabi = Experiment(
            uid="Amplitude Rabi",
            signals=self.qubit.experiment_signals(with_calibration=True),
        )

        with exp_rabi.acquire_loop_rt(
            uid="rabi_shots",
            count=exp_settings["num_averages"],
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with exp_rabi.sweep(uid="rabi_sweep", parameter=amplitude_sweep):
                with exp_rabi.section(
                    uid="qubit_excitation", alignment=SectionAlignment.RIGHT
                ):
                    exp_rabi.play(
                        signal=self.qubit.signals["drive"],
                        pulse=self.pulse_storage["drive_pulse"],
                        amplitude=amplitude_sweep,
                    )
                with exp_rabi.section(
                    uid="readout_section", play_after="qubit_excitation"
                ):
                    exp_rabi.play(
                        signal=self.qubit.signals["measure"],
                        pulse=self.pulse_storage["readout_pulse"],
                    )
                    exp_rabi.acquire(
                        signal=self.qubit.signals["acquire"],
                        handle="amp_rabi",
                        kernel=self.pulse_storage["kernel_pulse"],
                    )
                with exp_rabi.section(uid="delay", length=200e-6):
                    exp_rabi.reserve(signal=self.qubit.signals["measure"])
        return exp_rabi

    def set_extra_calibration(
        self, drive_range=None, measure_range=None, acquire_range=None
    ):
        self.exp.signals[self.qubit.signals["measure"]].range = measure_range
        self.exp.signals[self.qubit.signals["acquire"]].range = acquire_range

    def get_updated_value(self, analyzed_result):
        return analyzed_result

    def plot(self, result):
        plot_results(result, plot_height=4)


class Ramsey(TuneUpExperimentFactory):
    def __init__(
        self,
        parameters,
        qubits,
        exp_settings={
            "integration_time": 10e-6,
            "num_averages": 2**5,
            "readout_length": 2e-6,
        },
        ext_calls=None,
        pulse_storage=None,
    ):
        self.parameters = parameters
        self.exp_settings = exp_settings
        self.qubit = qubits

        if pulse_storage is None:
            readout_pulse = pulse_library.const(
                uid="readout_pulse", length=2e-6, amplitude=0.9
            )
            gaussian_drive = pulse_library.gaussian(
                uid="gaussian_drive",
                length=100e-9,
                amplitude=1.0,
            )
            self.pulse_storage = {
                "readout_pulse": readout_pulse,
                "drive_pulse": gaussian_drive,
            }
        else:
            self.pulse_storage = pulse_storage

        pi_pulse_amplitude = self.qubit.parameters.user_defined.get(
            "pi_pulse_amplitude"
        )
        if pi_pulse_amplitude is not None:
            logger.info(f"Setting amplitude of drive pulse to {pi_pulse_amplitude}")
            self.pulse_storage["drive_pulse"].amplitude = pi_pulse_amplitude

        self.exp = self._gen_experiment(self.parameters)

    def _gen_experiment(self, parameters):
        delay_sweep = parameters[0]
        readout_pulse = self.pulse_storage["readout_pulse"]
        x90 = self.pulse_storage["drive_pulse"]
        exp_ramsey = Experiment(
            uid="Ramsey Experiment",
            signals=self.qubit.experiment_signals(with_calibration=True),
        )
        with exp_ramsey.acquire_loop_rt(
            uid="ramsey_shots",
            count=self.exp_settings["num_averages"],
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            repetition_mode=RepetitionMode.AUTO,
        ):
            with exp_ramsey.sweep(
                uid="ramsey_sweep",
                parameter=delay_sweep,
                alignment=SectionAlignment.RIGHT,
            ):
                with exp_ramsey.section(uid="qubit_excitation"):
                    exp_ramsey.play(signal=self.qubit.signals["drive"], pulse=x90)
                    exp_ramsey.delay(
                        signal=self.qubit.signals["drive"], time=delay_sweep
                    )
                    exp_ramsey.play(signal=self.qubit.signals["drive"], pulse=x90)
                with exp_ramsey.section(
                    uid="readout_section", play_after="qubit_excitation"
                ):
                    exp_ramsey.play(
                        signal=self.qubit.signals["measure"], pulse=readout_pulse
                    )
                    exp_ramsey.acquire(
                        signal=self.qubit.signals["acquire"],
                        handle="ramsey",
                        kernel=readout_pulse,
                    )
                with exp_ramsey.section(uid="delay", length=100e-6):
                    exp_ramsey.reserve(signal=self.qubit.signals["measure"])
        return exp_ramsey

    def get_updated_value(self, analyzed_result):
        return analyzed_result
