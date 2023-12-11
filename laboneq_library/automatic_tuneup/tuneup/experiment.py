from abc import ABC, abstractmethod
from typing import Callable, Optional

from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_results
from laboneq.simple import *  # noqa: F403

from .qubit_config import QubitConfigs
from .tuneup_logging import initialize_logging

logger = initialize_logging()


class TuneUpExperiment(ABC):
    def __init__(
        self,
        qubit_configs: QubitConfigs,
        exp_settings: Optional[dict] = None,
        ext_calls: Optional[Callable] = None,
    ):
        self.qubit_configs = qubit_configs
        self.exp_settings = exp_settings
        self.ext_calls = ext_calls
        self.qubits = self.qubit_configs.get_qubits()
        self.exp = self._gen_experiment()

    @abstractmethod
    def _gen_experiment(self):
        """
        Return a complete experiment object, that can be run by the Scan object
        """
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


class MockExp(TuneUpExperiment):
    def __init__(
        self,
        qubit_configs: QubitConfigs,
        exp_settings=None,
        ext_calls: Callable = None,
    ):
        super().__init__(qubit_configs, exp_settings, ext_calls)

    def _gen_experiment(self):
        """
        Return a complete experiment object, that can be run by the Scan object
        """
        qubit = self.qubits[0]
        exp = Experiment(
            uid="Mock exp",
            signals=qubit.experiment_signals(with_calibration=True),
        )
        return exp

    def set_extra_calibration(
        self, drive_range=None, measure_range=None, acquire_range=None
    ):
        pass

    def plot(self, result):
        pass


class ResonatorCWSpec(TuneUpExperiment):
    def __init__(
        self,
        qubit_configs,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5},
        ext_calls=None,
    ):
        super().__init__(qubit_configs, exp_settings, ext_calls)

    def _gen_experiment(self):
        freq_sweep = self.qubit_configs[0].parameter.frequency[0]
        qubit = self.qubits[0]
        exp_settings = self.exp_settings
        exp_spec = Experiment(
            uid="Resonator Spectroscopy",
            signals=qubit.experiment_signals(with_calibration=True),
        )

        exp_spec.signals[qubit.signals["measure"]].oscillator = Oscillator(
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
                        signal=qubit.signals["acquire"],
                        handle="res_spec",
                        length=exp_settings["integration_time"],
                    )
                with exp_spec.section(uid="delay", length=1e-6):
                    exp_spec.reserve(signal=qubit.signals["measure"])
                    exp_spec.reserve(signal=qubit.signals["acquire"])

        return exp_spec

    def plot(self, result):
        plot_results(result, plot_height=4)

    def set_extra_calibration(
        self, drive_range=None, measure_range=None, acquire_range=None
    ):
        if drive_range is not None:
            self.exp.signals[self.qubits[0].signals["drive"]].range = drive_range
        if measure_range is not None:
            self.exp.signals[self.qubits[0].signals["measure"]].range = measure_range
        if acquire_range is not None:
            self.exp.signals[self.qubits[0].signals["acquire"]].range = acquire_range


class ParallelResSpecCW(TuneUpExperiment):
    def __init__(
        self,
        qubit_configs: QubitConfigs,
        ext_calls: Optional[Callable] = None,
        exp_settings: Optional[dict] = {
            "integration_time": 10e-6,
            "num_averages": 2**5,
        },
    ):
        super().__init__(qubit_configs, exp_settings, ext_calls)

    def _gen_experiment(self):
        qubits = self.qubits
        params = self.qubit_configs.get_parameters()
        freq_sweep = [param.frequency[0] for param in params]
        exp_settings = self.exp_settings

        signals = []
        for qubit in qubits:
            signals += qubit.experiment_signals(with_calibration=True)
        exp_spec = Experiment(
            uid="Resonator Spectroscopy",
            signals=signals,
        )

        with exp_spec.acquire_loop_rt(
            uid="shots",
            count=exp_settings.get("num_averages", 2**5) if exp_settings else 2**5,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp_spec.sweep(uid="resonator_frequency_inner", parameter=freq_sweep):
                for qubit in qubits:
                    with exp_spec.section(uid=f"resonator_spectroscopy_{qubit.uid}"):
                        exp_spec.acquire(
                            signal=qubit.signals["acquire"],
                            handle=f"resonator_spectroscopy_{qubit.uid}",
                            length=exp_settings["integration_time"],
                        )
                    with exp_spec.section(uid=f"delay_{qubit.uid}", length=1e-6):
                        exp_spec.reserve(signal=qubit.signals["measure"])
                        exp_spec.reserve(signal=qubit.signals["acquire"])

        for it, qubit in enumerate(qubits):
            exp_spec.signals[qubit.signals["measure"]].oscillator = Oscillator(
                f"measure_osc_{qubit.uid}",
                frequency=freq_sweep[it],
                modulation_type=ModulationType.HARDWARE,  # HAS TO USE HARDWARE MODULATION FOR SPECTROSCOPY MODE
            )
        return exp_spec

    def update_results(self):
        for config in self.qubit_configs:
            config._update_value = (
                config.analyzed_result + config.qubit.parameters.readout_lo_frequency
            )


class ReadoutSpectroscopyCWBiasSweep(ResonatorCWSpec):
    def __init__(
        self,
        qubit_configs,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5},
        ext_calls=None,
    ):
        if ext_calls is None:
            raise ValueError(
                "ext_calls must be provided for sweeping flux for this experiment"
            )
        super().__init__(qubit_configs, exp_settings, ext_calls)

    def _gen_experiment(self):
        freq_sweep = self.qubit_configs[0].parameter.frequency[0]
        dc_volt_sweep = self.qubit_configs[0].parameter.flux[0]
        qubit = self.qubits[0]
        exp_settings = self.exp_settings
        exp_spec = Experiment(
            uid="Resonator Spectroscopy Flux Bias Sweep",
            signals=qubit.experiment_signals(with_calibration=True),
        )

        exp_spec.signals[qubit.signals["measure"]].oscillator = Oscillator(
            "measure_osc",
            frequency=freq_sweep,
            modulation_type=ModulationType.HARDWARE,  # HAS TO USE HARDWARE MODULATION FOR SPECTROSCOPY MODE
        )

        with exp_spec.sweep(uid="dc_volt_sweep", parameter=dc_volt_sweep):
            exp_spec.call(self.ext_calls, qubit_uid=0, voltage=dc_volt_sweep)
            with exp_spec.acquire_loop_rt(
                uid="shots",
                count=exp_settings["num_averages"],
                acquisition_type=AcquisitionType.SPECTROSCOPY,
            ):
                with exp_spec.sweep(uid="res_freq", parameter=freq_sweep):
                    with exp_spec.section(uid="spectroscopy"):
                        exp_spec.acquire(
                            signal=qubit.signals["acquire"],
                            handle="res_spec",
                            length=exp_settings["integration_time"],
                        )
                    with exp_spec.section(uid="delay", length=1e-6):
                        exp_spec.reserve(signal=qubit.signals["measure"])
                        exp_spec.reserve(signal=qubit.signals["acquire"])

        return exp_spec


class ResonatorPulsedSpec(ResonatorCWSpec):
    def __init__(
        self,
        qubit_configs: QubitConfigs,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5},
        ext_calls=None,
    ):
        pulse_storage = qubit_configs[0].pulses
        if pulse_storage is None:
            readout_pulse = pulse_library.const(
                uid="readout_pulse", length=2e-6, amplitude=1.0
            )
            self.pulse_storage = {"readout_pulse": readout_pulse}
        else:
            self.pulse_storage = pulse_storage
        super().__init__(qubit_configs, exp_settings, ext_calls)

    def _gen_experiment(self):
        freq_sweep = self.qubit_configs[0].parameter.frequency[0]
        exp_settings = self.exp_settings
        qubit = self.qubits[0]
        exp_spec = Experiment(
            uid="Resonator Pulsed Spectroscopy",
            signals=qubit.experiment_signals(with_calibration=True),
        )

        exp_spec.signals[qubit.signals["measure"]].oscillator = Oscillator(
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
                        signal=qubit.signals["measure"],
                        pulse=self.pulse_storage["readout_pulse"],
                    )
                    exp_spec.acquire(
                        signal=qubit.signals["acquire"],
                        handle="res_spec",
                        length=self.pulse_storage["readout_pulse"].length,
                    )
                with exp_spec.section(uid="delay", length=10e-6):
                    exp_spec.reserve(signal=qubit.signals["measure"])
        return exp_spec


class PulsedQubitSpectroscopy(ResonatorCWSpec):
    def __init__(
        self,
        qubit_configs: QubitConfigs,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5},
        ext_calls=None,
    ):
        pulse_storage = qubit_configs[0].pulses
        if pulse_storage is None:
            readout_pulse = pulse_library.const(
                uid="readout_pulse", length=2e-6, amplitude=1.0
            )
            kernel_pulse = pulse_library.const(
                uid="kernel_pulse", length=2e-6, amplitude=1.0
            )

            drive_pulse = pulse_library.gaussian(
                uid="drive_pulse", length=100e-9, amplitude=1.0
            )
            self.pulse_storage = {
                "readout_pulse": readout_pulse,
                "kernel_pulse": kernel_pulse,
                "drive_pulse": drive_pulse,
            }
            logger.info(
                f"Default pulses (const, length: {readout_pulse.length}) are usedfor readout and kernel"
            )
        else:
            self.pulse_storage = pulse_storage
        super().__init__(qubit_configs, exp_settings, ext_calls)

    def _gen_experiment(self):
        freq_sweep = self.qubit_configs[0].parameter.frequency[0]
        exp_settings = self.exp_settings
        qubit = self.qubits[0]
        exp_qspec = Experiment(
            uid="Qubit Pulsed Spectroscopy",
            signals=qubit.experiment_signals(with_calibration=True),
        )

        exp_qspec.signals[qubit.signals["drive"]].oscillator = Oscillator(
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
                        signal=qubit.signals["drive"],
                        pulse=self.pulse_storage["drive_pulse"],
                    )
                with exp_qspec.section(
                    uid="readout_section", play_after="qubit_excitation"
                ):
                    exp_qspec.play(
                        signal=qubit.signals["measure"],
                        pulse=self.pulse_storage["readout_pulse"],
                    )
                    exp_qspec.acquire(
                        signal=qubit.signals["acquire"],
                        handle="res_spec",
                        kernel=self.pulse_storage["kernel_pulse"],
                    )
                with exp_qspec.section(uid="delay"):
                    exp_qspec.delay(signal=qubit.signals["measure"], time=10e-6)
                    exp_qspec.reserve(signal=qubit.signals["drive"])

        return exp_qspec


class XtalkPulsedQubitSpectroscopy(ResonatorCWSpec):
    def __init__(
        self,
        qubit_configs: QubitConfigs,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5},
        ext_calls=None,
    ):
        pulse_storage = qubit_configs[0].pulses
        if pulse_storage is None:
            readout_pulse = pulse_library.const(
                uid="readout_pulse", length=2e-6, amplitude=1.0
            )
            kernel_pulse = pulse_library.const(
                uid="kernel_pulse", length=2e-6, amplitude=1.0
            )

            drive_pulse = pulse_library.gaussian(
                uid="drive_pulse", length=100e-9, amplitude=1.0
            )
            self.pulse_storage = {
                "readout_pulse": readout_pulse,
                "kernel_pulse": kernel_pulse,
                "drive_pulse": drive_pulse,
            }
            logger.info(
                f"Default pulses (const, length: {readout_pulse.length}) are usedfor readout and kernel"
            )
        else:
            self.pulse_storage = pulse_storage
        super().__init__(qubit_configs, exp_settings, ext_calls)

    def _gen_experiment(self):
        freq_sweep = self.qubit_configs[0].parameter.frequency[0]
        exp_settings = self.exp_settings
        qubit = self.qubits[0]
        qubit1 = self.qubits[1]

        signals = []
        for qubit in self.qubits:
            signals += qubit.experiment_signals(with_calibration=True)
        print(signals)
        exp_qspec = Experiment(
            uid="Qubit Pulsed Spectroscopy",
            signals=signals,
        )

        exp_qspec.signals[qubit.signals["drive"]].oscillator = Oscillator(
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
                        signal=qubit.signals["drive"],
                        pulse=self.pulse_storage["drive_pulse"],
                    )
                    exp_qspec.play(
                        signal=qubit1.signals["drive"],
                        pulse=self.pulse_storage["drive_pulse"],
                    )
                with exp_qspec.section(
                    uid="readout_section", play_after="qubit_excitation"
                ):
                    exp_qspec.play(
                        signal=qubit.signals["measure"],
                        pulse=self.pulse_storage["readout_pulse"],
                    )
                    exp_qspec.acquire(
                        signal=qubit.signals["acquire"],
                        handle="res_spec",
                        kernel=self.pulse_storage["kernel_pulse"],
                    )
                with exp_qspec.section(uid="delay"):
                    exp_qspec.delay(signal=qubit.signals["measure"], time=10e-6)
                    exp_qspec.reserve(signal=qubit.signals["drive"])

        return exp_qspec


class PulsedQubitSpecBiasSweep(PulsedQubitSpectroscopy):
    def __init__(
        self,
        qubit_configs: QubitConfigs,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5},
        ext_calls=None,
    ):
        if ext_calls is None:
            raise ValueError(
                "ext_calls must be provided for sweeping flux for this experiment"
            )
        super().__init__(qubit_configs, exp_settings, ext_calls)

    def _gen_experiment(self):
        freq_sweep = self.qubit_configs[0].parameter.frequency[0]
        dc_volt_sweep = self.qubit_configs[0].parameter.flux[0]
        qubit = self.qubits[0]
        exp_settings = self.exp_settings
        exp_qspec = Experiment(
            uid="Qubit Spectroscopy Flux Bias Sweep",
            signals=qubit.experiment_signals(with_calibration=True),
        )

        exp_qspec.signals[qubit.signals["drive"]].oscillator = Oscillator(
            "drive_osc",
            frequency=freq_sweep,
            modulation_type=ModulationType.HARDWARE,
        )

        with exp_qspec.sweep(uid="dc_volt_sweep", parameter=dc_volt_sweep):
            exp_qspec.call(self.ext_calls, qubit_uid=0, voltage=dc_volt_sweep)
            with exp_qspec.acquire_loop_rt(
                uid="freq_shots",
                count=exp_settings["num_averages"],
                acquisition_type=AcquisitionType.INTEGRATION,
            ):
                with exp_qspec.sweep(uid="qfreq_sweep", parameter=freq_sweep):
                    with exp_qspec.section(uid="qubit_excitation"):
                        exp_qspec.play(
                            signal=qubit.signals["drive"],
                            pulse=self.pulse_storage["drive_pulse"],
                        )
                    with exp_qspec.section(
                        uid="readout_section", play_after="qubit_excitation"
                    ):
                        exp_qspec.play(
                            signal=qubit.signals["measure"],
                            pulse=self.pulse_storage["readout_pulse"],
                        )
                        exp_qspec.acquire(
                            signal=qubit.signals["acquire"],
                            handle="res_spec",
                            kernel=self.pulse_storage["kernel_pulse"],
                        )
                    with exp_qspec.section(uid="delay"):
                        exp_qspec.delay(signal=qubit.signals["measure"], time=10e-6)
                        exp_qspec.reserve(signal=qubit.signals["drive"])

        return exp_qspec


class AmplitudeRabi(TuneUpExperiment):
    def __init__(
        self,
        qubit_configs,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5},
        ext_calls=None,
    ):
        pulse_storage = qubit_configs[0].pulses
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
        super().__init__(qubit_configs, exp_settings, ext_calls)

    def _gen_experiment(self):
        amplitude_sweep = self.qubit_configs[0].parameter.amplitude[0]
        qubit = self.qubits[0]
        exp_settings = self.exp_settings
        exp_rabi = Experiment(
            uid="Amplitude Rabi",
            signals=qubit.experiment_signals(with_calibration=True),
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
                        signal=qubit.signals["drive"],
                        pulse=self.pulse_storage["drive_pulse"],
                        amplitude=amplitude_sweep,
                    )
                with exp_rabi.section(
                    uid="readout_section", play_after="qubit_excitation"
                ):
                    exp_rabi.play(
                        signal=qubit.signals["measure"],
                        pulse=self.pulse_storage["readout_pulse"],
                    )
                    exp_rabi.acquire(
                        signal=qubit.signals["acquire"],
                        handle="amp_rabi",
                        kernel=self.pulse_storage["kernel_pulse"],
                    )
                with exp_rabi.section(uid="delay", length=200e-6):
                    exp_rabi.reserve(signal=qubit.signals["measure"])
        return exp_rabi

    def set_extra_calibration(
        self, drive_range=None, measure_range=None, acquire_range=None
    ):
        self.exp.signals[self.qubit.signals["measure"]].range = measure_range
        self.exp.signals[self.qubit.signals["acquire"]].range = acquire_range
        self.exp.signals[self.qubit.signals["drive_range"]].range = drive_range

    def plot(self, result):
        plot_results(result, plot_height=4)


class Ramsey(AmplitudeRabi):
    def __init__(
        self,
        qubit_configs: QubitConfigs,
        exp_settings={"integration_time": 10e-6, "num_averages": 2**5},
        ext_calls=None,
    ):
        """Experiment for Ramsey measurement of single qubit
        Args:
            qubit_configs: QubitConfigs object
            exp_settings: experiment settings
            ext_calls: external calls
        pi_pulse_amplitude: amplitude of pi pulse must be set for the qubit used
        in this experiment.
        """
        super().__init__(qubit_configs, exp_settings, ext_calls)
        if self.qubits[0].parameters.user_defined["pi_pulse_amplitude"] is None:
            raise ValueError(
                "pi_pulse_amplitude must be set for the qubit used in this experiment"
            )

    def _gen_experiment(self):
        delay_sweep = self.qubit_configs[0].parameter.delay[0]
        qubit = self.qubit_configs[0].qubit
        readout_pulse = self.pulse_storage["readout_pulse"]
        x90 = self.pulse_storage["drive_pulse"]
        exp_ramsey = Experiment(
            uid="Ramsey Experiment",
            signals=qubit.experiment_signals(with_calibration=True),
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
                    exp_ramsey.play(signal=qubit.signals["drive"], pulse=x90)
                    exp_ramsey.delay(signal=qubit.signals["drive"], time=delay_sweep)
                    exp_ramsey.play(signal=qubit.signals["drive"], pulse=x90)
                with exp_ramsey.section(
                    uid="readout_section", play_after="qubit_excitation"
                ):
                    exp_ramsey.play(
                        signal=qubit.signals["measure"], pulse=readout_pulse
                    )
                    exp_ramsey.acquire(
                        signal=qubit.signals["acquire"],
                        handle="ramsey",
                        kernel=readout_pulse,
                    )
                with exp_ramsey.section(uid="delay", length=100e-6):
                    exp_ramsey.reserve(signal=qubit.signals["measure"])
        return exp_ramsey
