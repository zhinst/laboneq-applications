import datetime
import uuid
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from laboneq.simple import *  # noqa: F403

from .configs import get_config
from .experiment import TuneUpExperiment
from .qubit_config import QubitConfigs
from .tuneup_logging import initialize_logging

logger = initialize_logging()

config = get_config()


class ScanStatus(Enum):
    PENDING = "gray"
    RUNNING = "yellow"
    FINISHED = "blue"
    FAILED = "red"
    PASSED = "green"


class Scan:
    def __init__(
        self,
        uid: Optional[str] = None,
        session: Session = None,
        qubit_configs: QubitConfigs = None,
        exp_fac: TuneUpExperiment = None,
        exp_settings: Dict[str, Any] = None,
        ext_call: Callable = None,
        dependencies: Union[List["Scan"], "Scan"] = None,
    ) -> None:
        """
        Initialize a scan object.
        Args:
            uid: unique identifier for the scan
            session: L1Q session object containing a device setup
            qubit: Qubit object
            update_key: key of the qubit parameter to be updated
            params: parameter used for scanning (LinearSweepParameter)
            exp_fac: factory for experiment object
            exp_settings: settings for experiment
            ext_call: external call to be passed to the experiment
            analyzer: analyzer object
            pulse_storage: dictionary containing pulse objects and their associated names
            dependencies: set of scan objects that this scan depends on
        """

        # scan-related
        self.uid = uuid.uuid4() if uid is None else uid
        logger.info(f"Creating scan object {self.uid}")
        self.status = ScanStatus.PENDING
        if dependencies is None:
            self._dependencies = set()
        else:
            self._add_deps(dependencies)

        # qubit-related
        self.qubit_configs = qubit_configs
        self.qubit_configs_need_verify = self.qubit_configs.get_need_to_verify()
        self.qubits = qubit_configs.get_qubits()
        logger.debug(f"Number of Qubits: {len(self.qubit_configs)}")

        # exps-related
        self._extra_calib = None
        self.exp_settings = exp_settings
        self.ext_call = ext_call

        self.fig = None
        self.result = None

        # session-related
        self._session = session
        self._device_setup = self._session.device_setup

        # Generate exp
        self._clc_exp_fac = exp_fac

        # CH: I think we can postpone the generation of the experiment until the run method is called
        self._gen_exp()

        logger.info(f"Scan object {self.uid} created")

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, deps: Union[List["Scan"], "Scan"]):
        self._dependencies = set()
        self._add_deps(deps)

    def _gen_exp(self, reapply_extra_calib=False):
        self._exp_fac = self._clc_exp_fac(
            self.qubit_configs,
            exp_settings=self.exp_settings,
            ext_calls=self.ext_call,
        )
        self.experiment = self._exp_fac.exp
        if reapply_extra_calib:
            logger.debug("Reapply extra calib")
            if self._extra_calib is not None:
                self.set_extra_calibration(**self._extra_calib)
            else:
                logger.warning(
                    "Could not find extra calib. Probably call set_extra_calibration first"
                )

    def run(self, report=True, plot=True):
        """Run the scan only.
        Args:
            report: Whether to print out the report.
            plot: Whether to plot the result.
        """

        self.status = ScanStatus.RUNNING
        save_dir = Path(config.get("Settings", "save_dir"))
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"{self.uid}"
        try:
            logger.info(f"Running experiment {self.uid}")
            self.result = self._session.run(self.experiment)
            if report:
                self.report()
                show_pulse_sheet(filepath, self._session.compiled_experiment)
        except Exception as e:
            logger.error(f"Experiment failed with error {e}")
            self.status = ScanStatus.FAILED
            raise RuntimeError(f"Experiment {self.uid} failed") from e
        else:
            logger.info("Experiment finished")
            self.status = ScanStatus.FINISHED
            if plot:
                self.fig = self.plot()

    def run_complete(self, report=True, plot=True):
        """
        Run the scan and perform analysis and verification and update the qubit parameters.
        Args:
            report: Whether to print out the report.
            plot: Whether to plot the result.
        """
        self.run(report=report, plot=plot)
        self.analyze()
        self.verify()
        self.update()

    def report(self):
        for qubit in self.qubits:
            logger.debug("Qubit parameters")
            logger.debug(repr(qubit.parameters))
            signals = [l for _, l in qubit.signals.items()]
            for l in signals:
                c = self._device_setup.get_calibration(l)
                logger.debug(f"Device setup calibration for {l}")
                logger.debug(c)
        logger.debug("Experimental calibrations")
        logger.debug(repr(self.experiment.get_calibration()))
        return signals

    def analyze(self) -> None:
        """
        Analyze result using the analyzer object
        """
        logger.info("Analyzing scan")
        for qubit_config in self.qubit_configs_need_verify:
            logger.info(f"Analyzing qubit {qubit_config.qubit.uid}")
            if qubit_config.analyzer is None:
                logger.warn("No analyzer has been set for this qubit. Skipping")
                continue

            qubit_config._analyzed_result = qubit_config.analyzer.analyze(self.result)
            logger.info(f"Analyzed result: {qubit_config._analyzed_result}")

    def verify(self) -> bool:
        """
        Verify the analyzed scan result using the analyzer object.
        If the scan has not been analyzed, it will be analyzed first.
        """
        logger.info("Verifying scan")
        # Check if any of the qubits has not been analyzed
        # Rerun analyze and log which qubits have not been analyzed
        # Ignore qubit config that does not have to be verified
        analyzed_result = [
            qubit_config._analyzed_result
            for qubit_config in self.qubit_configs_need_verify
        ]

        if None in analyzed_result:
            logger.info("Some qubits have not been analyzed. Analyzing now")
            self.analyze()
        for qubit_config in self.qubit_configs_need_verify:
            analyzer = qubit_config.analyzer
            verified = analyzer.verify(qubit_config._analyzed_result)
            if verified:
                logger.info(f"Scan verified for qubit {qubit_config.qubit.uid}")
            else:
                logger.warning(
                    f"Scan failed verification for qubit {qubit_config.qubit.uid}"
                )
            qubit_config._verified = verified

        verified = self.qubit_configs.all_verified()
        if verified:
            self.status = ScanStatus.PASSED
            logger.info("Scan is verified")
        else:
            self.status = ScanStatus.FAILED
            logger.warning("Scan failed verification")
        return verified

    def update(self):
        """
        Update the qubit parameters
        """
        logger.info("Updating qubit parameters")

        for qubit_config in self.qubit_configs_need_verify:
            if qubit_config._verified:
                qubit_config.update_qubit()
            else:
                warnings.warn(
                    f"Scan for qubit {qubit_config.qubit.uid} failed. Parameters will not be updated"
                )

    def save_result(self):
        current_datetime = datetime.datetime.now()
        save_dir = config.get("Settings", "save_dir")
        datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M")

        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        filename = Path(save_dir_path, f"{self.uid}_{datetime_string}.json")
        self._session.save_results(filename)

    def set_extra_calibration(self, **kwargs):
        self._extra_calib = kwargs  # used to store extra calib that can be reapplied in the _gen_exp_ called by the tuneup
        self._exp_fac.set_extra_calibration(**self._extra_calib)
        logger.debug(f"Set extra calib to {self._extra_calib}")

    @classmethod
    def load(cls, filename):
        """Intend for loading a scan object from a file.
        Not implemented yet.
        """
        pass

    def plot(self):
        self.fig = self._exp_fac.plot(self.result)

    def add_dependencies(self, deps: Union[List["Scan"], "Scan"]):
        """
        Add dependencies to the scan.
        Args:
            deps: A list of Scan objects or a single Scan object
        """
        self._add_deps(deps)

    def _add_deps(self, deps: Union[List["Scan"], "Scan"]):
        if not isinstance(deps, list):
            deps = {deps}
        else:
            deps = set(deps)
        for d in deps:
            if d is self:
                error_msg = "Cannot add a dependency as itself"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if not isinstance(d, self.__class__):
                error_msg = "The dependency must be a Scan object"
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                self._dependencies.add(d)
        logger.info(f"Dependency set to {self._dependencies}")

    def reset_status(self) -> None:
        """Reset the status of the scan to PENDING."""
        self.status = ScanStatus.PENDING

    def __hash__(self):
        # Todo: better hashing
        return hash(self.uid)

    def __str__(self):
        return f"{self.uid}"
