import datetime
import uuid
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from laboneq.dsl.experiment.pulse import Pulse
from laboneq.simple import *  # noqa: F403

from .analyzer import Analyzer, DefaultAnalyzer
from .configs import get_config
from .experiment import TuneUpExperimentFactory
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
        qubit: Qubit = None,
        update_key: str = "",
        params: List[LinearSweepParameter] = None,
        exp_fac: TuneUpExperimentFactory = None,
        exp_settings: Dict[str, Any] = None,
        ext_call: Callable = None,
        analyzer: Analyzer = None,
        pulse_storage: Dict[str, Pulse] = None,
        analyzing_parameters=None,
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

        self.uid = uuid.uuid4() if uid is None else uid
        logger.info(f"Creating scan object {self.uid}")

        self.analyzing_parameters = (
            analyzing_parameters if analyzing_parameters is not None else {}
        )

        self.result = None
        self.parameters = params
        self.qubit = qubit
        self.fig = None
        self.analyzed_result = None

        self.status = ScanStatus.PENDING

        if dependencies is None:
            self._dependencies = set()
        else:
            self._add_deps(dependencies)

        self.update_key = update_key

        self._session = session
        self._device_setup = self._session.device_setup
        self.pulse_storage = pulse_storage

        if analyzer is not None:
            logger.info(f"Using provided analyzer {analyzer}")
            self.analyzer = analyzer
        else:
            logger.info("Creating a default analyzer")
            self.analyzer = DefaultAnalyzer()

        self._extra_calib = None

        self.exp_settings = exp_settings
        self.ext_call = ext_call

        # Generate exp
        self._clc_exp_fac = exp_fac

        # CH: I think we can postpone the generation of the experiment until the run method is called
        self._gen_exp()

        logger.info(f"Scan object {self.uid} created")

    @property
    def analyzer(self):
        return self._analyzer

    @analyzer.setter
    def analyzer(self, analyzer):
        if isinstance(analyzer, Analyzer):
            self._analyzer = analyzer
            logger.info(f"Analyzer is set to {analyzer}")
        else:
            raise ValueError(
                "The analyzer must be an instance of Analyzer or a subclass of it"
            )

    @property
    def update_key(self):
        return self._update_key

    @update_key.setter
    def update_key(self, update_key: str):
        self._update_key_in_user_defined = False
        if hasattr(self.qubit.parameters, update_key):
            self._update_key = update_key
            logger.info(f"Parameter {update_key} will be scan and updated")
        elif update_key in self.qubit.parameters.user_defined:
            self._update_key = update_key
            self._update_key_in_user_defined = True
            logger.info(
                f"Parameter {update_key} of user defined parameters will be scan and updated"
            )
        else:
            logger.warning("The update key must be a valid parameter of the qubit")
            self._update_key = None
            # raise ValueError("The update key must be a valid parameter of the qubit")

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, deps: Union[List["Scan"], "Scan"]):
        self._dependencies = set()
        self._add_deps(deps)

    def _gen_exp(self, reapply_extra_calib=False):
        self._exp_fac = self._clc_exp_fac(
            self.parameters,
            self.qubit,
            self.exp_settings,
            self.ext_call,
            self.pulse_storage,
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
            raise
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
        logger.debug("Qubit parameters")
        logger.debug(repr(self.qubit.parameters))

        signals = [l for _, l in self.qubit.signals.items()]
        for l in signals:
            c = self._device_setup.get_calibration(l)
            logger.debug(f"Device setup calibration for {l}")
            logger.debug(c)
        logger.debug("Experimental calibrations")
        logger.debug(repr(self.experiment.get_calibration()))
        return signals

    def _analyze(self, **kwargs):
        """
        Analyze result using the analyzer object
        """
        logger.info("Analyzing scan")
        if self.analyzer is None:
            logger.warn("No analyzer has been set")
            return
        if len(self.analyzing_parameters) == 0:
            analyzed_res = self.analyzer.analyze(self.result, **kwargs)
        else:
            logger.info(f"Using parameters set in the init {self.analyzing_parameters}")
            analyzed_res = self.analyzer.analyze(
                self.result, **self.analyzing_parameters
            )
        logger.info(f"Analyzed result: {self.analyzed_result}")
        return analyzed_res

    def analyze(self, **kwargs):
        self.analyzed_result = self._analyze(**kwargs)
        return self.analyzed_result

    def verify(self):
        logger.info("Verifying scan")
        if self.analyzed_result is None:
            self.analyzed_result = self.analyze()
        verified = self.analyzer.verify(self.analyzed_result)
        if verified:
            self.status = ScanStatus.PASSED
            logger.info("Scan verified")
        else:
            self.status = ScanStatus.FAILED
            logger.warning("Scan failed verification")
        return verified

    def update(self, force_value=None):
        """
        Update the qubit parameters
        Args:
            force_value: The value to update the parameter with. If given, the parameter will be updated with this value regardless of the scan status.
            If None, the analyzed result will be used.
        """
        logger.info("Updating qubit parameters")
        update_value = (
            force_value
            if force_value is not None
            else self._exp_fac.get_updated_value(self.analyzed_result)
        )
        if self._update_key is None:
            warnings.warn(
                "No update key has been set. Please set one to update the parameters"
            )
            return
        if force_value is not None:
            self._update(update_value)
        else:
            if self.status == ScanStatus.PASSED:
                self._update(update_value)
            else:
                warnings.warn("Scan was not successful. Parameters will not be updated")

    def _update(self, update_value):
        if self._update_key_in_user_defined:
            self.qubit.parameters.user_defined[self._update_key] = update_value
            logger.info(
                f"User defined parameter {self._update_key} of qubit {self.qubit.uid} is updated to {update_value}"
            )
        else:
            setattr(self.qubit.parameters, self._update_key, update_value)
            logger.info(
                f"Parameter {self._update_key} of qubit {self.qubit.uid} was updated to {update_value}"
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
