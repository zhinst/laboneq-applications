import uuid
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx

from .scan import Scan, ScanStatus
from .tuneup_logging import initialize_logging

logger = initialize_logging()


class TuneUp:
    def __init__(self, uid: Optional[str], scans: Optional[List[Scan]]) -> None:
        """Initializes an instance of TuneUp with the given parameters.

        Args:
            uid (str, optional): unique identifier for the tuneup
            scans (str, optional): list of scans to be included in the tuneup
        """

        self.uid = uid if uid is not None else uuid.uuid4()
        logger.info(f"Creating Tune up with uid: {self.uid}")
        self.scans = {}
        logger.debug("Adding scans to the tuneup")
        for scan in scans:
            self._add_scans(scan)
        logger.debug(f"List of scans added: {self.scans}")

        # graph that contains only the scan_uid as nodes and their dependencies as edges
        self.graph = nx.DiGraph()
        self._generate_graph()

    def _add_scans(self, scan):
        self.scans.update({scan.uid: scan})
        for d in scan.dependencies:
            self._add_scans(d)

    def _generate_graph(self):
        logger.debug("Generating graph")
        for scan_uid, scan in self.scans.items():
            self.graph.add_node(scan_uid)
            for dependency in scan.dependencies:
                self.graph.add_edge(dependency.uid, scan_uid)

    @property
    def run_sequence(self):
        return self._run_sequence

    def _check_up_to(self, scan, ignore_passed_scans=False):
        # neh, scan must implement eq method ?
        # check uid for now instead
        logger.debug(f"Checking up prerequisites for {scan.uid}")
        if scan.uid not in self.scans.keys():
            raise ValueError("Scan not in list")
        res = self.find_required_nodes(scan.uid)
        if ignore_passed_scans:
            self._run_sequence = [
                self.scans.get(s)
                for s in res
                if self.scans.get(s).status != ScanStatus.PASSED
            ]
        else:
            self._run_sequence = [self.scans.get(s) for s in res]
        self._run_sequence_ids = [s.uid for s in self._run_sequence]
        logger.debug(f"Sequence need to run: {self._run_sequence_ids}")

    def _find_required_nodes(self, node):
        """
        May contain duplicated nodes
        """
        required_nodes = []
        predecessors = sorted(list(self.graph.predecessors(node)))
        required_nodes.extend(predecessors)
        for predecessor in predecessors:
            required_nodes.extend(self._find_required_nodes(predecessor))
        return required_nodes

    def find_required_nodes(self, scan_uid: str):
        """
        Return a list of required scans in the order of execution
        """
        res = self._find_required_nodes(scan_uid)
        result_list = []
        [result_list.append(x) for x in res if x not in result_list]
        result_list.reverse()
        return result_list

    def _atomic_run(self, scan, plot=True, analyze=True, verify=True, update=True):
        """
        Make sure to call _gen_exp before running experiments.
        _gen_exp reapplies qubits and recall set_extra_calibration.
        """
        scan._gen_exp(reapply_extra_calib=True)
        scan.run(plot=plot)
        if analyze:
            scan.analyze()
        if verify:
            scan.verify()
        if update:
            scan.update()

    def run(
        self,
        scan: Scan,
        plot_graph=False,
        plot=True,
        ignore_passed_scans=False,
        stop_at_failed=True,
        verify=True,
        analyze=True,
        update=False,
    ) -> bool:
        """Run a scan and all its dependencies

        Args:
            scan (Scan): the target scan
            plot_graph (bool, optional): enable plotting status graph. Defaults to False.
            plot (bool, optional): enable plotting of scan results. Defaults to True.
            with_check (bool, optional): If True, only run scans that are not PASSED. Defaults to False.
            stop_at_failed (bool, optional): stop the run sequence at the first failed scan. Defaults to True.
            verify (bool, optional): if True, verify each scans. Defaults to True.
            analyze (bool, optional): if True, analyze the result of each scan. Defaults to True.
            update (bool, optional): if True, update the qubits. Defaults to False.

        Returns:
            bool: True if all scans passed, False otherwise
        """
        res = self.run_up_to(
            scan,
            plot_graph=plot_graph,
            plot=plot,
            ignore_passed_scans=ignore_passed_scans,
            stop_at_failed=stop_at_failed,
            update=update,
            analyze=analyze,
            verify=verify,
        )
        if not res:
            logger.warning("Prerequisite scans failed. Do not proceed")
            return False
        else:
            self._atomic_run(
                scan, plot=plot, analyze=analyze, verify=verify, update=update
            )
            if plot_graph:
                self.plot()
            if scan.status != ScanStatus.PASSED:
                logger.warning(f"Scan {scan.uid} failed!")
                return False
            else:
                return True

    def run_up_to(
        self,
        scan,
        plot_graph=False,
        plot=True,
        ignore_passed_scans=False,
        stop_at_failed=True,
        analyze=True,
        verify=True,
        update=False,
    ) -> bool:
        """Run all dependencies of the target scan.

        Args:
            scan (Scan): the target scan
            plot_graph (bool, optional): enable plotting status graph. Defaults to False.
            plot (bool, optional): enable plotting of scan results. Defaults to True.
            with_check (bool, optional): If True, only run scans that are not PASSED. Defaults to False.
            stop_at_failed (bool, optional): stop the run sequence at the first failed scan. Defaults to True.
            verify (bool, optional): if True, verify each scans. Defaults to True.
            analyze (bool, optional): if True, analyze the result of each scan. Defaults to True.
            update (bool, optional): if True, update the qubits. Defaults to False.

        Returns:
            bool: True if all scans passed, False otherwise
        """
        self._check_up_to(scan, ignore_passed_scans)
        for s in self._run_sequence:
            if plot_graph:
                self.plot()

            self._atomic_run(
                s, plot=plot, analyze=analyze, verify=verify, update=update
            )

            if plot_graph:
                self.plot()

            if stop_at_failed:
                if s.status == ScanStatus.FAILED:
                    logger.info(f"{s.uid} failed. Stop the tune up here.")
                    return False
        return True

    def run_backtracing(self, scan, plot=True, plot_graph=True):
        """
        Run a scan, if failed, run its predecessors.
        If all pass. run it again
        If one failed, run backtracking on that failed scan.
        NOT TESTED.
        """
        logger.info(f"Run tuneup in backtracking mode for {scan.uid}")
        self.plot()
        self._atomic_run(scan, plot=plot, analyze=True, verify=True, update=True)
        if scan.status == ScanStatus.PASSED:
            logger.info(f"{scan.uid} passed")
            self.plot()
            return True
        else:
            logger.info(f"Scan {scan.uid} failed. Rerun its dependencies")
            self.plot()
            predecessors = list(self.graph.predecessors(scan.uid))
            if not predecessors:
                logger.warning("Reaching the end with failed scan")
                return False
            else:
                for suid in predecessors:
                    res = self.run_backtracking(self.scans.get(suid))
                    # collected_child_run_results.append(res)
                    if not res:
                        logger.warning(
                            f"Dependency branch {suid} failed. Terminate the process!"
                        )
                        return False
        logger.info("After running predecessors, rerun scan")
        self.plot()
        self._atomic_run(scan, plot=plot, analyze=True, verify=True, update=True)
        if scan.status == ScanStatus.PASSED:
            logger.info(f"{scan.uid} passed")
            self.plot()
            return True
        else:
            logger.info("After running predecessors, scan still failed")
            return False

    def _node_properties(self, scan):
        return scan.status.value

    def plot(self):
        """Plot status graph of the tuneup sequence"""
        plt.figure()
        node_colors = [
            self._node_properties(self.scans.get(node)) for node in self.graph.nodes
        ]
        nx.draw(
            self.graph,
            with_labels=True,
            node_color=node_colors,
            pos=nx.planar_layout(self.graph),
            font_weight="bold",
        )
        plt.show()

    def display_status(self, scan=None):
        if scan is not None:
            if scan in self.scans.values():
                print(f"{scan}: {scan.status}")
            else:
                warning.warn(f"Scan {scan} not in scan store")
        else:
            for suid, s in self.scans.items():
                print(f"{suid}: {s.status}")

    def reset_status(self, scan=None):
        """Reset the status of a scan or all scans in the tuneup

        Args:
            scan (Scan, optional): the scan to be reset. Defaults to None.
        """
        if scan is not None:
            if scan in self.scans.values():
                self.scans[scan].reset_status()
            else:
                warning.warn(f"Scan {scan} not in scan store")
        else:
            for suid, s in self.scans.items():
                s.reset_status()
                logger.debug(f"Reset status of {suid}")
