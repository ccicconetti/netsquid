#!/usr/bin/env python3
"""
Calibration experiments.
"""

import logging
import netsquid as ns
from uiiit.simstat import Conf, MultiStat
from uiiit.utils import SocketParallerRunner
from run_simulation import run_simulation

__all__ = []

if __name__ == "__main__":
    statfile = ""

    logging.basicConfig(level=logging.INFO)
    statfile = "first_calibration.json"  # if empty prints to screen
    timeslots = 100
    nworkers = 10

    # Load the previously existing statistics from the JSON file, if any.
    # If such a file is not specified or does not exist then MultiStat is empty.
    mstat = MultiStat() if not statfile else MultiStat.json_load_from_file(statfile)

    common_conf = {
        "topology": "chain",
        "app": "const-farthest",
        "algorithm": "spf-hops",
        "cardinality": 1,
        "node_distance_delta": 0,  # random fraction of node_distance, ibidem
        "size": None,  # km, used only with random topology
        "threshold": None,  # km, used only with random topology
        "timeslots": timeslots,
        "seed": 42,
        "p_loss_init": 0.0,
        "p_loss_length": 0.0,
        "gate_duration": 10,  # ns
    }

    total_length = 15  # km
    confs = []
    for depol_rate in [1e3, 1e4]:
        for dephase_rate in [1e5, 1e6, 1e7]:
            for num_node in range(3, 16):
                conf = Conf(
                    **common_conf,
                    num_nodes=num_node,
                    node_distance=float(total_length / (num_node - 1)),
                    dephase_rate=dephase_rate,
                    depol_rate=depol_rate
                )
                if conf not in mstat:  # skip experiments already in the collection
                    confs.append(conf)

    for depol_rate in [1e3, 1e4]:
        for dephase_rate in [1e5, 1e6, 1e7]:
            for node_distance in [1, 2.5, 5, 7.5]:
                conf = Conf(
                    **common_conf,
                    num_nodes=3,
                    node_distance=float(node_distance),
                    dephase_rate=dephase_rate,
                    depol_rate=depol_rate
                )
                if conf not in mstat:  # skip experiments already in the collection
                    confs.append(conf)

    # Run the experiments in parallel with the number of workers specified.
    if confs:
        if len(confs) == 1:
            stats = [run_simulation(confs[0])]
        else:
            stats = SocketParallerRunner("localhost", 21001).run(
                nworkers, run_simulation, confs
            )

        if None in stats:
            raise RuntimeError("Some experiments were not executed")

        # Dump the simulation data to the JSON file if specified, otherwise print.
        mstat.add(stats)
        if statfile:
            mstat.json_dump_to_file(statfile)
        else:
            mstat.print()
