#!/usr/bin/env python3
"""
Variable size grids.
"""

import logging
import netsquid as ns
from uiiit.simstat import Conf, MultiStat
from uiiit.utils import SocketParallerRunner
from run_simulation import run_simulation

__all__ = []

if __name__ == "__main__":
    statfile = ''

    logging.basicConfig(level=logging.INFO)
    statfile = "first_grid.json" # if empty prints to screen
    timeslots = 10000
    nworkers = 50
    grid_length = 9
    num_nodes = [3, 4, 5]
    algos = ['spf-hops', 'minmax-dist']

    # Load the previously existing statistics from the JSON file, if any.
    # If such a file is not specified or does not exist then MultiStat is empty.
    mstat = MultiStat() if not statfile else MultiStat.json_load_from_file(statfile)

    confs = []
    for algo in algos:
        for num_node in num_nodes:
            node_distance = grid_length / num_node
            conf = Conf(
                topology="grid",
                app="const-farthest",
                algorithm=algo,
                cardinality=1,    # unused with const-farthest app
                num_nodes=num_node,
                node_distance=node_distance, # km, unused with random topology
                node_distance_delta=0.01,    # random fraction of node_distance, ibidem
                size=None,        # km, used only with random topology
                threshold=None,   # km, used only with random topology
                timeslots=timeslots, seed=42,
                p_loss_init=0.1, p_loss_length=0.1,
                depol_rate=5e3,    # Hz
                dephase_rate=1e6,  # Hz
                gate_duration=10,  # ns
            )
            if conf not in mstat: # skip experiments already in the collection
                confs.append(conf)

    # Run the experiments in parallel with the number of workers specified.
    if confs:
        if len(confs) == 1:
            stats = [run_simulation(confs[0])]
        else:
            stats = SocketParallerRunner('localhost', 21001).run(
                nworkers, run_simulation, confs)

        if None in stats:
            raise RuntimeError('Some experiments were not executed')

        # Dump the simulation data to the JSON file if specified, otherwise print.
        mstat.add(stats)
        if statfile:
            mstat.json_dump_to_file(statfile)
        else:
            mstat.print()