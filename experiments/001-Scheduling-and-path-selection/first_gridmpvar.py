#!/usr/bin/env python3
"""
5x5 grid with variable cardinalities and irregular path distances.
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
    statfile = "first_gridmpvar.json" # if empty prints to screen
    timeslots = 10000
    nworkers = 50
    algos = ['spf-hops', 'minmax', 'always-skip', 'random-skip']
    cardinalities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Load the previously existing statistics from the JSON file, if any.
    # If such a file is not specified or does not exist then MultiStat is empty.
    mstat = MultiStat() if not statfile else MultiStat.json_load_from_file(statfile)

    confs = []
    for algo in algos:
        for cardinality in cardinalities:
            conf = Conf(
                topology="grid",
                app="poisson-multi-all",
                algorithm=algo,
                cardinality=cardinality,
                num_nodes=5,
                node_distance=9/5, # km
                node_distance_delta=0.4,
                size=None,
                threshold=None,
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
            logging.warning('Some experiments were not executed')

        # Dump the simulation data to the JSON file if specified, otherwise print.
        mstat.add([stat for stat in stats if stat is not None])
        if statfile:
            mstat.json_dump_to_file(statfile)
        else:
            mstat.print()