#!/usr/bin/env python
"""
Nodes dropped randomly on a disc.
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
    statfile = "first_disc.json" # if empty prints to screen
    timeslots = 1000
    nworkers = 50
    disc_radius = 6
    threshold = 2
    algos = ['spf-hops', 'minmax', 'always-skip', 'random-skip']
    num_runs = 200
    num_nodes = 25
    cardinalities = [1, 3, 5]
    dephase_rates = [1e5, 1e6]

    # Load the previously existing statistics from the JSON file, if any.
    # If such a file is not specified or does not exist then MultiStat is empty.
    mstat = MultiStat() if not statfile else MultiStat.json_load_from_file(statfile)

    confs = []
    for seed in range(num_runs):
        for algo in algos:
            for cardinality in cardinalities:
                for dephase_rate in dephase_rates:
                    conf = Conf(
                        topology="random",
                        app="random-multi-all",
                        algorithm=algo,
                        cardinality=cardinality,
                        num_nodes=num_nodes,
                        node_distance=None,       # km, unused with random topology
                        node_distance_delta=None, # random fraction of node_distance, ibidem
                        size=disc_radius,         # km, used only with random topology
                        threshold=threshold,      # km, used only with random topology
                        timeslots=timeslots, seed=seed,
                        p_loss_init=0.1, p_loss_length=0.1,
                        depol_rate=5e3,            # Hz
                        dephase_rate=dephase_rate, # Hz
                        gate_duration=10,          # ns
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