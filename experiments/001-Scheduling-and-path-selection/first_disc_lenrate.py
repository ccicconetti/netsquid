#!/usr/bin/env python
"""
Analysis of first_disc.py results.
"""

import logging
from uiiit.simstat import Conf, MultiStat, Stat

__all__ = []

if __name__ == "__main__":
    statfile = ""

    logging.basicConfig(level=logging.INFO)
    statfile = "first_disc.json"  # if empty prints to screen
    outdir = "results-disc-lenrate"

    algos = ["spf-hops", "minmax", "always-skip", "random-skip"]
    cardinalities = [1, 3, 5]
    dephase_rates = [1e5, 1e6]

    mstat = MultiStat.json_load_from_file(statfile)

    newmstat = MultiStat()

    confs = []
    for algo in algos:
        for cardinality in cardinalities:
            for dephase_rate in dephase_rates:
                stats = mstat.get_stats(
                    algorithm=algo, cardinality=cardinality, dephase_rate=dephase_rate
                )
                for i in range(1, 10):
                    newstat = Stat(
                        Conf(
                            algorithm=algo,
                            cardinality=cardinality,
                            dephase_rate=dephase_rate,
                            lenrate=i,
                        )
                    )
                    for stat in stats:
                        metric = f"lenrate-{i}"
                        if metric in stat:
                            newstat.add("lenrate", stat.get_avg(metric))
                    newmstat.add(newstat)

    newmstat.single_factor_export("lenrate", outdir)
