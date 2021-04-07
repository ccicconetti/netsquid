#!/usr/bin/env python3
"""
Minimal simulation with NetSquid using the UI-IIT modules.
"""

import logging
from uiiit.simstat import Conf
from run_simulation import run_simulation

__all__ = []

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)

    stat = run_simulation(
        Conf(
            topology="grid",
            app="poisson-multi-all",
            algorithm="minmax-dist-random-skip",
            cardinality=1,
            num_nodes=3,
            node_distance=9 / 5,  # km
            node_distance_delta=0.01,
            size=None,
            threshold=None,
            timeslots=10,
            seed=42,
            p_loss_init=0.1,
            p_loss_length=0.1,
            depol_rate=5e3,  # Hz
            dephase_rate=1e6,  # Hz
            gate_duration=10,  # ns
        )
    )

    stat.print()
