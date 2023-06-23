"""
Plot results of example experiment with quantum repeaters using uiiit modules.
"""

import matplotlib.pyplot as plt
import os
from uiiit.simstat import Conf, MultiStat, Stat, plot_all, plot_all_same

__all__ = []


def save_to_file(filename, x_values, y_values):
    assert len(x_values) == len(y_values)
    with open(filename, "w") as outfile:
        for i in range(len(x_values)):
            outfile.write(f"{x_values[i]} {y_values[i]}\n")


if __name__ == "__main__":
    statfile = "first_calibration.json"
    outdir = "results-calibration"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    mstat = MultiStat.json_load_from_file(statfile)

    if len(mstat) == 0:
        raise RuntimeError(f"No simulation results found in {statfile}")

    to_merge = ["fidelity", "latency"]
    for m in to_merge:
        mstat.apply_to_all(lambda x: x.merge(f"{m}-.*", m))

    _, ax = plt.subplots()
    ax.set_xlabel("Node distance [km]")
    node_distances = [1, 2.5, 5, 7.5]
    for depol_rate in [1e3, 1e4]:
        for dephase_rate in [1e5, 1e6, 1e7]:
            y_values = []
            for node_distance in node_distances:
                stats = mstat.get_stats(
                    num_nodes=3,
                    node_distance=node_distance,
                    depol_rate=depol_rate,
                    dephase_rate=dephase_rate,
                )
                assert len(stats) == 1
                y_values.append(stats[0].get_avg("fidelity"))
            ax.plot(
                node_distances,
                y_values,
                marker="x",
                label=f"depol {depol_rate}, dephase {dephase_rate}",
            )
            save_to_file(
                f"{outdir}/node_distance.depol_rate={depol_rate}.dephase_rate={dephase_rate}-fidelity.dat",
                node_distances,
                y_values,
            )
    ax.grid()
    plt.legend(ncol=3, bbox_to_anchor=(0, 1), loc="lower left", fontsize="small")
    plt.show(block=False)

    _, ax = plt.subplots()
    ax.set_xlabel("Num nodes")
    total_length = 15  # km
    num_nodes = range(3, 16)
    for depol_rate in [1e3, 1e4]:
        for dephase_rate in [1e5, 1e6, 1e7]:
            y_values = []
            for num_node in num_nodes:
                stats = mstat.get_stats(
                    num_nodes=num_node,
                    node_distance=float(total_length / (num_node - 1)),
                    depol_rate=depol_rate,
                    dephase_rate=dephase_rate,
                )
                assert len(stats) == 1
                y_values.append(stats[0].get_avg("fidelity"))
            ax.plot(
                num_nodes,
                y_values,
                marker="x",
                label=f"depol {depol_rate}, dephase {dephase_rate}",
            )
            save_to_file(
                f"{outdir}/num_nodes.depol_rate={depol_rate}.dephase_rate={dephase_rate}-fidelity.dat",
                num_nodes,
                y_values,
            )
    ax.grid()
    plt.legend(ncol=3, bbox_to_anchor=(0, 1), loc="lower left", fontsize="small")
    plt.show(block=True)
