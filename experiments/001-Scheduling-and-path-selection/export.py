"""
Export uiiit.simstat.MultiStat results to files
"""

import argparse

from uiiit.simstat import MultiStat

__all__ = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("statfile", help="The file to export")
    parser.add_argument("--factor", default="", help="Export single factor results")
    parser.add_argument(
        "--outdir", default="results", help="The directory where to export"
    )
    args = parser.parse_args()

    mstat = MultiStat.json_load_from_file(args.statfile)

    if len(mstat) == 0:
        raise RuntimeError(f"No simulation results found in {args.statfile}")

    to_merge = ["fidelity", "latency"]
    for m in to_merge:
        mstat.apply_to_all(lambda x: x.merge(f"{m}-.*", m))

    if args.factor:
        mstat.single_factor_export(args.factor, args.outdir)

    else:
        to_add_avg = to_merge + ["meas", "length", "delay"]
        for m in to_add_avg:
            mstat.apply_to_all(lambda x: x.add_avg(m))

        mstat.export(args.outdir)
