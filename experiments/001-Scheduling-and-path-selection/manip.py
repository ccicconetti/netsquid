"""
Manipulate uiiit.simstat.MultiStat results.
"""

import argparse

from uiiit.simstat import MultiStat

__all__ = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("statfile", help="The file to read.")
    parser.add_argument(
        "--remove", default="", help="Remove the simulations with key=value"
    )
    parser.add_argument(
        "--outfile", default="", help="Output file. If empty overwrite the input."
    )
    args = parser.parse_args()

    mstat = MultiStat.json_load_from_file(args.statfile)

    if len(mstat) == 0:
        raise RuntimeError(f"No simulation results found in {args.statfile}")

    if args.remove:
        key, value = args.remove.split("=")
        if not key or not value:
            raise RuntimeError(f"Invalid key=value pair to remove: {args.remove}")
        mstat.remove(key, value)

        if args.outfile:
            mstat.json_dump_to_file(args.outfile)
        else:
            mstat.json_dump_to_file(args.statfile)
