"""Simple utility to dump/load uiiit.simstat objects."""

import argparse
import os
from uiiit.simstat import MultiStat

parser = argparse.ArgumentParser(
    description='Convert uiiit.simstat Objectcs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--infile", type=str, default="",
                    help="JSON input file")
parser.add_argument("--outdir", type=str, default="out",
                    help="Output directory where to save the text files")
args = parser.parse_args()

if not args.infile:
    raise ValueError('Empty input file name provided')

with open(args.infile, 'r') as infile:
    mstat = MultiStat.json_load(infile)
    mstat.export(args.outdir)

