#!/bin/bash

a="disc grid"
for i in $a ; do
  if [ ! -d results-$i ] ; then
    python export.py first_$i.json --outdir results-$i
  fi
done
a="gridmp gridmpvar"
for i in $a ; do
  if [ ! -d results-$i ] ; then
    python export.py first_$i.json --outdir results-$i --factor cardinality
  fi
done

python first_disc_lenrate.py
