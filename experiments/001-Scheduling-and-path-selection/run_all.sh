#!/bin/bash

a="grid gridmp gridmpvar disc"
for i in $a ; do
  python first_$i.py
done
