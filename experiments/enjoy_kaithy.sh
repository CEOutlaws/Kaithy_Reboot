#!/bin/bash

python ./enjoy_kaithy.py $1 | tee "enjoy_kaithy_$1_output.txt.$(date +%Y-%m-%d-%H-%-M-%S)"