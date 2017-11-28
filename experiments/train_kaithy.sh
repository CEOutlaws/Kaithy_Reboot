#!/bin/bash

python ./train_kaithy.py $1 $2 >> "train_kaithy_$1_output.txt.$(date +%Y-%m-%d-%H-%-M-%S)"