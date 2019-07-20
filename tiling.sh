#!/bin/bash
for i in ${@:2}
do
    python3 main.py --grate_tile  --simulate_sparsity --layer $i --model $1;
done

