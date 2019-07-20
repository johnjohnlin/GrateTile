#!/bin/bash
for i in ${@:2}
do
    python3 main.py --simulate_sparsity --layer $i --model $1;
done

