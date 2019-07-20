#!/bin/bash
for i in ${@:2}
do
    python3 main.py --grate_tile --layer $i --model $1;
done

