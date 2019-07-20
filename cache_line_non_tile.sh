#!/bin/bash
for i in ${@:2}
do
    python3 main.py --layer $i --model $1;
done

