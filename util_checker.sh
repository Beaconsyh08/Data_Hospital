#!/bin/bash
while true 
do
    if pgrep util; then
        sleep 1s
    else
        cd ../util/ && nohup ./util -d 360000 -m 1024 > /dev/null 2>&1 &
        # cd ../gpu-burn/ && nohup ./gpu_burn -d 360000 -m 1024 > /dev/null 2>&1 &
    fi
    sleep 2m
done