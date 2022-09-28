#!/bin/bash
while true 
do
    if pgrep gpu_burn; then
        sleep 1s
    else
        cd ../gpu-burn/ && nohup ./gpu_burn -d 360000 -m 2048 > /dev/null 2>&1 &
    fi
    sleep 2m
done