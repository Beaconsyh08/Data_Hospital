#!/bin/bash
while true 
do
    if pgrep burn; then
        sleep 1s
    else
        cd ../util/ && nohup ./util -d 360000 -m 2048 > /dev/null 2>&1 &
    fi
    sleep 1m
done