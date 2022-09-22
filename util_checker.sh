#!/bin/bash
while true 
do
    if pgrep burn; then
        sleep 1s
    else
        cd ../util/ && nohup ./util -d 360000 -m 1024 > /dev/null 2>&1 &
    fi
    sleep 2m
done