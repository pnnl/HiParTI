#!/bin/bash
  
while true
do
    echo "`date +%H:%m:%s`" >> ./mem.txt
    free >> ./mem.txt
    sleep 1
done