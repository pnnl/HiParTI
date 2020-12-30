#!/bin/bash

sudo modprobe msr

sudo pcm-memory -pmm -csv=./bw.csv  -nc 1