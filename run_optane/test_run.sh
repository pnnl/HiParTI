#!/bin/bash

export DRAM_NODE=0
export OPTANE_NODE=2
export EXPERIMENT_MODES=4

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 2 -y 0 1 2 -o 0 -t 24