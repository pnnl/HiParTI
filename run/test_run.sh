#!/bin/bash

export EXPERIMENT_MODES=3

numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 2 -y 0 1 2 -t 12