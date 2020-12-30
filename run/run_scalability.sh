#!/bin/bash

export EXPERIMENT_MODES=3

for threads in 12 8 4 2 1
do
echo "Dataset: NIPS mode1: $threads" 
numactl --membind=0 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 1 -x 2 -y 2 -t $threads
done

for threads in 12 8 4 2 1
do
echo "Dataset: VAST mode2: $threads" 
numactl --membind=0 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 1 3 -y 1 3 -t $threads
done

for threads in 12 8 4 2 1
do
echo "Dataset: NIPS mode3: $threads" 
numactl --membind=0 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 3 -y 0 1 3 -t $threads
done

