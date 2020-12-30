#!/bin/bash

export EXPERIMENT_MODES=4

# Need to switch to the Intel Optane Memory Mode manually
echo "---------[Memory Mode]-------" 
MEMORY=true
if ${MEMORY}; then
export DRAM_NODE="0"
export OPTANE_NODE="0"

echo "Chicago: 1 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 1 -x 0 -y 0 -t 24
echo "Chicago: 2 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 2 -x 2 3  -y 2 3 -t 24
echo "Chicago: 3 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 3 -x 1 2 3 -y 1 2 3 -t 24

echo "NIPS: 1 Mode"
numactl --membind=0 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 1 -x 0 -y 0 -t 24
echo "NIPS: 2 Mode"
numactl --membind=0 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 0 3 -y 0 3 -t 24
echo "NIPS: 3 Mode"
numactl --membind=0 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 3 -y 0 1 3 -t 24

echo "Flickr: 1 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 1 -x 1 -y 1 -t 24
echo "Flickr: 2 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 2 -x 0 2 -y 0 2 -t 24
echo "Flickr: 3 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 3 -x 0 2 3 -y 0 2 3 -t 24

echo "Vast: 1 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 1 -x 0 -y 0 -t 24
echo "Vast: 2 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 1 3 -y 1 3 -t 24
echo "Vast: 3 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 3 -x 1 2 4 -y 1 2 4 -t 24

echo "Delicious: 2 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 2 -x 1 2 -y 1 2 -t 24
echo "Delicious: 3 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -t 24

echo "Nell-2: 2 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nell-2.bin -Y $TENSOR_DIR/nell-2.bin -m 2 -x 0 1 -y 0 1 -t 24
fi
