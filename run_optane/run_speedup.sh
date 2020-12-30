#!/bin/bash

export EXPERIMENT_MODES=4

echo "---------[Sparta]-------"
SPARTA=true
if ${SPARTA}; then
export DRAM_NODE="0"
export OPTANE_NODE="2"

echo "Chicago: 1 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 1 -x 0 -y 0 -t 24
echo "Chicago: 2 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 2 -x 2 3  -y 2 3 -t 24
echo "Chicago: 3 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 3 -x 1 2 3 -y 1 2 3 -t 24

echo "NIPS: 1 Mode"
numactl --membind=0,2 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 1 -x 0 -y 0 -t 24
echo "NIPS: 2 Mode"
numactl --membind=0,2 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 0 3 -y 0 3 -t 24
echo "NIPS: 3 Mode"
numactl --membind=0,2 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 3 -y 0 1 3 -t 24

echo "Flickr: 1 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 1 -x 1 -y 1 -t 24
echo "Flickr: 2 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 2 -x 0 2 -y 0 2 -t 24
echo "Flickr: 3 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 3 -x 0 2 3 -y 0 2 3 -t 24

echo "Vast: 1 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 1 -x 0 -y 0 -t 24
echo "Vast: 2 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 1 3 -y 1 3 -t 24
echo "Vast: 3 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 3 -x 1 2 4 -y 1 2 4 -t 24

echo "Delicious: 2 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 2 -x 1 2 -y 1 2 -t 24
echo "Delicious: 3 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -t 24

echo "Nell-2: 2 Mode"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nell-2.bin -Y $TENSOR_DIR/nell-2.bin -m 2 -x 0 1 -y 0 1 -t 24
fi


echo "---------[Optane-only]-------"
OPTANE=true
if ${OPTANE}; then
export DRAM_NODE="2"
export OPTANE_NODE="2"

echo "Chicago: 1 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 1 -x 0 -y 0 -t 24
echo "Chicago: 2 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 2 -x 2 3  -y 2 3 -t 24
echo "Chicago: 3 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 3 -x 1 2 3 -y 1 2 3 -t 24

echo "NIPS: 1 Mode"
numactl --membind=2 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 1 -x 0 -y 0 -t 24
echo "NIPS: 2 Mode"
numactl --membind=2 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 0 3 -y 0 3 -t 24
echo "NIPS: 3 Mode"
numactl --membind=2 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 3 -y 0 1 3 -t 24

echo "Flickr: 1 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 1 -x 1 -y 1 -t 24
echo "Flickr: 2 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 2 -x 0 2 -y 0 2 -t 24
echo "Flickr: 3 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 3 -x 0 2 3 -y 0 2 3 -t 24

echo "Vast: 1 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 1 -x 0 -y 0 -t 24
echo "Vast: 2 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 1 3 -y 1 3 -t 24
echo "Vast: 3 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 3 -x 1 2 4 -y 1 2 4 -t 24

echo "Delicious: 2 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 2 -x 1 2 -y 1 2 -t 24
echo "Delicious: 3 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -t 24

echo "Nell-2: 2 Mode"
numactl --membind=2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nell-2.bin -Y $TENSOR_DIR/nell-2.bin -m 2 -x 0 1 -y 0 1 -t 24
fi

echo "---------[IAL]-------"
IAL=true
if ${IAL}; then
export DRAM_NODE="0"
export OPTANE_NODE="2"

echo "Chicago: 1 Mode"
$SPARTA_DIR/nimble_scripts/exe/chicago_1.sh
echo "Chicago: 2 Mode"
$SPARTA_DIR/nimble_scripts/exe/chicago_2.sh
echo "Chicago: 3 Mode"
$SPARTA_DIR/nimble_scripts/exe/chicago_3.sh

echo "NIPS: 1 Mode"
$SPARTA_DIR/nimble_scripts/exe/nips_1.sh
echo "NIPS: 2 Mode"
$SPARTA_DIR/nimble_scripts/exe/nips_2.sh
echo "NIPS: 3 Mode"
$SPARTA_DIR/nimble_scripts/exe/nips_3.sh

echo "Vast: 1 Mode"
$SPARTA_DIR/nimble_scripts/exe/vast_1.sh
echo "Vast: 2 Mode"
$SPARTA_DIR/nimble_scripts/exe/vast_2.sh
echo "Vast: 3 Mode"
$SPARTA_DIR/nimble_scripts/exe/vast_3.sh

echo "Flickr: 1 Mode"
$SPARTA_DIR/nimble_scripts/exe/flickr_1.sh
echo "Flickr: 2 Mode"
$SPARTA_DIR/nimble_scripts/exe/flickr_2.sh
echo "Flickr: 3 Mode"
$SPARTA_DIR/nimble_scripts/exe/flickr_3.sh

echo "Delicious: 2 Mode"
$SPARTA_DIR/nimble_scripts/exe/delicious_2.sh
echo "Delicious: 3 Mode"
$SPARTA_DIR/nimble_scripts/exe/delicious_3.sh

echo "Nell-2: 2 Mode"
$SPARTA_DIR/nimble_scripts/exe/nell2_2.sh
fi

echo "---------[DRAM-Only]-------"
DRAM=true
if ${DRAM}; then
export DRAM_NODE="0"
export OPTANE_NODE="0"

echo "Chicago: 1 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 1 -x 0 -y 0 -t 24
echo "Chicago: 3 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 3 -x 1 2 3 -y 1 2 3 -t 24

echo "NIPS: 3 Mode"
numactl --membind=0 --cpunodebind=0  $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 3 -y 0 1 3 -t 24

echo "Flickr: 1 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/flickr-4d.bin -Y $TENSOR_DIR/flickr-4d.bin -m 1 -x 1 -y 1 -t 24

echo "Vast: 2 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 1 3 -y 1 3 -t 24
echo "Vast: 3 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 3 -x 1 2 4 -y 1 2 4 -t 24

echo "Delicious: 3 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -t 24

echo "Nell-2: 2 Mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nell-2.bin -Y $TENSOR_DIR/nell-2.bin -m 2 -x 0 1 -y 0 1 -t 24
fi


echo "-------[Placement]-------"
PLACEMENT=true
if ${PLACEMENT}; then
export DRAM_NODE="0"
export OPTANE_NODE="2"
export EXPERIMENT_MODES="5"
echo "All in DRAM"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -p 0 -t 24

echo "X"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -p 1 -t 24

echo "Y"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -p 2 -t 24

echo "HtY"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -p 3 -t 24

echo "HtA"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -p 4 -t 24

echo "Z_local"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -p 5 -t 24

echo "Z"
numactl --membind=0,2 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/delicious.bin -Y $TENSOR_DIR/delicious.bin -m 3 -x 0 2 3 -y 0 2 3 -p 6 -t 24
fi