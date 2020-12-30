#!/bin/bash

# Mode 0: COOY + SPA
# Mode 1: COOY + HTA
# Mode 3: HTY + HTA

#Chicago
export EXPERIMENT_MODES=3
echo "Dataset: Chicago, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Chicago 1 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 1 -x 0 -y 0 -t 12
echo "Dataset: Chicago 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 2 -x 0 1 -y 0 1 -t 12
echo "Dataset: Chicago 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 3 -x 1 2 3 -y 1 2 3 -t 12

export EXPERIMENT_MODES=1
echo "Dataset: Chicago, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Chicago 1 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 1 -x 0 -y 0 -t 12
echo "Dataset: Chicago 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 2 -x 0 1 -y 0 1 -t 12
echo "Dataset: Chicago 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 3 -x 1 2 3 -y 1 2 3 -t 12

export EXPERIMENT_MODES=0
echo "Dataset: Chicago, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Chicago 1 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 1 -x 0 -y 0 -t 12
echo "Dataset: Chicago 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 2 -x 0 1 -y 0 1 -t 12
echo "Dataset: Chicago 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/chicago.bin -Y $TENSOR_DIR/chicago.bin -m 3 -x 1 2 3 -y 1 2 3 -t 12


#NIPS
export EXPERIMENT_MODES=3
echo "Dataset: NIPS, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: NIPS 1 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 1 -x 2 -y 2 -t 12
echo "Dataset: NIPS 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 2 3 -y 2 3 -t 12
echo "Dataset: NIPS 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 3 -y 0 1 3 -t 12

export EXPERIMENT_MODES=1
echo "Dataset: NIPS, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: NIPS 1 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 1 -x 2 -y 2 -t 12
echo "Dataset: NIPS 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 2 3 -y 2 3 -t 12
echo "Dataset: NIPS 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 3 -y 0 1 3 -t 12

export EXPERIMENT_MODES=0
echo "Dataset: NIPS, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: NIPS 1 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 1 -x 2 -y 2 -t 12
echo "Dataset: NIPS 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 2 -x 2 3 -y 2 3 -t 12
echo "Dataset: NIPS 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/nips.bin -Y $TENSOR_DIR/nips.bin -m 3 -x 0 1 3 -y 0 1 3 -t 12


#Uber
export EXPERIMENT_MODES=3
echo "Dataset: Uber, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Uber 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 2 -x 0 2 -y 0 2 -t 12
echo "Dataset: Uber 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 3 -x 1 2 3 -y 1 2 3 -t 12

export EXPERIMENT_MODES=1
echo "Dataset: Uber, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Uber 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 2 -x 0 2 -y 0 2 -t 12
echo "Dataset: Uber 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 3 -x 1 2 3 -y 1 2 3 -t 12

export EXPERIMENT_MODES=0
echo "Dataset: Uber, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Uber 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 2 -x 0 2 -y 0 2 -t 12
echo "Dataset: Uber 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uber.bin -Y $TENSOR_DIR/uber.bin -m 3 -x 1 2 3 -y 1 2 3 -t 12


#Vast
export EXPERIMENT_MODES=3
echo "Dataset: Vast, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Vast 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 1 3 -y 1 3 -t 12
echo "Dataset: Vast 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 3 -x 1 2 4 -y 1 2 4 -t 12

export EXPERIMENT_MODES=1
echo "Dataset: Vast, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Vast 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 1 3 -y 1 3 -t 12
echo "Dataset: Vast 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 3 -x 1 2 4 -y 1 2 4 -t 12

export EXPERIMENT_MODES=0
echo "Dataset: Vast, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Vast 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 2 -x 1 3 -y 1 3 -t 12
echo "Dataset: Vast 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/vast.bin -Y $TENSOR_DIR/vast.bin -m 3 -x 1 2 4 -y 1 2 4 -t 12


#Uracil
export EXPERIMENT_MODES=3
echo "Dataset: Uracil, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Uracil 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -Y $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -m 2 -x 0 1 -y 0 1 -t 12
echo "Dataset: Uracil 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -Y $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -m 3 -x 0 1 2 -y 0 1 2 -t 12

export EXPERIMENT_MODES=1
echo "Dataset: Uracil, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Uracil 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -Y $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -m 2 -x 0 1 -y 0 1 -t 12
echo "Dataset: Uracil 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -Y $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -m 3 -x 0 1 2 -y 0 1 2 -t 12

export EXPERIMENT_MODES=0
echo "Dataset: Uracil, execution Mode: $EXPERIMENT_MODES"
echo "Dataset: Uracil 2 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -Y $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -m 2 -x 0 1 -y 0 1 -t 12
echo "Dataset: Uracil 3 mode"
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -Y $TENSOR_DIR/uracil_trimer.sto-3g_T2.bin -m 3 -x 0 1 2 -y 0 1 2 -t 12


