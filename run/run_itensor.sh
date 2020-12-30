#!/bin/bash
export EXPERIMENT_MODES=3

#T1
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2137_B.bin -Y $TENSOR_DIR/tensor_2137_A.bin -m 2 -x 3 0 -y 1 3 -o 0 -t 12 #We turn off the optional output sorting by setting up '-o 0' as ITensor does not perform output sorting in the core tensor contraction execution

#T2
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2143_B.bin -Y $TENSOR_DIR/tensor_2143_A.bin -m 2 -x 3 0 -y 1 3 -o 0 -t 12

#T3
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2151_B.bin -Y $TENSOR_DIR/tensor_2151_A.bin -m 2 -x 3 0 -y 0 3 -o 0 -t 12

#T4 
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2164_B.bin -Y $TENSOR_DIR/tensor_2164_A.bin -m 2 -x 3 0 -y 1 3 -o 0 -t 12

#T5
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2163_B.bin -Y $TENSOR_DIR/tensor_2163_A.bin -m 2 -x 3 0 -y 0 3 -o 0 -t 12

#T6
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2169_B.bin -Y $TENSOR_DIR/tensor_2169_A.bin -m 2 -x 3 0 -y 0 3 -o 0 -t 12

#T7
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2170_B.bin -Y $TENSOR_DIR/tensor_2170_A.bin -m 2 -x 3 0 -y 1 3 -o 0 -t 12

#T8
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2177_B.bin -Y $TENSOR_DIR/tensor_2177_A.bin -m 2 -x 3 0 -y 1 3 -o 0 -t 12

#T9
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2178_B.bin -Y $TENSOR_DIR/tensor_2178_A.bin -m 2 -x 3 0 -y 0 3 -o 0 -t 12

#T10
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2190_B.bin -Y $TENSOR_DIR/tensor_2190_A.bin -m 2 -x 3 0 -y 0 3 -o 0 -t 12

#T1
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2137_B.bin -Y $TENSOR_DIR/tensor_2137_A.bin -m 2 -x 3 0 -y 1 3 -o 0 -t 12 #We turn off the optional output sorting by setting up '-o 0' as ITensor does not perform output sorting in the core tensor contraction execution

#T2
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2143_B.bin -Y $TENSOR_DIR/tensor_2143_A.bin -m 2 -x 3 0 -y 1 3 -o 0 -t 12

#T3
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2151_B.bin -Y $TENSOR_DIR/tensor_2151_A.bin -m 2 -x 3 0 -y 0 3 -o 0 -t 12

#T4 
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2164_B.bin -Y $TENSOR_DIR/tensor_2164_A.bin -m 2 -x 3 0 -y 1 3 -o 0 -t 12

#T5
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2163_B.bin -Y $TENSOR_DIR/tensor_2163_A.bin -m 2 -x 3 0 -y 0 3 -o 0 -t 12

#T6
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2169_B.bin -Y $TENSOR_DIR/tensor_2169_A.bin -m 2 -x 3 0 -y 0 3 -o 0 -t 12

#T7
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2170_B.bin -Y $TENSOR_DIR/tensor_2170_A.bin -m 2 -x 3 0 -y 1 3 -o 0 -t 12

#T8
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2177_B.bin -Y $TENSOR_DIR/tensor_2177_A.bin -m 2 -x 3 0 -y 1 3 -o 0 -t 12

#T9
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2178_B.bin -Y $TENSOR_DIR/tensor_2178_A.bin -m 2 -x 3 0 -y 0 3 -o 0 -t 12

#T10
numactl --membind=0 --cpunodebind=0 $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/tensor_2190_B.bin -Y $TENSOR_DIR/tensor_2190_A.bin -m 2 -x 3 0 -y 0 3 -o 0 -t 12

