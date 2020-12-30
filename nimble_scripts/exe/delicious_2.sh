#!/bin/sh
result_home=$SPARTA_DIR/nimble_results/delicious_2
app_exe=$SPARTA_DIR/nimble_scripts/delicious_2.sh
script_dir=$IAL_SRC


CPU_NODE="0"
FAST_NODE="0"
SLOW_NODE="2"
FAST_MEM_SIZE="90000"
MIGRATION_THREADS="24"
MIGRATION_INTERVAL="5"
MAX_MANAGED_SIZE_MB="760000"

appout_dir="$result_home"
appout="$appout_dir/appout.txt"

echo "appout=$appout"

rm -rf $appout_dir 2>/dev/zero
mkdir -p $appout_dir 2>/dev/zero

export OMP_NUM_THREADS=24
export DRAM_NODE=0
export OPTANE_NODE=2

$script_dir/launch_optane_new.sh --cpu-node=$CPU_NODE \
                            --fast-node=$FAST_NODE \
                            --slow-node=$SLOW_NODE \
                            --fast-mem-size=$FAST_MEM_SIZE \
                            --migration-threads-num=$MIGRATION_THREADS \
                            --migration-interval=$MIGRATION_INTERVAL \
                            --managed-size=$MAX_MANAGED_SIZE_MB \
                            $app_exe | tee $appout