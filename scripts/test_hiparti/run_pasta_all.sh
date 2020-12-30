#!/bin/bash

if [[ $# < 5 ]]; then
	echo "./run_pasta_all tsr_path out_path nt gpu_dev_id machine_name"
	exit
fi

tsr_path=$1		# "${SCRATCH}/BIGTENSORS"
out_path=$2		# "./timing-results"
nt=$3			# 32
gpu_dev_id=$4	# 0, 1, ...
machine_name=$5	# dgx2, wingtip-bigmem2, bluesky

echo "./run_pasta_all ${tsr_path} ${out_path} ${nt} ${gpu_dev_id} ${machine_name}"
echo

script_path="./benchmarks/test_scripts"
apped="${tsr_path} ${out_path} ${nt} ${gpu_dev_id} ${machine_name}"

# HiCOO
${script_path}/run_pasta_all_hicoo.sh ${apped}

# COO
${script_path}/run_pasta_all_coo.sh ${apped}