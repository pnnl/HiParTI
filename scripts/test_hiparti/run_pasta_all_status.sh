#!/bin/bash

if [[ $# < 5 ]]; then
	echo "./run_pasta_all_hicoo tsr_path out_path nt gpu_dev_id machine_name"
	exit
fi

tsr_path=$1		# "${SCRATCH}/BIGTENSORS"
out_path=$2		# "./timing-results"
nt=$3			# 32
gpu_dev_id=$4	# 0, 1, ...
machine_name=$5	# dgx2, wingtip-bigmem2, bluesky

echo "./run_pasta_all_hicoo ${tsr_path} ${out_path} ${nt} ${gpu_dev_id} ${machine_name}"
echo

script_path="./benchmarks/test_scripts"

declare -a dev_ids=("-2")	# Need to modify for platforms
declare -a modes=("3" "4")

for id in "${dev_ids[@]}"
do
	for nmodes in "${modes[@]}"
	do
		apped="${tsr_path} ${out_path} ${nmodes} ${nt} ${id} ${machine_name}"

		${script_path}/run_pasta_status.sh ${apped}

	done
done