#!/bin/bash

# echo "./prog_name tsr_path out_path nmodes nt dev_id machine_name"

tsr_path=$1		# "${SCRATCH}/BIGTENSORS"
out_path=$2		# "/global/homes/j/jiajiali/Work/SpTenBench/timing-results/pasta/coo"
nmodes=$3 		# 3, or 4
nt=$4			# 32
dev_id=$5	# 0, 1, ...
machine_name=$6	# dgx2, wingtip-bigmem2, bluesky

modes="$(seq -s ' ' 0 $((${nmodes}-1)))"

if [[ ${nmodes} = "3" ]]; then
	run_tsrs=("${s3tsrs[@]}" "${s3tsrs_pl[@]}") 
elif [[ ${nmodes} = "4" ]]; then
	run_tsrs=("${s4tsrs[@]}" "${s4tsrs_pl[@]}") 
fi

# numa_str=""
numa_str="numactl --interleave=all"
if [[ ${machine_name} = "dgx2" ]]; then	# 40 cores
	if [[ ${nt} = "40" ]]; then
		numa_str="numactl --interleave=all --physcpubind=40-79"
	fi
elif [[ ${machine_name} = "wingtip-bigmem2" ]]; then	# 56 cores
	if [[ ${nt} = "56" ]]; then
		numa_str="numactl --interleave=all --physcpubind=56-111"
	fi
elif [[ ${machine_name} = "cori" ]]; then	# 24 cores
	if [[ ${nt} = "32" ]]; then
		numa_str="numactl --interleave=all --physcpubind=32-63"
	fi
elif [[ ${machine_name} = "summit" ]]; then	# 24 cores
	if [[ ${nt} = "64" ]]; then
		numa_str="numactl --interleave=all --physcpubind=32-63,96-127"
	fi
fi