#!/bin/bash

source ./benchmarks/test_scripts/dataset.sh

source ./benchmarks/test_scripts/common.sh

if [[ ${dev_id} = "-1" || ${dev_id} = "-2" ]]; then
	prog_name="dadd_eq_hicoo"
else
	prog_name="dadd_eq_hicoo_gpu"
fi

echo "${prog_name} ${tsr_path} ${out_path} ${nmodes} ${nt} ${dev_id} ${machine_name}"
echo

for tsr_name in "${run_tsrs[@]}"
do
	sb=7
	if [ ${tsr_name} = "chicago-crime-comm-4d" ] || [ ${tsr_name} = "uber-4d" ]; then
		sb=4
	fi

	if [[ ${dev_id} = "-2" ]]; then
		# Sequetial code
		myprogram="./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}.bin -Y ${tsr_path}/${tsr_name}.bin -d ${dev_id} -b ${sb} > ${out_path}/${tsr_name}_${prog_name}-b${sb}-seq.txt"
		echo ${myprogram}
		eval ${myprogram}

	elif [[ ${dev_id} = "-1" ]]; then
		# OpenMP code
		export OMP_NUM_THREADS=${nt}
		myprogram="${numa_str} ./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}.bin -Y ${tsr_path}/${tsr_name}.bin -d ${dev_id} -b ${sb} > ${out_path}/${tsr_name}_${prog_name}-b${sb}-t${nt}.txt"
		echo ${myprogram}
		eval ${myprogram}

	else
		# CUDA code
		myprogram="./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}.bin -Y ${tsr_path}/${tsr_name}.bin -d ${dev_id} -b ${sb} > ${out_path}/${tsr_name}_${prog_name}-b${sb}-gpu.txt"
		echo ${myprogram}
		eval ${myprogram}
	fi
	
	echo 

done
