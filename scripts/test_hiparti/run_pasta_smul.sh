#!/bin/bash

source ./benchmarks/test_scripts/dataset.sh

source ./benchmarks/test_scripts/common.sh

if [[ ${dev_id} = "-1" || ${dev_id} = "-2" ]]; then
	prog_name="smul"
else
	prog_name="smul_gpu"
fi

a_value=5.0

echo "${prog_name} ${tsr_path} ${out_path} ${nmodes} ${nt} ${dev_id} ${machine_name}"
echo

for tsr_name in "${run_tsrs[@]}"
do
	if [[ ${dev_id} = "-2" ]]; then
		# Sequetial code
		myprogram="./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}.bin -a ${a_value} -d ${dev_id} > ${out_path}/${tsr_name}_${prog_name}-seq.txt"
		echo ${myprogram}
		eval ${myprogram}

	elif [[ ${dev_id} = "-1" ]]; then
		# OpenMP code
		export OMP_NUM_THREADS=${nt}
		myprogram="${numa_str} ./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}.bin -a ${a_value} -d ${dev_id} > ${out_path}/${tsr_name}_${prog_name}-t${nt}.txt"
		echo ${myprogram}
		eval ${myprogram}

	else
		# CUDA code
		myprogram="./build/benchmarks/${prog_name} -X ${tsr_path}/${tsr_name}.bin -a ${a_value} -d ${dev_id} > ${out_path}/${tsr_name}_${prog_name}-gpu.txt"
		echo ${myprogram}
		eval ${myprogram}
	fi

	echo 

done
