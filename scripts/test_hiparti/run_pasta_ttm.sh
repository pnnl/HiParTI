#!/bin/bash

source ./benchmarks/test_scripts/dataset.sh

source ./benchmarks/test_scripts/common.sh

if [[ ${dev_id} = "-1" || ${dev_id} = "-2" ]]; then
	prog_name="ttm"
else
	prog_name="ttm_gpu"
fi

echo "${prog_name} ${tsr_path} ${out_path} ${nmodes} ${nt} ${dev_id} ${machine_name}"
echo

for R in 8 32 64
# for R in 16
do
	for tsr_name in "${run_tsrs[@]}"
	do
		for mode in ${modes[@]}
		do
			if [[ ${dev_id} = "-2" ]]; then
				# Sequetial code
				myprogram="./build/benchmarks/${prog_name} -i ${tsr_path}/${tsr_name}.bin -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}_${prog_name}-m${mode}-r${R}-seq.txt"
				echo ${myprogram}
				eval ${myprogram}

			elif [[ ${dev_id} = "-1" ]]; then
				# OpenMP code
				export OMP_NUM_THREADS=${nt}
				myprogram="${numa_str} ./build/benchmarks/${prog_name} -i ${tsr_path}/${tsr_name}.bin -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}_${prog_name}-m${mode}-r${R}-t${nt}.txt"
				echo ${myprogram}
				eval ${myprogram}

			else
				# CUDA code
				myprogram="./build/benchmarks/${prog_name} -i ${tsr_path}/${tsr_name}.bin -m ${mode} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}_${prog_name}-m${mode}-r${R}-gpu.txt"
				echo ${myprogram}
				eval ${myprogram}
			fi

			echo 
		done
	done
done
