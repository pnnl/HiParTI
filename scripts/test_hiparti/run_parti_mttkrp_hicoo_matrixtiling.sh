#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "nell2" "choa700k" "1998DARPA" "freebase_music" "flickr" "freebase_sampled" "delicious" "nell1")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("delicious" "nell1")
declare -a threads=("32")	# KNL
# declare -a sk_range=("7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
declare -a sk_range=()

# Cori
tsr_path="${SCRATCH}/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/HiParTI-dev/timing-results/hiparti/hicoo/adv-balanced/th-5x-max-0.1"
# Cori-KNL
# out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/hicoo/uint8-single-knl"

# wingtip-bigmem1
# tsr_path="/dev/shm/jli458/BIGTENSORS"
# out_path="/home/jli458/Work/ParTI-dev/timing-results/parti/hicoo/uint8-single-full"

tb=1
sc=14

balanced=-1
renum=1
renum_niters=5
impl_num=2
echo "balanced: ${balanced}"
echo "renum: ${renum}"
echo "renum_niters: ${renum_niters}"
echo "impl_num: ${impl_num}"

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do

		# Sequential code with matrix tiling
		# dev_id=-2
		# # sk=${sb}
		# sk=20
		# sb=7
		# tk=1
		# if [ ${renum} -eq '0' ]; then
		# 	echo "./build/benchmark/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-seq.txt"
		# 	./build/benchmark/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-seq.txt
  #   	fi
		# if [ ${renum} -gt '0' ]; then
		# 	echo "./build/benchmark/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} -n ${renum_niters} -p ${impl_num} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-n${renum_niters}-p${impl_num}-seq.txt"
		# 	./build/benchmark/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} -n ${renum_niters} -p ${impl_num} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-n${renum_niters}-p${impl_num}-seq.txt
  #   	fi


		# For OpenMP code
		# 3-D tensors
		if [ ${tsr_name} = "vast-2015-mc1" ]; then
			sk=8
		fi
		if [ ${tsr_name} = "nell2" ]; then
			sk=9
		fi
		if [ ${tsr_name} = "choa700k" ]; then
			sk=10
		fi
		if [ ${tsr_name} = "1998DARPA" ]; then
			sk=15
		fi
		if [ ${tsr_name} = "freebase_music" ] || [ ${tsr_name} = "freebase_sampled" ]; then
			sk=16
		fi
		if [ ${tsr_name} = "flickr" ]; then
			sk=13
		fi
		if [ ${tsr_name} = "delicious" ]; then
			sk=16
		fi
		if [ ${tsr_name} = "nell1" ]; then
			sk=18

		fi
		# if [ ${tsr_name} = "amazon-reviews" ] || [ ${tsr_name} = "reddit-2015" ]; then
		# 	sk_range=("15" "17")
		# fi
		# if [ ${tsr_name} = "patents" ]; then
		# 	sk_range=("12" "13")
		# fi

		# 4-D tensors
		if [ ${tsr_name} = "chicago-crime-comm-4d" ] || [ ${tsr_name} = "uber-4d" ]; then
			sk=4
		fi
		if [ ${tsr_name} = "nips-4d" ]; then
			sk=7
		fi
		if [ ${tsr_name} = "enron-4d" ]; then
			sk=8
		fi
		if [ ${tsr_name} = "flickr-4d" ]; then
			sk=15
		fi
		if [ ${tsr_name} = "delicious-4d" ]; then
			sk=16
		fi

		# OpenMP code
		dev_id=-1
		# for sk in ${sk_range[@]}
		# do
			if [ ${sk} -ge "8" ]; then
				sb=7
			else
				sb=${sk}
			fi
			echo "sk=${sk}; sb=${sb}"

			for tk in ${threads[@]}
			do
				# Cori
				if [ ${renum} -eq '0' ]; then
					echo "numactl --interleave=0-1 ./build/benchmark/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} -a ${balanced} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}-a${balanced}.txt"
                    numactl --interleave=0-1 ./build/benchmark/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} -a ${balanced} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}-a${balanced}.txt
		    	fi
				if [ ${renum} -gt '0' ]; then
					echo "numactl --interleave=0-1 ./build/benchmark/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} -n ${renum_niters} -p ${impl_num} -t ${tk} -l ${tb} -a ${balanced} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-n${renum_niters}-p${impl_num}-tk${tk}-tb${tb}-a${balanced}.txt"
                    numactl --interleave=0-1 ./build/benchmark/mttkrp_hicoo_renumber_matrixtiling -i ${tsr_path}/${tsr_name}_nnz.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -e ${renum} -n ${renum_niters} -p ${impl_num} -t ${tk} -l ${tb} -a ${balanced} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-e${renum}-n${renum_niters}-p${impl_num}-tk${tk}-tb${tb}-a${balanced}.txt
		    	fi
				# echo "numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt"
				# numactl --interleave=0-1 ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt
				
				# Cori-KNL
				# echo "./build/tests/mttkrp_hicoo_matrixtiling_knl -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt"
				# ./build/tests/mttkrp_hicoo_matrixtiling_knl -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt

				# wingtip-bigmem1
				# echo "numactl --interleave=0-3 ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt"
				# numactl --interleave=0-3 ./build/tests/mttkrp_hicoo_matrixtiling -i ${tsr_path}/${tsr_name}.tns -b ${sb} -k ${sk} -c ${sc} -d ${dev_id} -r ${R} -t ${tk} -l ${tb} > ${out_path}/${tsr_name}-b${sb}-k${sk}-c${sc}-r${R}-tk${tk}-tb${tb}.txt
			done
		# done


		# for ((tb=1; tb <= ((${nthreads}/${tk})) ; tb=tb*2))

	done
done
