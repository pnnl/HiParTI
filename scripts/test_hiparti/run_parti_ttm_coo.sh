#!/bin/bash

declare -a array=("one" "two" "three")
declare -a s3tsrs=("vast-2015-mc1" "choa700k" "1998DARPA" "nell2" "freebase_music" "flickr" "freebase_sampled" "nell1" "delicious")
declare -a l3tsrs=("amazon-reviews" "patents" "reddit-2015")
declare -a s4tsrs=("chicago-crime-comm-4d" "uber-4d" "nips-4d" "enron-4d" "flickr-4d" "delicious-4d")
declare -a dense3dtsrs=("128" "192" "256" "320" "384" "448" "512")
declare -a test_tsr_names=("freebase_sampled")
declare -a threads=("32")

tsr_path="/people/liji541/BIGTENSORS"
out_path="/global/homes/j/jiajiali/Work/ParTI-dev/timing-results/parti/coo/single"

nmodes=3
modes="$(seq -s ' ' 1 ${nmodes})"
renumber=1

# for R in 8 16 32 64
for R in 16
do
	for tsr_name in "${s3tsrs[@]}"
	do
		for m in ${modes[@]}
		do

			#### Sequetial code ####
			dev_id=-2
			echo "./build/benchmark/ttm_renumber ${tsr_path}/${tsr_name}.tns ${m} ${renumber} ${dev_id} ${R} > ${out_path}/${tsr_name}-r${R}-m${m}-lexi-seq.txt"
			# ./build/benchmark/ttm_renumber ${tsr_path}/${tsr_name}.tns ${m} ${renumber} ${dev_id} ${R} > ${out_path}/${tsr_name}-r${R}-m${m}-lexi-seq.txt
		done
	done
done
