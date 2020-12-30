HiParTI - Benchmarking Sparse Tensor Kernels
------

# Tensor Manipulation 

## Tensor load/store
### COO format

	Usage: ./build/benchmark/tensor/loadstore [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         --help

## Format conversion
### COO format to HiCOO format
Usage: ./build/benchmark/tensor/convert_hicoo [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)
	         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)
	         --help

### COO format to sCOO format

	Usage: ./build/benchmark/tensor/convert_ssp [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -m MODE, --mode=MODE
	         --help

## Sorting
### COO format only

	Usage: ./build/benchmark/tensor/sort [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -m MODE, --mode=MODE
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits) (required when -s 3) 
	         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits) (required when -s 3) 
	         -s sortcase, --sortcase=SORTCASE (0,1,2,3,4)
	         -t NTHREADS, --nthreads=NTHREADS
	         --help

## Reordering
### COO format only
	Usage: ./build/benchmark/tensor/reorder [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -e RENUMBER, --renumber=RENUMBER
	         -n NITERS_RENUM
	         -p IMPL_NUM, --impl-num=IMPL_NUM
	         -t NTHREADS, --nthreads=NTHREADS
	         --help

# Tensor Operations

## Tensor-Scalar Operations (TS)
### COO format 
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/tensor/smul [options] 

	Options: -X INPUT (.tns file)
	         -a INPUT (a scalar)
	         -Z OUTPUT (output file name)
	         Parallel CPU: use 'export OMP_NUM_THREADS = [number]'
	         --help

For NVIDIA GPU:

	Usage: ./build/benchmark/tensor/smul_gpu [options] 

	Options: -X INPUT (.tns file)
	         -a INPUT (a scalar)
	         -d CUDA_DEV_ID (>=0:GPU id)
	         --help

## Element-wise Tensor Operations (TEW)

### COO format
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/tensor/dmul [options] 

	Options: -x X INPUT
	         -y Y INPUT
	         -o OUTPUT
	         -d CUDA_DEV_ID
	         --help

For GPU:

	Usage: ./build/benchmark/tensor/dmul_gpu [options] 

	Options: -x X INPUT
	         -y Y INPUT
	         -o OUTPUT
	         -d CUDA_DEV_ID
	         --help

## Tensor-Times-Matrix (TTM)
### COO format
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/tensor/ttm [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -m MODE, --mode=MODE
	         -r RANK
	         -d DEV_ID, --dev-id=DEV_ID
	         -t NTHREADS, --nthreads=NTHREADS
	         --help

For GPU:

	Usage: ./build/benchmark/tensor/ttm_gpu [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -m MODE, --mode=MODE
	         -p IMPL_NUM, --impl-num=IMPL_NUM
	         -r RANK
	         -d CUDA_DEV_ID, --cuda-dev-id=CUDA_DEV_ID
	         --help

### sCOO format for semi-sparse tensors
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/tensor/sttm [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -m MODE, --mode=MODE
	         -r RANK
	         -d DEV_ID (Only sequential code is supported here)
	         --help

For GPU:

	Usage: ./build/benchmark/tensor/sttm_gpu [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -m MODE, --mode=MODE
	         -r RANK
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         --help


## Matricized tensor times Khatri-Rao product (MTTKRP)
### COO format
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/tensor/mttkrp [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -m MODE, --mode=MODE (default -1: loop all modes)
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)
	         -s sortcase, --sortcase=SORTCASE (1,2,3,4)
	         -p IMPL_NUM, --impl-num=IMPL_NUM
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -r RANK
	         -t NTHREADS, --nt=NT
	         -u use_reduce, --ur=use_reduce
	         --help

For GPU:

	Usage: ./build/benchmark/tensor/mttkrp_gpu [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -m MODE, --mode=MODE
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)
	         -s sortcase, --sortcase=SORTCASE (1,2,3,4)
	         -p IMPL_NUM, --impl-num=IMPL_NUM
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -r RANK
	         -t NTHREADS, --nt=NT
	         -u use_reduce, --ur=use_reduce
	         --help

### HiCOO format
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/tensor/mttkrp_hicoo [options] 
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)
	         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)
	         -m MODE, --mode=MODE
	         -p IMPL_NUM, --impl-num=IMPL_NUM
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -r RANK
	         -t TK, --tk=TK
	         -l TB, --tb=TB

For GPU:

	Usage: ./build/benchmark/tensor/mttkrp_hicoo_gpu [options] 
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)
	         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)
	         -e RENUMBER, --renumber=RENUMBER
	         -m MODE, --mode=MODE
	         -p IMPL_NUM, --impl-num=IMPL_NUM
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -r RANK
	         -t TK, --tk=TK
	         -l TB, --tb=TB

# Tensor Decompositions

## CANDECOMP/PARAFAC decomposition (CPD)
### COO format
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/tensor/cpd [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -e RENUMBER, --renumber=RENUMBER
	         -n NITERS_RENUM
	         -p IMPL_NUM, --impl-num=IMPL_NUM
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -r RANK
	         -t NTHREADS, --nt=NT
	         -u use_reduce, --ur=use_reduce
	         --help

For GPU:

	Usage: ./build/benchmark/tensor/cpd_gpu [options] 

	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -p IMPL_NUM, --impl-num=IMPL_NUM
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -r RANK
	         -t NTHREADS, --nt=NT
	         -u use_reduce, --ur=use_reduce
	         --help

### HiCOO format
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/tensor/cpd_hicoo [options] 
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)
	         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)
	         -e RENUMBER, --renumber=RENUMBER
	         -n NITERS_RENUM
	         -p IMPL_NUM, --impl-num=IMPL_NUM
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -r RANK
	         -t TK, --tk=TK
	         -l TB, --tb=TB
	         -a balanced
	         --help

For GPU:

	Usage: ./build/benchmark/tensor/cpd_hicoo_gpu [options] 
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)
	         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)
	         -p IMPL_NUM, --impl-num=IMPL_NUM
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -r RANK
	         -t TK, --tk=TK
	         -h TB, --tb=TB

