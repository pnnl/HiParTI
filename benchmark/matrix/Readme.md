HiParTI - Benchmarking Sparse Matrix Kernels
------

# Matrix Manipulation 

## Matrix load/store

	Usage: ./build/benchmark/matrix/loadstore_mat input output

## Format conversion
### COO format to HiCOO format

	Usage: ./build/benchmark/matrix/convert_hicoo_mat
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -k SUPERBLOCKSIZE (bits), --kernelsize=SUPERBLOCKSIZE (bits)

### COO format to CSR format
	Usage: ./build/benchmark/matrix/convert_csr_mat
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT

## Sorting
### COO format only
	Usage: ./build/benchmark/matrix/sort_mat
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -s SORT
	         -b block_bits

## Reordering
### COO format only
	Usage: ./build/benchmark/matrix/reorder_mat
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -r RELABEL

# Matrix Operations

## Sparse matrix-vector multiplication (SpMV)
### COO format 
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/matrix/spmv_mat
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -u use_reduce, --ur=use_reduce

### CSR format 
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/matrix/spmv_csr_mat
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID

### HiCOO format 
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/matrix/spmv_hicoo_mat
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -k SUPERBLOCKSIZE (bits), --kernelsize=SUPERBLOCKSIZE (bits)
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -u use_schedule, --ur=use_schedule

## Sparse matrix-matrix multiplication (SpMM)
### COO format 
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/matrix/spmm_mat
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -R RANK
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -u use_reduce, --ur=use_reduce

### CSR format 
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/matrix/spmm_csr_mat
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -R RANK
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID

### HiCOO format 
For CPU, sequential and multithreading:

	Usage: ./build/benchmark/matrix/spmm_hicoo_mat
	Options: -i INPUT, --input=INPUT
	         -o OUTPUT, --output=OUTPUT
	         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)
	         -k SUPERBLOCKSIZE (bits), --kernelsize=SUPERBLOCKSIZE (bits)
	         -R RANK
	         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID
	         -u use_schedule, --ur=use_schedule

