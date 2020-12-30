HiParTI - Benchmarking Sparse Tensor/Matrix Kernels in Python
------

# Matrix Manipulation 

## Matrix load/store

	Usage: python3 benchmarkPy/matrix/loadstore_mat.py [.mtx] [.txt]

## Format conversion
### COO format to HiCOO and CSR formats

	Usage: python3 benchmarkPy/matrix/convert.py [.mtx]

## Sorting
### COO format only
	Usage: python3 benchmarkPy/matrix/sort.py [.mtx]

## Reordering
### COO format only
	Usage: python3 benchmarkPy/matrix/sort.py [.mtx]

# Matrix Operations

## Sparse matrix-vector multiplication (SpMV)
### COO, CSR, HiCOO formats
For CPU, sequential and multithreading:

	Usage: python3 benchmarkPy/matrix/spmv.py [.mtx] [niters]

## Sparse matrix-matrix multiplication (SpMM)
### COO, CSR, HiCOO formats
For CPU, sequential and multithreading:

	Usage: python3 benchmarkPy/matrix/spmm.py [.mtx] [ncols] [niters]


# Tensor Manipulation 
## Load/Store

	Usage: python3 benchmarkPy/matrix/loadstore.py [.tns] [.tns]

# Tensor Operations