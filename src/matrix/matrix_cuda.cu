/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <HiParTI.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>


__global__ static void pti_MatrixDotMulSeqKernel(
    ptiIndex const mode,
    ptiIndex const nmodes,
    ptiIndex const rank,
    ptiIndex const stride,
    ptiValue ** dev_ata)
{
    const ptiIndex tidx = (ptiIndex)threadIdx.x;
    const ptiIndex tidy = (ptiIndex)threadIdx.y;

    ptiValue * ovals = dev_ata[nmodes];
    ovals[tidx * stride + tidy] = 1;
    __syncthreads();

    for(ptiIndex m=1; m < nmodes; ++m) {
        ptiIndex const pm = (mode + m) % nmodes;
        ptiValue const * vals = dev_ata[pm];
        ovals[tidx * stride + tidy] *= vals[tidx * stride + tidy];
    }
    __syncthreads();
}


int ptiCudaMatrixDotMulSeq(
    ptiIndex const mode,
    ptiIndex const nmodes,
    ptiIndex const rank,
    ptiIndex const stride,
    ptiValue ** dev_ata)
{
    dim3 nthreads(rank, rank);  // rank <=  16
    dim3 nblocks(1, 1);

    pti_MatrixDotMulSeqKernel<<<nblocks, nthreads>>> (mode, nmodes, rank, stride, dev_ata);
    
    int result = cudaThreadSynchronize();
    pti_CheckCudaError(result != 0, "CUDA Matrix ptiCudaMatrixDotMulSeq");

    return 0;
}



__global__ static void pti_Matrix2NormKernel(
    ptiIndex const nrows,
    ptiIndex const ncols,
    ptiIndex const stride,
    ptiValue * const dev_vals,
    ptiValue * const dev_lambda)
{
    const ptiIndex tidx = (ptiIndex)threadIdx.x;
    const ptiIndex tidy = (ptiIndex)threadIdx.y;
    const ptiIndex bidx = (ptiIndex)blockIdx.x;
    const ptiIndex i = bidx * blockDim.x + tidx;

    if(i < nrows)
        atomicAdd(&(dev_lambda[tidy]), dev_vals[i*stride + tidy] * dev_vals[i*stride + tidy]);
    __syncthreads();

    dev_lambda[tidy] = sqrt(dev_lambda[tidy]);
    __syncthreads();

    if(i < nrows)
        dev_vals[i*stride + tidy] /= dev_lambda[tidy];
    __syncthreads();

}



int ptiCudaMatrix2Norm(
    ptiIndex const nrows,
    ptiIndex const ncols,
    ptiIndex const stride,
    ptiValue * const dev_vals,
    ptiValue * const dev_lambda)
{
    dim3 nthreads(16, ncols);  // ncols <=  16
    dim3 nblocks((nrows + 16 -1) / 16);

    pti_Matrix2NormKernel<<<nblocks, nthreads>>>(nrows, ncols, stride, dev_vals, dev_lambda);
    int result = cudaThreadSynchronize();
    pti_CheckCudaError(result != 0, "CUDA Matrix ptiCudaMatrix2Norm");

    return 0;
}

