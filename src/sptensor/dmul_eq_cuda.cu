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
#include "sptensor.h"

__global__ static void pti_DotMulKernel(size_t nnz, ptiValue *Z_val, ptiValue *X_val, ptiValue *Y_val)
{
    const ptiNnzIndex tidx = threadIdx.x;
    const ptiNnzIndex i = (ptiNnzIndex) (blockIdx.x * blockDim.x + tidx);

    if(i < nnz) {
        Z_val[i] = X_val[i] * Y_val[i];
    }
    __syncthreads();
}



/**
 * CUDA parallelized Element wise multiply two sparse tensors, with exactly the same nonzero
 * distribution.
 * @param[out] Z the result of X*Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 */
int ptiCudaSparseTensorDotMulEq(ptiSparseTensor *Z, const ptiSparseTensor *X, const ptiSparseTensor *Y) {
    int result;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "CUDA SpTns DotMul", "shape mismatch");
    }
    for(ptiIndex i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CUDA SpTns DotMul", "shape mismatch");
        }
    }
    /* Ensure X and Y have exactly the same nonzero distribution */
    if(Y->nnz != X->nnz) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "SpTns DotMul", "nonzero distribution mismatch");
    }
    ptiNnzIndex nnz = X->nnz;

    ptiCopySparseTensor(Z, X, 1);

    ptiValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * sizeof (ptiValue));
    pti_CheckCudaError(result != 0, "CUDA SpTns DotMul");
    result = cudaMemcpy(X_val, X->values.data, X->nnz * sizeof (ptiValue), cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA SpTns DotMul");

    ptiValue *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, Y->nnz * sizeof (ptiValue));
    pti_CheckCudaError(result != 0, "CUDA SpTns DotMul");
    result = cudaMemcpy(Y_val, Y->values.data, Y->nnz * sizeof (ptiValue), cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA SpTns DotMul");

    ptiValue *Z_val = NULL;
    result = cudaMalloc((void **) &Z_val, X->nnz * sizeof (ptiValue));
    pti_CheckCudaError(result != 0, "CUDA SpTns DotMul");
    result = cudaMemset(Z_val, 0, X->nnz * sizeof (ptiValue));
    pti_CheckCudaError(result != 0, "CUDA SpTns DotMul");

    ptiNnzIndex nthreads = 128;
    ptiNnzIndex nblocks = (nnz + nthreads -1)/ nthreads;

    ptiTimer timer;
    ptiNewTimer(&timer, 0);
    ptiStartTimer(timer);

    pti_DotMulKernel<<<nblocks, nthreads>>>(nnz, Z_val, X_val, Y_val);
    result = cudaThreadSynchronize();

    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "CUDA  SpTns DotMul");
    ptiFreeTimer(timer);

    cudaMemcpy(Z->values.data, Z_val, Z->nnz * sizeof (ptiValue), cudaMemcpyDeviceToHost);

    result = cudaFree(X_val);
    pti_CheckCudaError(result != 0, "CUDA SpTns DotMul");
    result = cudaFree(Y_val);
    pti_CheckCudaError(result != 0, "CUDA SpTns DotMul");
    result = cudaFree(Z_val);
    pti_CheckCudaError(result != 0, "CUDA SpTns DotMul");

    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    pti_SparseTensorCollectZeros(Z);
    /* Sort the indices */
    ptiSparseTensorSortIndex(Z, 1, 1);

    return 0;
}
