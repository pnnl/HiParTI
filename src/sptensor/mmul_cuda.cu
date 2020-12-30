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
#include <stdio.h>
#include <stdlib.h>
#include "sptensor.h"
#include "mmul_cuda_kernels.h"


int ptiCudaSparseTensorMulMatrix(
    ptiSemiSparseTensor *Y,
    ptiSparseTensor *X,
    const ptiMatrix *U,
    ptiIndex const mode
) {
    int result;
    ptiIndex *ind_buf;
    ptiIndex m;
    ptiNnzIndexVector fiberidx;
    if(mode >= X->nmodes) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "CUDA SpTns * Mtx", "shape mismatch");
    }
    if(X->ndims[mode] != U->nrows) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "CUDA SpTns * Mtx", "shape mismatch");
    }
    ptiSparseTensorSortIndexAtMode(X, mode, 0);
    ind_buf = new ptiIndex[X->nmodes * sizeof *ind_buf];
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = ptiNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    delete[] ind_buf;
    pti_CheckError(result, "CUDA SpTns * Mtx", NULL);
    ptiSemiSparseTensorSetIndices(Y, &fiberidx, X);

    ptiValue *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, Y->nnz * Y->stride * sizeof (ptiValue));
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    // jli: Add memset to Y.
    cudaMemset(Y_val, 0, Y->nnz * Y->stride * sizeof (ptiValue));
    ptiValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * sizeof (ptiValue));
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(X_val, X->values.data, X->nnz * sizeof (ptiValue), cudaMemcpyHostToDevice);
    ptiIndex *X_inds_m = NULL;
    result = cudaMalloc((void **) &X_inds_m, X->nnz * sizeof (ptiValue));
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(X_inds_m, X->inds[mode].data, X->nnz * sizeof (ptiValue), cudaMemcpyHostToDevice);
    ptiValue *U_val = NULL;
    result = cudaMalloc((void **) &U_val, U->nrows * U->stride * sizeof (ptiValue));
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(U_val, U->values, U->nrows * U->stride * sizeof (ptiValue), cudaMemcpyHostToDevice);
    ptiNnzIndex *fiberidx_val = NULL;
    result = cudaMalloc((void **) &fiberidx_val, fiberidx.len * sizeof (ptiNnzIndex));
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(fiberidx_val, fiberidx.data, fiberidx.len * sizeof (ptiNnzIndex), cudaMemcpyHostToDevice);

    const char *env_PARTI_TTM_KERNEL = getenv("PARTI_TTM_KERNEL");
    const bool use_naive_kernel = env_PARTI_TTM_KERNEL && !strcmp(env_PARTI_TTM_KERNEL, "naive");

    const ptiNnzIndex max_nblocks = 32768;
    const ptiNnzIndex max_nthreads = 1024;
    // size_t sharedMem = (Y->ndims[mode] + X->ndims[mode])*sizeof (ptiScalar) + X->ndims[mode]*sizeof (size_t);
    const char *env_PARTI_TTM_NTHREADS = getenv("PARTI_TTM_NTHREADS");
    ptiNnzIndex nthreadsX = 32;
    if(env_PARTI_TTM_NTHREADS) {
        sscanf(env_PARTI_TTM_NTHREADS, "%lu", &nthreadsX);
    }
    ptiNnzIndex sharedMem = nthreadsX * Y->stride * sizeof (ptiValue);

    ptiNnzIndex all_nblocks = Y->nnz % nthreadsX == 0 ? Y->nnz / nthreadsX : Y->nnz / nthreadsX + 1;
    assert(U->ncols < max_nthreads);
    dim3 dimBlock(nthreadsX, U->ncols);
    // size_t nblocks = Y->nnz < max_nblocks ? Y->nnz : max_nblocks;

    if(!use_naive_kernel) {
        fprintf(stderr, "[CUDA SpTns * Mtx] pti_TTMKernel<<<%zu, (%u, %u), %zu>>>\n", all_nblocks, dimBlock.x, dimBlock.y, sharedMem);
    } else {
        fprintf(stderr, "[CUDA SpTns * Mtx] pti_TTMNaiveKernel<<<%zu, (%u, %u), 0>>>\n", all_nblocks, dimBlock.x, dimBlock.y);
    }

    ptiTimer timer;
    ptiNewTimer(&timer, 0);
    ptiStartTimer(timer);

    for(ptiNnzIndex block_offset = 0; block_offset < all_nblocks; block_offset += max_nblocks) {
        ptiNnzIndex nblocks = all_nblocks - block_offset;
        if(nblocks > max_nblocks) {
            nblocks = max_nblocks;
        }
        if(!use_naive_kernel) {
            pti_TTMKernel<<<nblocks, dimBlock, sharedMem>>>(
                Y_val, Y->stride, Y->nnz,
                X_val, X->nnz, X_inds_m,
                fiberidx_val, fiberidx.len,
                U_val, U->nrows, U->ncols, U->stride,
                block_offset
            );
        } else {
            pti_TTMNaiveKernel<<<nblocks, dimBlock>>>(
                Y_val, Y->stride, Y->nnz,
                X_val, X->nnz, X_inds_m,
                fiberidx_val, fiberidx.len,
                U_val, U->nrows, U->ncols, U->stride,
                block_offset
            );
        }
        result = cudaThreadSynchronize();
        pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx kernel");
    }

    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "CUDA SpTns * Mtx");
    ptiFreeTimer(timer);

    cudaMemcpy(Y->values.values, Y_val, Y->nnz * Y->stride * sizeof (ptiValue), cudaMemcpyDeviceToHost);
    result = cudaFree(fiberidx_val);
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    result = cudaFree(U_val);
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    result = cudaFree(X_inds_m);
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    result = cudaFree(X_val);
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    result = cudaFree(Y_val);
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    ptiFreeNnzIndexVector(&fiberidx);

    return 0;
}
