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


int ptiCudaSparseTensorMulMatrixOneKernel(
    ptiSemiSparseTensor *Y,
    ptiSparseTensor *X,
    const ptiMatrix *U,
    ptiIndex const mode,
    ptiIndex const impl_num,
    ptiNnzIndex const smen_size)
{
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

    double flen = (double)X->nnz / fiberidx.len;
    ptiNnzIndex tmp_flen = (fiberidx.data[1] - fiberidx.data[0]) - flen;
    double fvar = tmp_flen * tmp_flen;
    for(ptiNnzIndex i=1; i<fiberidx.len - 1; ++i) {
        tmp_flen = (fiberidx.data[i+1] - fiberidx.data[i]) - flen;
        fvar += tmp_flen * tmp_flen;
    }
    tmp_flen = (X->nnz - fiberidx.data[fiberidx.len - 1]) - flen;
    fvar += tmp_flen * tmp_flen;
    fvar = sqrt(fvar);
    printf("nfibs: %zu, flen: %.2f, fvar: %.2f\n", fiberidx.len, flen, fvar);

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
    result = cudaMalloc((void **) &X_inds_m, X->nnz * sizeof (ptiIndex));
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(X_inds_m, X->inds[mode].data, X->nnz * sizeof (ptiIndex), cudaMemcpyHostToDevice);
    ptiValue *U_val = NULL;
    result = cudaMalloc((void **) &U_val, U->nrows * U->stride * sizeof (ptiValue));
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(U_val, U->values, U->nrows * U->stride * sizeof (ptiValue), cudaMemcpyHostToDevice);
    ptiNnzIndex *fiberidx_val = NULL;
    result = cudaMalloc((void **) &fiberidx_val, fiberidx.len * sizeof (ptiNnzIndex));
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx");
    cudaMemcpy(fiberidx_val, fiberidx.data, fiberidx.len * sizeof (ptiNnzIndex), cudaMemcpyHostToDevice);

    const ptiNnzIndex max_nblocks = 32768;
    const ptiNnzIndex max_nthreads_per_block = 256;
    ptiNnzIndex max_nthreadsy = 16;

    ptiNnzIndex nthreadsx = 1;
    ptiNnzIndex nthreadsy = 1;
    ptiNnzIndex all_nblocks = 0;
    ptiNnzIndex nblocks = 0;

    const char *env_PARTI_TTM_NTHREADS = getenv("PARTI_TTM_NTHREADS");

    switch(impl_num) {
    // case 1:
    case 11: // Naive, 1D
        if(Y->nnz < max_nthreads_per_block) {
            nthreadsx = Y->nnz;
            nblocks = 1;
        } else {
            nthreadsx = max_nthreads_per_block;
            all_nblocks = (Y->nnz + nthreadsx -1) / nthreadsx;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        break;
    case 12:
        if(U->ncols <= max_nthreadsy)
            nthreadsy = U->ncols;
        else
            nthreadsy = max_nthreadsy;
        nthreadsx = max_nthreads_per_block / nthreadsy;

        if(Y->nnz < nthreadsx) {
            nthreadsx = Y->nnz;
            nblocks = 1;
        } else {
            all_nblocks = (Y->nnz + nthreadsx -1) / nthreadsx;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        break;
    case 13:
    case 14:
        if(U->ncols <= max_nthreadsy)
            nthreadsx = U->ncols;
        else
            nthreadsx = max_nthreadsy;
        nthreadsy = max_nthreads_per_block / nthreadsx;

        if(Y->nnz < nthreadsy) {
            nthreadsy = Y->nnz;
            nblocks = 1;
        } else {
            all_nblocks = (Y->nnz + nthreadsy -1) / nthreadsy;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        break;
    case 15:
        if(U->ncols <= max_nthreadsy)
            nthreadsx = U->ncols;
        else
            nthreadsx = max_nthreadsy;
        nthreadsy = max_nthreads_per_block / nthreadsx;

        if(Y->nnz < nthreadsy) {
            nthreadsy = Y->nnz;
            nblocks = 1;
        } else {
            all_nblocks = (Y->nnz + nthreadsy -1) / nthreadsy;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        ptiAssert(smen_size >= nthreadsx * nthreadsy * sizeof (ptiValue));
        break;
    }
    dim3 dimBlock(nthreadsx, nthreadsy);
    printf("all_nblocks: %lu, nthreadsx: %lu, nthreadsy: %lu\n", all_nblocks, nthreadsx, nthreadsy);

    ptiTimer timer;
    ptiNewTimer(&timer, 0);
    ptiStartTimer(timer);


    switch(impl_num) {
    // case 1:
    case 11: // Naive
        printf("[CUDA SpTns * Mtx] pti_TTMNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        pti_TTMNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, Y->stride, Y->nnz,
            X_val, X->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, U->stride);
        break;
    case 12:  
        printf("[CUDA SpTns * Mtx] pti_TTMNnzRankKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        pti_TTMNnzRankKernel<<<nblocks, dimBlock>>>(
            Y_val, Y->stride, Y->nnz,
            X_val, X->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, U->stride);
        break; 
    case 13:  
        printf("[CUDA SpTns * Mtx] pti_TTMRankNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        pti_TTMRankNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, Y->stride, Y->nnz,
            X_val, X->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, U->stride);
        break; 
    case 14:  
        printf("[CUDA SpTns * Mtx] pti_TTMRankRBNnzKernel<<<%lu, (%lu, %lu)>>>\n", nblocks, nthreadsx, nthreadsy);
        pti_TTMRankRBNnzKernel<<<nblocks, dimBlock>>>(
            Y_val, Y->stride, Y->nnz,
            X_val, X->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, U->stride);
        break; 
    case 15:  
        printf("[CUDA SpTns * Mtx] pti_TTMRankRBNnzKernelSM<<<%lu, (%lu, %lu), %lu>>>\n", nblocks, nthreadsx, nthreadsy, smen_size);
        pti_TTMRankRBNnzKernelSM<<<nblocks, dimBlock, smen_size>>>(
            Y_val, Y->stride, Y->nnz,
            X_val, X->nnz, X_inds_m,
            fiberidx_val, fiberidx.len,
            U_val, U->nrows, U->ncols, U->stride);
        break; 
    }
    result = cudaThreadSynchronize();
    pti_CheckCudaError(result != 0, "CUDA SpTns * Mtx kernel");

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
