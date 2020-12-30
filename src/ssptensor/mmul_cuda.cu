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

__global__ static void pti_TTMKernel(
    ptiValue *Y_val,
    const ptiValue *X_val,
    ptiIndex XY_stride,
    ptiNnzIndex XY_nnz,
    const ptiValue *U_val,
    ptiIndex U_nrows, ptiIndex U_ncols, ptiIndex U_stride,
    ptiIndex mode
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < XY_nnz) {
        size_t r, k;
        for(k = 0; k < U_ncols; ++k) {
            Y_val[tid*XY_stride + k] = 0;
            for(r = 0; r < U_nrows; ++r) {
                Y_val[tid*XY_stride + k] += X_val[tid*XY_stride + r] * U_val[r*U_stride + k];
            }
        }
    }
}

static ptiNnzIndex pti_GetBlockCount(ptiNnzIndex threads) {
    return (threads / 256) + ((threads & 255) != 0);
}

int ptiCudaSemiSparseTensorMulMatrix(
    ptiSemiSparseTensor *Y,
    const ptiSemiSparseTensor *X,
    const ptiMatrix *U,
    ptiIndex mode
) {
    int result;
    ptiIndex *ind_buf;
    ptiIndex m;
    if(mode >= X->nmodes) {
        return -1;
    }
    if(X->ndims[mode] != U->nrows) {
        return -1;
    }
    ind_buf = new ptiIndex[X->nmodes * sizeof *ind_buf];
    if(!ind_buf) {
        return -1;
    }
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    result = ptiNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    delete[] ind_buf;
    if(result) {
        return result;
    }
    for(m = 0; m < Y->nmodes; ++m) {
        if(m != mode) {
            ptiFreeIndexVector(&Y->inds[m]);
            result = ptiCopyIndexVector(&Y->inds[m], &X->inds[m], 1);
            if(result != 0) {
                return result;
            }
        }
    }
    result = ptiResizeMatrix(&Y->values, X->nnz);
    if(result != 0) {
        return result;
    }
    Y->nnz = X->nnz;

    ptiNnzIndex blocks_count = pti_GetBlockCount(Y->nnz);
    ptiNnzIndex threads_count = blocks_count * 256;
    ptiValue *Y_val = NULL;
    result = cudaMalloc((void **) &Y_val, threads_count * Y->stride * sizeof (ptiValue));
    if(result != 0) {
        return result; // TODO: map error code?
    }
    ptiValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, threads_count * X->stride * sizeof (ptiValue));
    if(result != 0) {
        return result; // TODO: map error code?
    }
    cudaMemcpy(X_val, X->values.values, X->nnz * X->stride * sizeof (ptiValue), cudaMemcpyHostToDevice);
    ptiValue *U_val = NULL;
    result = cudaMalloc((void **) &U_val, U->nrows * U->stride * sizeof (ptiValue));
    if(result != 0) {
        return result;
    }
    cudaMemcpy(U_val, U->values, U->nrows * U->stride * sizeof (ptiValue), cudaMemcpyHostToDevice);

    pti_TTMKernel<<<blocks_count, 256>>>(Y_val, X_val, Y->stride, Y->nnz, U_val, U->nrows, U->ncols, U->stride, mode);
    result = cudaGetLastError();
    if(result != 0) {
        return result;
    }

    cudaMemcpy(Y->values.values, Y_val, Y->nnz * Y->stride * sizeof (ptiValue), cudaMemcpyDeviceToHost);
    cudaFree(U_val); cudaFree(X_val); cudaFree(Y_val);

    return 0;
}
