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


/* impl_num = 01 */
__global__ void pti_TTMNaiveKernel(
    ptiValue *Y_val, ptiIndex Y_stride, ptiNnzIndex Y_nnz,
    const ptiValue *X_val, ptiNnzIndex X_nnz, const ptiIndex *X_inds_m,
    const ptiNnzIndex *fiberidx_val, ptiNnzIndex fiberidx_len,
    const ptiValue *U_val, ptiIndex U_nrows, ptiIndex U_ncols, ptiIndex U_stride,
    ptiNnzIndex block_offset)
{
    const ptiNnzIndex tidx = threadIdx.x;
    const ptiNnzIndex tidy = threadIdx.y;
    const ptiNnzIndex i = (blockIdx.x + block_offset) * blockDim.x + tidx;

    if(i >= Y_nnz || tidy >= U_ncols) return;
    const ptiNnzIndex inz_begin = fiberidx_val[i];
    const ptiNnzIndex inz_end = fiberidx_val[i+1];

    Y_val[i*Y_stride + tidy] = 0;
    for(ptiNnzIndex j = inz_begin; j < inz_end; ++j) {
        const ptiIndex r = X_inds_m[j];
        Y_val[i*Y_stride + tidy] += X_val[j] * U_val[r*U_stride + tidy];
    }
}


/* impl_num = 02 */
__global__ void pti_TTMKernel(
    ptiValue *Y_val, ptiIndex Y_stride, ptiNnzIndex Y_nnz,
    const ptiValue *X_val, ptiNnzIndex X_nnz, const ptiIndex *X_inds_m,
    const ptiNnzIndex *fiberidx_val, ptiNnzIndex fiberidx_len,
    const ptiValue *U_val, ptiIndex U_nrows, ptiIndex U_ncols, ptiIndex U_stride,
    ptiNnzIndex block_offset)
{
    extern __shared__ ptiValue mem_pool[];

    const ptiNnzIndex tidx = threadIdx.x;
    const ptiNnzIndex tidy = threadIdx.y;
    const ptiNnzIndex i = (blockIdx.x + block_offset) * blockDim.x + tidx;
    //const ptiNnzIndex off = blockIdx.x * blockDim.x + tidx;

    ptiNnzIndex inz_begin, inz_end;
    if(i < Y_nnz) {
        inz_begin = fiberidx_val[i];
        inz_end = fiberidx_val[i+1];
    }
    __syncthreads();

    //ptiValue * const Y_shr = (ptiValue *) &mem_pool[tidx*Y_stride]; // size U_ncols
    ptiValue * const Y_shr = (ptiValue *) mem_pool; // size U_ncols
    if(i < Y_nnz && tidy < U_ncols) {
        Y_shr[tidx * Y_stride + tidy] = 0;
    }
    __syncthreads();

    if(i < Y_nnz && tidy < U_ncols) {
        for(ptiNnzIndex j = inz_begin; j < inz_end; ++j) {
            const ptiIndex r = X_inds_m[j];
            Y_shr[tidx * Y_stride + tidy] += X_val[j] * U_val[r*U_stride + tidy];
        }
    }
    __syncthreads();

    if(i < Y_nnz && tidy < U_ncols) {
        Y_val[i*Y_stride + tidy] = Y_shr[tidx*Y_stride + tidy];
    }
    __syncthreads();
}


/* impl_num = 11 */
__global__ void pti_TTMNnzKernel(
    ptiValue *Y_val,
    ptiIndex Y_stride,
    ptiNnzIndex Y_nnz,
    const ptiValue * __restrict__ X_val,
    ptiNnzIndex X_nnz,
    const ptiIndex * __restrict__ X_inds_m,
    const ptiNnzIndex * __restrict__ fiberidx_val,
    ptiNnzIndex fiberidx_len,
    const ptiValue * __restrict__ U_val,
    ptiIndex U_nrows,
    ptiIndex U_ncols,
    ptiIndex U_stride)
{
    ptiNnzIndex num_loops_nnz = 1;
    ptiNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const ptiNnzIndex tidx = threadIdx.x;
    ptiNnzIndex x;

    for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < Y_nnz) {
            const ptiNnzIndex inz_begin = fiberidx_val[x];
            const ptiNnzIndex inz_end = fiberidx_val[x+1];

            for(ptiNnzIndex i = inz_begin; i < inz_end; ++i) {
                const ptiIndex row = X_inds_m[i];
                for(ptiIndex r=0; r<U_ncols; ++r) {
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                }
            }
        }
        __syncthreads();
    }

}


/* impl_num = 12 */
__global__ void pti_TTMNnzRankKernel(
    ptiValue *Y_val,
    ptiIndex Y_stride,
    ptiNnzIndex Y_nnz,
    const ptiValue * __restrict__ X_val,
    ptiNnzIndex X_nnz,
    const ptiIndex * __restrict__ X_inds_m,
    const ptiNnzIndex * __restrict__ fiberidx_val,
    ptiNnzIndex fiberidx_len,
    const ptiValue * __restrict__ U_val,
    ptiIndex U_nrows,
    ptiIndex U_ncols,
    ptiIndex U_stride)
{
    ptiNnzIndex num_loops_nnz = 1;
    ptiNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const ptiNnzIndex tidx = threadIdx.x;    // Index nnz
    const ptiNnzIndex tidy = threadIdx.y;    // Index rank
    ptiNnzIndex x;
    const ptiIndex num_loops_r = U_ncols / blockDim.y;
    const ptiIndex rest_loop = U_ncols - num_loops_r * blockDim.y;

    for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < Y_nnz) {
            const ptiNnzIndex inz_begin = fiberidx_val[x];
            const ptiNnzIndex inz_end = fiberidx_val[x+1];
            ptiIndex r;

            for(ptiNnzIndex i = inz_begin; i < inz_end; ++i) {
                const ptiIndex row = X_inds_m[i];
                for(ptiIndex l=0; l<num_loops_r; ++l) {
                    r = tidy + l * blockDim.y;
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }

                if(rest_loop > 0 && tidy < rest_loop) {
                    r = tidy + num_loops_r * blockDim.y;
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
        __syncthreads();
    }

}


/* impl_num = 13 */
__global__ void pti_TTMRankNnzKernel(
    ptiValue *Y_val,
    ptiIndex Y_stride,
    ptiNnzIndex Y_nnz,
    const ptiValue * __restrict__ X_val,
    ptiNnzIndex X_nnz,
    const ptiIndex * __restrict__ X_inds_m,
    const ptiNnzIndex * __restrict__ fiberidx_val,
    ptiNnzIndex fiberidx_len,
    const ptiValue * __restrict__ U_val,
    ptiIndex U_nrows,
    ptiIndex U_ncols,
    ptiIndex U_stride)
{
    ptiNnzIndex num_loops_nnz = 1;
    ptiNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const ptiNnzIndex tidx = threadIdx.x;    // Index rank
    const ptiNnzIndex tidy = threadIdx.y;    // Index nnz
    ptiNnzIndex x;
    const ptiIndex num_loops_r = U_ncols / blockDim.x;
    const ptiIndex rest_loop = U_ncols - num_loops_r * blockDim.x;
    ptiIndex r;

    for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
        if(x < Y_nnz) {
            const ptiNnzIndex inz_begin = fiberidx_val[x];
            const ptiNnzIndex inz_end = fiberidx_val[x+1];

            for(ptiNnzIndex i = inz_begin; i < inz_end; ++i) {
                const ptiIndex row = X_inds_m[i];
                for(ptiIndex l=0; l<num_loops_r; ++l) {
                    r = tidx + l * blockDim.x;
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }

                if(rest_loop > 0 && tidx < rest_loop) {
                    r = tidx + num_loops_r * blockDim.x;
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
        __syncthreads();
    }
}


/* impl_num = 14 */
__global__ void pti_TTMRankRBNnzKernel(
    ptiValue *Y_val,
    ptiIndex Y_stride,
    ptiNnzIndex Y_nnz,
    const ptiValue * __restrict__ X_val,
    ptiNnzIndex X_nnz,
    const ptiIndex * __restrict__ X_inds_m,
    const ptiNnzIndex * __restrict__ fiberidx_val,
    ptiNnzIndex fiberidx_len,
    const ptiValue * __restrict__ U_val,
    ptiIndex U_nrows,
    ptiIndex U_ncols,
    ptiIndex U_stride)
{
    ptiNnzIndex num_loops_nnz = 1;
    ptiNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const ptiNnzIndex tidx = threadIdx.x;    // Index rank
    const ptiNnzIndex tidy = threadIdx.y;    // Index nnz
    ptiNnzIndex x;
    const ptiIndex num_loops_r = U_ncols / blockDim.x;
    const ptiIndex rest_loop = U_ncols - num_loops_r * blockDim.x;
    ptiIndex r;

    for(ptiIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;

        for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < Y_nnz) {
                const ptiNnzIndex inz_begin = fiberidx_val[x];
                const ptiNnzIndex inz_end = fiberidx_val[x+1];

                for(ptiNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const ptiIndex row = X_inds_m[i];
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
    }

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < Y_nnz) {
                const ptiNnzIndex inz_begin = fiberidx_val[x];
                const ptiNnzIndex inz_end = fiberidx_val[x+1];

                for(ptiNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const ptiIndex row = X_inds_m[i];
                    Y_val[x*Y_stride + r] += X_val[i] * U_val[row*U_stride + r];
                    __syncthreads();
                }
            }
        }
    }

}


/* impl_num = 15 */
__global__ void pti_TTMRankRBNnzKernelSM(
    ptiValue *Y_val,
    ptiIndex Y_stride, ptiNnzIndex Y_nnz,
    const ptiValue * __restrict__ X_val,
    ptiNnzIndex X_nnz,
    const ptiIndex * __restrict__ X_inds_m,
    const ptiNnzIndex * __restrict__ fiberidx_val,
    ptiNnzIndex fiberidx_len,
    const ptiValue * __restrict__ U_val,
    ptiIndex U_nrows,
    ptiIndex U_ncols,
    ptiIndex U_stride)
{
    extern __shared__ ptiValue mem_pool[];
    ptiValue * const Y_shr = (ptiValue *) mem_pool; // size U_ncols

    ptiNnzIndex num_loops_nnz = 1;
    ptiNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(Y_nnz > nnz_per_loop) {
        num_loops_nnz = (Y_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }
    
    const ptiNnzIndex tidx = threadIdx.x;
    const ptiNnzIndex tidy = threadIdx.y;
    ptiNnzIndex x;
    const ptiIndex num_loops_r = U_ncols / blockDim.x;
    const ptiIndex rest_loop = U_ncols - num_loops_r * blockDim.x;
    ptiIndex r;


    for(ptiIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;
        for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

            Y_shr[tidy * Y_stride + tidx] = 0;
            __syncthreads();

            if(x < Y_nnz) {
                const ptiNnzIndex inz_begin = fiberidx_val[x];
                const ptiNnzIndex inz_end = fiberidx_val[x+1];
                for(ptiNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const ptiIndex row = X_inds_m[i];
                    Y_shr[tidy*Y_stride + tidx] += X_val[i] * U_val[row*U_stride + r]; 
                }
                __syncthreads();

                Y_val[x*Y_stride + r] = Y_shr[tidy*Y_stride + tidx];
                __syncthreads();
            }
        }
    }


    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;

            Y_shr[tidy * Y_stride + tidx] = 0;
            __syncthreads();

            if(x < Y_nnz) {
                const ptiNnzIndex inz_begin = fiberidx_val[x];
                const ptiNnzIndex inz_end = fiberidx_val[x+1];
                for(ptiNnzIndex i = inz_begin; i < inz_end; ++i) {
                    const ptiIndex row = X_inds_m[i];
                    Y_shr[tidy*Y_stride + tidx] += X_val[i] * U_val[row*U_stride + r]; 
                }
                __syncthreads();

                Y_val[x*Y_stride + r] = Y_shr[tidy*Y_stride + tidx];
                __syncthreads();
            }
        }
    }

}