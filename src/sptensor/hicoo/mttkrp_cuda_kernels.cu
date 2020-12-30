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
#include "hicoo.h"
#include "mttkrp_cuda_kernels.h"
#include <inttypes.h>

int ptiMTTKRPKernelHiCOO(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiNnzIndex max_nnzb,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sc_bits,
    const ptiIndex blength,
    const int impl_num,
    const ptiNnzIndex kptr_begin,
    const ptiNnzIndex kptr_end,
    ptiIndex * const dev_ndims,
    ptiNnzIndex * const dev_cptr,
    ptiNnzIndex * const dev_bptr,
    ptiBlockIndex ** const dev_binds,
    ptiElementIndex ** const dev_einds,
    ptiValue * const dev_values,
    ptiIndex * const dev_mats_order,
    ptiValue ** const dev_mats)
{
    int result = 0;

    /* Maximum settings */
    ptiIndex max_nthreads_per_block = 256;
    ptiIndex max_nblocks = 32768;
    ptiIndex max_R = 4;

    ptiIndex nthreadsx = 0;
    ptiIndex nthreadsy = 0;
    ptiIndex nblocks = 0;
    ptiIndex shr_size = 0;
    ptiNnzIndex all_nblocks = blength;

    switch(nmodes) {
    case 3: /* 3-D tensors */
        switch(impl_num) {
        case 1: // Naive, 1D
            /* Set number of blocks and threads */
            nthreadsx = max_nnzb;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
            // shr_size = 2 * nmodes * sizeof(ptiIndex);
            break;
        case 2:
            nthreadsy = R;
            nthreadsx = max_nnzb;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
            // shr_size = 2 * nmodes * sizeof(ptiIndex);
            break;
        case 3:
            nthreadsx = R;
            nthreadsy = max_nnzb;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
            // shr_size = 2 * nmodes * sizeof(ptiIndex);
            break;
        case 4:
            nthreadsx = R;
            if(R <= max_R)
                nthreadsx = R;
            else
                nthreadsx = max_R;
            nthreadsy = max_nnzb;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
            // shr_size = 2 * nmodes * sizeof(ptiIndex);
            break;

        /* Matrix blocked implementations */
        case 14:
            nthreadsx = R;
            if(R <= max_R)
                nthreadsx = R;
            else
                nthreadsx = max_R;
            nthreadsy = max_nnzb;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
            // shr_size = 2 * nmodes * sizeof(ptiIndex);
            break;

        /* Shared memory for the output matrix + switch block sizes */
        case 15:
            nthreadsx = R;
            if(R <= max_R) {
                nthreadsx = R;
            }
            else {
                nthreadsx = max_R;
            }
            nthreadsy = max_nnzb;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
            shr_size = (ptiIndex)pow(2, sb_bits) * R * sizeof(ptiValue);
            break;

        /* Shared memory for three matrices + switch block sizes */
        case 16:
            nthreadsx = R;
            if(R <= max_R) {
                nthreadsx = R;
            }
            else {
                nthreadsx = max_R;
            }
            nthreadsy = max_nnzb;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
            shr_size = nmodes * (ptiIndex)pow(2, sb_bits) * R * sizeof(ptiValue);
            break;

        }

        dim3 dimBlock(nthreadsx, nthreadsy);
        switch(impl_num) {
        case 1: // Naive, 1D
            printf("\nExecute pti_MTTKRPKernelHiCOO_3D_naive (%u, %u)\n", nblocks, nthreadsx);
            pti_MTTKRPKernelHiCOO_3D_naive<<<nblocks, nthreadsx>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                sb_bits,
                sc_bits,
                blength,
                kptr_begin,
                kptr_end,
                dev_ndims,
                dev_cptr,
                dev_bptr,
                dev_binds,
                dev_einds,
                dev_values,
                dev_mats_order,
                dev_mats);
            break;
        case 2:
            printf("\nExecute pti_MTTKRPKernelRankHiCOO_3D_naive (%u, %u, %u)\n", nblocks, nthreadsx, nthreadsy);

            pti_MTTKRPKernelRankHiCOO_3D_naive<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                sb_bits,
                sc_bits,
                blength,
                kptr_begin,
                kptr_end,
                dev_ndims,
                dev_cptr,
                dev_bptr,
                dev_binds,
                dev_einds,
                dev_values,
                dev_mats_order,
                dev_mats);
            break;
        case 3:
            printf("\nExecute pti_MTTKRPKernelRankSplitHiCOO_3D_naive (%u, %u, %u)\n", nblocks, nthreadsx, nthreadsy);

            pti_MTTKRPKernelRankSplitHiCOO_3D_naive<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                sb_bits,
                sc_bits,
                blength,
                kptr_begin,
                kptr_end,
                dev_ndims,
                dev_cptr,
                dev_bptr,
                dev_binds,
                dev_einds,
                dev_values,
                dev_mats_order,
                dev_mats);
            break;
        case 4:
            printf("\nExecute pti_MTTKRPKernelRankSplitHiCOORB_3D_naive (%u, %u, %u)\n", nblocks, nthreadsx, nthreadsy);

            pti_MTTKRPKernelRankSplitHiCOORB_3D_naive<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                sb_bits,
                sc_bits,
                blength,
                kptr_begin,
                kptr_end,
                dev_ndims,
                dev_cptr,
                dev_bptr,
                dev_binds,
                dev_einds,
                dev_values,
                dev_mats_order,
                dev_mats);
            break;

        /* Matrix blocked implementations */
        case 14:
            printf("\nExecute pti_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked (%u, %u, %u)\n", nblocks, nthreadsx, nthreadsy);

            pti_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                sb_bits,
                sc_bits,
                blength,
                kptr_begin,
                kptr_end,
                dev_ndims,
                dev_cptr,
                dev_bptr,
                dev_binds,
                dev_einds,
                dev_values,
                dev_mats_order,
                dev_mats);
            break;

        /* Shared memory for the output matrix + switch block sizes */
        case 15:
            printf("\nExecute pti_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked_SM (%u, %u, %u), SM: %u bytes\n", nblocks, nthreadsx, nthreadsy, shr_size);

            pti_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked_SM<<<nblocks, dimBlock, shr_size>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                sb_bits,
                sc_bits,
                blength,
                kptr_begin,
                kptr_end,
                dev_ndims,
                dev_cptr,
                dev_bptr,
                dev_binds,
                dev_einds,
                dev_values,
                dev_mats_order,
                dev_mats);
            break;

        /* Shared memory for three matrices + switch block sizes */
        case 16:
            printf("\nExecute pti_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked_AllSM (%u, %u, %u), SM: %u bytes\n", nblocks, nthreadsx, nthreadsy, shr_size);

            pti_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked_AllSM<<<nblocks, dimBlock, shr_size>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                sb_bits,
                sc_bits,
                blength,
                kptr_begin,
                kptr_end,
                dev_ndims,
                dev_cptr,
                dev_bptr,
                dev_binds,
                dev_einds,
                dev_values,
                dev_mats_order,
                dev_mats);
            break;

         }

    break;
    }   // End switch nmodes
    result = cudaThreadSynchronize();
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");

    return 0;
}


/* impl_num = 01  Naive, 1-D 
 * Limitation: blockDim.x (max_nnz) <= 1024.
 */
__global__ void pti_MTTKRPKernelHiCOO_3D_naive(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sc_bits,
    const ptiNnzIndex blength,
    const ptiNnzIndex kptr_begin,
    const ptiNnzIndex kptr_end,
    ptiIndex * const dev_ndims,
    ptiNnzIndex * const dev_cptr,
    ptiNnzIndex * const dev_bptr,
    ptiBlockIndex ** const dev_binds,
    ptiElementIndex ** const dev_einds,
    ptiValue * const dev_values,
    ptiIndex * const dev_mats_order,
    ptiValue ** const dev_mats)
{
    ptiNnzIndex const all_nblocks = blength;
    const ptiIndex tidx = threadIdx.x;
    ptiNnzIndex z;
    ptiIndex block_coord_mode, block_coord_1, block_coord_2;

    ptiValue * const mvals = dev_mats[nmodes];
    ptiIndex const times_mat_index_1 = dev_mats_order[1];
    ptiValue * const times_mat_1 = dev_mats[times_mat_index_1];
    ptiIndex const times_mat_index_2 = dev_mats_order[2];
    ptiValue * const times_mat_2 = dev_mats[times_mat_index_2];

    ptiNnzIndex num_loops_blocks = 1;
    if(all_nblocks > gridDim.x) {
        num_loops_blocks = (all_nblocks + gridDim.x - 1) / gridDim.x;
    }
    for(ptiNnzIndex nb=0; nb<num_loops_blocks; ++nb) {
        /* Block level */
        ptiNnzIndex b = blockIdx.x + nb * gridDim.x;
        if(b < blength) {
            /* Block indices */
            block_coord_mode = dev_binds[mode][b] << sb_bits;
            block_coord_1 = dev_binds[times_mat_index_1][b] << sb_bits;
            block_coord_2 = dev_binds[times_mat_index_2][b] << sb_bits;

            /* TODO: duplicated in registers */
            ptiNnzIndex const bptr_begin = dev_bptr[b];
            ptiNnzIndex const bptr_end = dev_bptr[b+1];

            /* Thread level */
            z = tidx + bptr_begin;
            if(z < bptr_end) {
                ptiValue const entry = dev_values[z];
                ptiNnzIndex const mode_i = block_coord_mode + dev_einds[mode][z];
                ptiNnzIndex const tmp_i_1 = block_coord_1 + dev_einds[times_mat_index_1][z];
                ptiNnzIndex const tmp_i_2 = block_coord_2 + dev_einds[times_mat_index_2][z];

                ptiValue tmp_val = 0;
                for(ptiIndex r=0; r<R; ++r) {
                    tmp_val = entry * times_mat_1[tmp_i_1 * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                    atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                }

            }   // End loop entries
        }
    }   // End loop blocks

}

/* impl_num = 02  Naive, 2-D 
 * Limitation: blockDim.x (max_nnz) * R <= 1024.
 */
__global__ void pti_MTTKRPKernelRankHiCOO_3D_naive(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sc_bits,
    const ptiNnzIndex blength,
    const ptiNnzIndex kptr_begin,
    const ptiNnzIndex kptr_end,
    ptiIndex * const dev_ndims,
    ptiNnzIndex * const dev_cptr,
    ptiNnzIndex * const dev_bptr,
    ptiBlockIndex ** const dev_binds,
    ptiElementIndex ** const dev_einds,
    ptiValue * const dev_values,
    ptiIndex * const dev_mats_order,
    ptiValue ** const dev_mats)
{
    ptiNnzIndex const all_nblocks = blength;
    const ptiIndex tidx = threadIdx.x;
    const ptiIndex tidy = threadIdx.y;
    ptiNnzIndex z;
    ptiIndex block_coord_mode, block_coord_1, block_coord_2;

    ptiValue * const mvals = dev_mats[nmodes];
    ptiIndex const times_mat_index_1 = dev_mats_order[1];
    ptiValue * const times_mat_1 = dev_mats[times_mat_index_1];
    ptiIndex const times_mat_index_2 = dev_mats_order[2];
    ptiValue * const times_mat_2 = dev_mats[times_mat_index_2];

    ptiNnzIndex num_loops_blocks = 1;
    if(all_nblocks > gridDim.x) {
        num_loops_blocks = (all_nblocks + gridDim.x - 1) / gridDim.x;
    }
    for(ptiNnzIndex nb=0; nb<num_loops_blocks; ++nb) {
        /* Block level */
        ptiNnzIndex b = blockIdx.x + nb * gridDim.x;
        if(b < blength) {
            /* Block indices */
            block_coord_mode = dev_binds[mode][b] << sb_bits;
            block_coord_1 = dev_binds[times_mat_index_1][b] << sb_bits;
            block_coord_2 = dev_binds[times_mat_index_2][b] << sb_bits;

            /* TODO: duplicated in registers */
            ptiNnzIndex const bptr_begin = dev_bptr[b];
            ptiNnzIndex const bptr_end = dev_bptr[b+1];

            /* Thread level */
            z = tidx + bptr_begin;
            if(z < bptr_end) {
                /* TODO: duplicated in R threads */
                ptiValue const entry = dev_values[z];
                ptiNnzIndex const mode_i = block_coord_mode + dev_einds[mode][z];
                ptiNnzIndex const tmp_i_1 = block_coord_1 + dev_einds[times_mat_index_1][z];
                ptiNnzIndex const tmp_i_2 = block_coord_2 + dev_einds[times_mat_index_2][z];

                ptiValue tmp_val = 0;
                tmp_val = entry * times_mat_1[tmp_i_1 * stride + tidy] * times_mat_2[tmp_i_2 * stride + tidy];
                atomicAdd(&(mvals[mode_i * stride + tidy]), tmp_val);

            }   // End loop entries
        }
    }   // End loop blocks

}

/* impl_num = 03  Naive, 2-D, exchange tidx and tidy.
 * Limitation: R * blockDim.y (max_nnz) <= 1024.
 */
__global__ void pti_MTTKRPKernelRankSplitHiCOO_3D_naive(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sc_bits,
    const ptiNnzIndex blength,
    const ptiNnzIndex kptr_begin,
    const ptiNnzIndex kptr_end,
    ptiIndex * const dev_ndims,
    ptiNnzIndex * const dev_cptr,
    ptiNnzIndex * const dev_bptr,
    ptiBlockIndex ** const dev_binds,
    ptiElementIndex ** const dev_einds,
    ptiValue * const dev_values,
    ptiIndex * const dev_mats_order,
    ptiValue ** const dev_mats)
{
    ptiNnzIndex const all_nblocks = blength;
    const ptiIndex tidx = threadIdx.x;
    const ptiIndex tidy = threadIdx.y;
    ptiNnzIndex z;
    ptiIndex block_coord_mode, block_coord_1, block_coord_2;

    ptiValue * const mvals = dev_mats[nmodes];
    ptiIndex const times_mat_index_1 = dev_mats_order[1];
    ptiValue * const times_mat_1 = dev_mats[times_mat_index_1];
    ptiIndex const times_mat_index_2 = dev_mats_order[2];
    ptiValue * const times_mat_2 = dev_mats[times_mat_index_2];

    ptiNnzIndex num_loops_blocks = 1;
    if(all_nblocks > gridDim.x) {
        num_loops_blocks = (all_nblocks + gridDim.x - 1) / gridDim.x;
    }
    for(ptiNnzIndex nb=0; nb<num_loops_blocks; ++nb) {
        /* Block level */
        ptiNnzIndex b = blockIdx.x + nb * gridDim.x;
        if(b < blength) {
            /* Block indices */
            block_coord_mode = dev_binds[mode][b] << sb_bits;
            block_coord_1 = dev_binds[times_mat_index_1][b] << sb_bits;
            block_coord_2 = dev_binds[times_mat_index_2][b] << sb_bits;

            /* TODO: duplicated in registers */
            ptiNnzIndex const bptr_begin = dev_bptr[b];
            ptiNnzIndex const bptr_end = dev_bptr[b+1];

            /* Thread level */
            z = tidy + bptr_begin;
            if(z < bptr_end) {
                ptiValue const entry = dev_values[z];
                ptiNnzIndex const mode_i = block_coord_mode + dev_einds[mode][z];
                ptiNnzIndex const tmp_i_1 = block_coord_1 + dev_einds[times_mat_index_1][z];
                ptiNnzIndex const tmp_i_2 = block_coord_2 + dev_einds[times_mat_index_2][z];

                ptiValue tmp_val = 0;
                tmp_val = entry * times_mat_1[tmp_i_1 * stride + tidx] * times_mat_2[tmp_i_2 * stride + tidx];
                atomicAdd(&(mvals[mode_i * stride + tidx]), tmp_val);

            }   // End loop entries
        }
    }   // End loop blocks

}

/* impl_num = 04  Naive, 2-D, with rank blocking.
 * Limitation: max_R * blockDim.y (max_nnz) <= 1024.
 */
__global__ void pti_MTTKRPKernelRankSplitHiCOORB_3D_naive(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sc_bits,
    const ptiNnzIndex blength,
    const ptiNnzIndex kptr_begin,
    const ptiNnzIndex kptr_end,
    ptiIndex * const dev_ndims,
    ptiNnzIndex * const dev_cptr,
    ptiNnzIndex * const dev_bptr,
    ptiBlockIndex ** const dev_binds,
    ptiElementIndex ** const dev_einds,
    ptiValue * const dev_values,
    ptiIndex * const dev_mats_order,
    ptiValue ** const dev_mats)
{
    ptiNnzIndex const all_nblocks = blength;
    const ptiIndex tidx = threadIdx.x;
    const ptiIndex tidy = threadIdx.y;
    ptiNnzIndex z;
    ptiIndex block_coord_mode, block_coord_1, block_coord_2;
    const ptiIndex num_loops_r = R / blockDim.x;
    const ptiIndex rest_loop = R - num_loops_r * blockDim.x;

    ptiValue * const mvals = dev_mats[nmodes];
    ptiIndex const times_mat_index_1 = dev_mats_order[1];
    ptiValue * const times_mat_1 = dev_mats[times_mat_index_1];
    ptiIndex const times_mat_index_2 = dev_mats_order[2];
    ptiValue * const times_mat_2 = dev_mats[times_mat_index_2];

    ptiNnzIndex num_loops_blocks = 1;
    if(all_nblocks > gridDim.x) {
        num_loops_blocks = (all_nblocks + gridDim.x - 1) / gridDim.x;
    }

    for(ptiNnzIndex nb=0; nb<num_loops_blocks; ++nb) {
        /* Block level */
        ptiNnzIndex b = blockIdx.x + nb * gridDim.x;
        if(b < blength) {
            /* Block indices */
            block_coord_mode = dev_binds[mode][b] << sb_bits;
            block_coord_1 = dev_binds[times_mat_index_1][b] << sb_bits;
            block_coord_2 = dev_binds[times_mat_index_2][b] << sb_bits;

            /* TODO: duplicated in registers */
            ptiNnzIndex const bptr_begin = dev_bptr[b];
            ptiNnzIndex const bptr_end = dev_bptr[b+1];

            /* Thread level */
            z = tidy + bptr_begin;
            if(z < bptr_end) {
                ptiValue const entry = dev_values[z];
                ptiNnzIndex const mode_i = block_coord_mode + dev_einds[mode][z];
                ptiNnzIndex const tmp_i_1 = block_coord_1 + dev_einds[times_mat_index_1][z];
                ptiNnzIndex const tmp_i_2 = block_coord_2 + dev_einds[times_mat_index_2][z];

                ptiIndex r;
                ptiValue tmp_val = 0;
                for(ptiIndex l=0; l<num_loops_r; ++l) {
                    r = tidx + l * blockDim.x;
                    tmp_val = entry * times_mat_1[tmp_i_1 * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                    atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                }

                if(rest_loop > 0 && tidx < rest_loop) {
                    r = tidx + num_loops_r * blockDim.x;
                    tmp_val = entry * times_mat_1[tmp_i_1 * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                    atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
                }

            }   // End loop entries
        }
    }   // End loop blocks

}



/* impl_num = 14  Matrix Blocked, 2-D, with rank blocking.
 * Limitation: max_R * blockDim.y (max_nnz) <= 1024.
 */
__global__ void pti_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sc_bits,
    const ptiNnzIndex blength,
    const ptiNnzIndex kptr_begin,
    const ptiNnzIndex kptr_end,
    ptiIndex * const dev_ndims,
    ptiNnzIndex * const dev_cptr,
    ptiNnzIndex * const dev_bptr,
    ptiBlockIndex ** const dev_binds,
    ptiElementIndex ** const dev_einds,
    ptiValue * const dev_values,
    ptiIndex * const dev_mats_order,
    ptiValue ** const dev_mats)
{
    ptiNnzIndex const all_nblocks = blength;
    const ptiIndex tidx = threadIdx.x;
    const ptiIndex tidy = threadIdx.y;
    ptiNnzIndex z;
    const ptiIndex num_loops_r = R / blockDim.x;
    const ptiIndex rest_loop = R - num_loops_r * blockDim.x;

    ptiValue * const mvals = dev_mats[nmodes];
    ptiIndex const times_mat_index_1 = dev_mats_order[1];
    ptiValue * const times_mat_1 = dev_mats[times_mat_index_1];
    ptiIndex const times_mat_index_2 = dev_mats_order[2];
    ptiValue * const times_mat_2 = dev_mats[times_mat_index_2];

    ptiNnzIndex num_loops_blocks = 1;
    if(all_nblocks > gridDim.x) {
        num_loops_blocks = (all_nblocks + gridDim.x - 1) / gridDim.x;
    }

    for(ptiNnzIndex nb=0; nb<num_loops_blocks; ++nb) {
        /* Block level */
        ptiNnzIndex b = blockIdx.x + nb * gridDim.x;
        if(b < blength) {
            /* TODO: duplicated in registers */
            ptiValue * blocked_mvals = mvals + (dev_binds[mode][b] << sb_bits) * stride;
            ptiValue * blocked_times_mat_1 = times_mat_1 + (dev_binds[times_mat_index_1][b] << sb_bits) * stride;
            ptiValue * blocked_times_mat_2 = times_mat_2 + (dev_binds[times_mat_index_2][b] << sb_bits) * stride;

            ptiNnzIndex const bptr_begin = dev_bptr[b];
            ptiNnzIndex const bptr_end = dev_bptr[b+1];

            /* Thread level */
            z = tidy + bptr_begin;
            if(z < bptr_end) {
                ptiValue const entry = dev_values[z];
                ptiElementIndex const mode_i = dev_einds[mode][z];
                ptiElementIndex const tmp_i_1 = dev_einds[times_mat_index_1][z];
                ptiElementIndex const tmp_i_2 = dev_einds[times_mat_index_2][z];

                ptiValue * const bmvals_row = blocked_mvals + mode_i * stride;

                ptiIndex r;
                ptiValue tmp_val = 0;
                for(ptiIndex l=0; l<num_loops_r; ++l) {
                    r = tidx + l * blockDim.x;
                    tmp_val = entry * blocked_times_mat_1[tmp_i_1 * stride + r] * blocked_times_mat_2[tmp_i_2 * stride + r];
                    atomicAdd(&(bmvals_row[r]), tmp_val);
                }

                if(rest_loop > 0 && tidx < rest_loop) {
                    r = tidx + num_loops_r * blockDim.x;
                    tmp_val = entry * blocked_times_mat_1[tmp_i_1 * stride + r] * blocked_times_mat_2[tmp_i_2 * stride + r];
                    atomicAdd(&(bmvals_row[r]), tmp_val);
                }

            }   // End loop entries
        }
    }   // End loop blocks

}



/* impl_num = 15  Matrix Blocked, 2-D, with rank blocking.
 * + switch according to block size
 * use shared memory for the output matrix
 * Limitation: max_R * blockDim.y (max_nnz) <= 1024.
 */
__global__ void pti_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked_SM(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sc_bits,
    const ptiNnzIndex blength,
    const ptiNnzIndex kptr_begin,
    const ptiNnzIndex kptr_end,
    ptiIndex * const dev_ndims,
    ptiNnzIndex * const dev_cptr,
    ptiNnzIndex * const dev_bptr,
    ptiBlockIndex ** const dev_binds,
    ptiElementIndex ** const dev_einds,
    ptiValue * const dev_values,
    ptiIndex * const dev_mats_order,
    ptiValue ** const dev_mats)
{
    ptiIndex const sb_size = (ptiIndex) 1 << sb_bits;
    /* Data in shared memory */
    extern __shared__ ptiValue mempool[];
    ptiValue * sm_blocked_mvals = mempool;

    ptiNnzIndex const all_nblocks = blength;
    const ptiIndex tidx = threadIdx.x;
    const ptiIndex tidy = threadIdx.y;
    ptiNnzIndex z;
    const ptiIndex num_loops_r = R / blockDim.x;
    const ptiIndex rest_loop = R - num_loops_r * blockDim.x;
    ptiIndex r;

    ptiValue * const mvals = dev_mats[nmodes];
    ptiIndex const times_mat_index_1 = dev_mats_order[1];
    ptiValue * const times_mat_1 = dev_mats[times_mat_index_1];
    ptiIndex const times_mat_index_2 = dev_mats_order[2];
    ptiValue * const times_mat_2 = dev_mats[times_mat_index_2];

    ptiNnzIndex num_loops_blocks = 1;
    if(all_nblocks > gridDim.x) {
        num_loops_blocks = (all_nblocks + gridDim.x - 1) / gridDim.x;
    }

    for(ptiNnzIndex nb=0; nb<num_loops_blocks; ++nb) {
        /* Block level */
        ptiNnzIndex b = blockIdx.x + nb * gridDim.x;
        if(b < blength) {
            /* TODO: duplicated in registers */
            ptiIndex blocked_mode_i = dev_binds[mode][b] << sb_bits;
            ptiValue * blocked_mvals = mvals + (dev_binds[mode][b] << sb_bits) * stride;
            ptiValue * blocked_times_mat_1 = times_mat_1 + (dev_binds[times_mat_index_1][b] << sb_bits) * stride;
            ptiValue * blocked_times_mat_2 = times_mat_2 + (dev_binds[times_mat_index_2][b] << sb_bits) * stride;

            ptiNnzIndex const bptr_begin = dev_bptr[b];
            ptiNnzIndex const bptr_end = dev_bptr[b+1];

            /* Enough matrix reuse */
            if (bptr_end - bptr_begin > sb_size) {

                /* Load mats[nmodes] into shared memory, use R instead of stride. */
                if (tidy < sb_size) {
                    for(ptiIndex l=0; l<num_loops_r; ++l) {
                        r = tidx + l * blockDim.x;
                        sm_blocked_mvals[tidy * R + r] = 0;
                    }
                    if(rest_loop > 0 && tidx < rest_loop) {
                        r = tidx + num_loops_r * blockDim.x;
                        sm_blocked_mvals[tidy * R + r] = 0;
                    }
                }
                __syncthreads();

                /* Thread level */
                z = tidy + bptr_begin;
                if(z < bptr_end) {
                    ptiValue const entry = dev_values[z];
                    ptiElementIndex const mode_i = dev_einds[mode][z];
                    ptiElementIndex const tmp_i_1 = dev_einds[times_mat_index_1][z];
                    ptiElementIndex const tmp_i_2 = dev_einds[times_mat_index_2][z];

                    ptiValue * const bmvals_row = sm_blocked_mvals + mode_i * R;
                    ptiValue * const blocked_times_mat_1_row = blocked_times_mat_1 + tmp_i_1 * stride;
                    ptiValue * const blocked_times_mat_2_row = blocked_times_mat_2 + tmp_i_2 * stride;

                    ptiValue tmp_val = 0;
                    for(ptiIndex l=0; l<num_loops_r; ++l) {
                        r = tidx + l * blockDim.x;
                        tmp_val = entry * blocked_times_mat_1_row[r] * blocked_times_mat_2_row[r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                    if(rest_loop > 0 && tidx < rest_loop) {
                        r = tidx + num_loops_r * blockDim.x;
                        tmp_val = entry * blocked_times_mat_1_row[r] * blocked_times_mat_2_row[r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                }   // End loop entries

                /* Store back mats[nmodes] from shared memory */
                if (tidy < sb_size) {
                    for(ptiIndex l=0; l<num_loops_r; ++l) {
                        r = tidx + l * blockDim.x;
                        atomicAdd( &(blocked_mvals[tidy * stride + r]),  sm_blocked_mvals[tidy * stride + r] );
                    }
                    if(rest_loop > 0 && tidx < rest_loop) {
                        r = tidx + num_loops_r * blockDim.x;
                        atomicAdd( &(blocked_mvals[tidy * stride + r]),  sm_blocked_mvals[tidy * stride + r] );
                    }
                }


            } else { /* Not enough matrix reuse */
                /* Thread level */
                z = tidy + bptr_begin;
                if(z < bptr_end) {
                    ptiValue const entry = dev_values[z];
                    ptiElementIndex const mode_i = dev_einds[mode][z];
                    ptiElementIndex const tmp_i_1 = dev_einds[times_mat_index_1][z];
                    ptiElementIndex const tmp_i_2 = dev_einds[times_mat_index_2][z];

                    ptiValue * const bmvals_row = blocked_mvals + mode_i * stride;

                    ptiValue tmp_val = 0;
                    for(ptiIndex l=0; l<num_loops_r; ++l) {
                        r = tidx + l * blockDim.x;
                        tmp_val = entry * blocked_times_mat_1[tmp_i_1 * stride + r] * blocked_times_mat_2[tmp_i_2 * stride + r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                    if(rest_loop > 0 && tidx < rest_loop) {
                        r = tidx + num_loops_r * blockDim.x;
                        tmp_val = entry * blocked_times_mat_1[tmp_i_1 * stride + r] * blocked_times_mat_2[tmp_i_2 * stride + r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                }   // End loop entries
            }   // End if: block size

        }   // End if: block range
    }   // End loop blocks

}



/* impl_num = 16  Matrix Blocked, 2-D, with rank blocking. TODO: BUG EXISTS.
 * + switch according to block size
 * use shared memory for three matrices
 * Limitation: max_R * blockDim.y (max_nnz) <= 1024.
 */
__global__ void pti_MTTKRPKernelRankSplitHiCOORB_3D_MatrixBlocked_AllSM(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sc_bits,
    const ptiNnzIndex blength,
    const ptiNnzIndex kptr_begin,
    const ptiNnzIndex kptr_end,
    ptiIndex * const dev_ndims,
    ptiNnzIndex * const dev_cptr,
    ptiNnzIndex * const dev_bptr,
    ptiBlockIndex ** const dev_binds,
    ptiElementIndex ** const dev_einds,
    ptiValue * const dev_values,
    ptiIndex * const dev_mats_order,
    ptiValue ** const dev_mats)
{
    ptiIndex const sb_size = (ptiIndex)powf(2, sb_bits);
    /* Data in shared memory */
    extern __shared__ ptiValue mempool[];
    ptiValue * sm_blocked_mvals = mempool;
    ptiValue * sm_blocked_times_mat_1 = mempool + sb_size * R * sizeof(ptiValue);
    ptiValue * sm_blocked_times_mat_2 = sm_blocked_times_mat_1 + sb_size * R * sizeof(ptiValue);

    ptiNnzIndex const all_nblocks = blength;
    const ptiIndex tidx = threadIdx.x;
    const ptiIndex tidy = threadIdx.y;
    ptiNnzIndex z;
    const ptiIndex num_loops_r = R / blockDim.x;
    const ptiIndex rest_loop = R - num_loops_r * blockDim.x;
    ptiIndex r;

    ptiValue * const mvals = dev_mats[nmodes];
    ptiIndex const times_mat_index_1 = dev_mats_order[1];
    ptiValue * const times_mat_1 = dev_mats[times_mat_index_1];
    ptiIndex const times_mat_index_2 = dev_mats_order[2];
    ptiValue * const times_mat_2 = dev_mats[times_mat_index_2];

    ptiNnzIndex num_loops_blocks = 1;
    if(all_nblocks > gridDim.x) {
        num_loops_blocks = (all_nblocks + gridDim.x - 1) / gridDim.x;
    }

    for(ptiNnzIndex nb=0; nb<num_loops_blocks; ++nb) {
        /* Block level */
        ptiNnzIndex b = blockIdx.x + nb * gridDim.x;
        if(b < blength) {
            /* TODO: duplicated in registers */
            ptiIndex blocked_mode_i = dev_binds[mode][b] << sb_bits;
            ptiValue * blocked_mvals = mvals + (dev_binds[mode][b] << sb_bits) * stride;
            ptiValue * blocked_times_mat_1 = times_mat_1 + (dev_binds[times_mat_index_1][b] << sb_bits) * stride;
            ptiValue * blocked_times_mat_2 = times_mat_2 + (dev_binds[times_mat_index_2][b] << sb_bits) * stride;

            ptiNnzIndex const bptr_begin = dev_bptr[b];
            ptiNnzIndex const bptr_end = dev_bptr[b+1];

            /* Enough matrix reuse */
            if (bptr_end - bptr_begin > sb_size) {

                /* Load mats[nmodes] into shared memory, use R instead of stride. */
                if (tidy < sb_size) {
                    for(ptiIndex l=0; l<num_loops_r; ++l) {
                        r = tidx + l * blockDim.x;
                        sm_blocked_mvals[tidy * R + r] = 0;
                        sm_blocked_times_mat_1[tidy * R + r] = blocked_times_mat_1[tidy * stride + r];
                        sm_blocked_times_mat_2[tidy * R + r] = blocked_times_mat_2[tidy * stride + r];
                    }
                    if(rest_loop > 0 && tidx < rest_loop) {
                        r = tidx + num_loops_r * blockDim.x;
                        sm_blocked_mvals[tidy * R + r] = 0;
                        sm_blocked_times_mat_1[tidy * R + r] = blocked_times_mat_1[tidy * stride + r];
                        sm_blocked_times_mat_2[tidy * R + r] = blocked_times_mat_2[tidy * stride + r];
                    }
                }
                __syncthreads();

                /* Thread level */
                z = tidy + bptr_begin;
                if(z < bptr_end) {
                    ptiValue const entry = dev_values[z];
                    ptiElementIndex const mode_i = dev_einds[mode][z];
                    ptiElementIndex const tmp_i_1 = dev_einds[times_mat_index_1][z];
                    ptiElementIndex const tmp_i_2 = dev_einds[times_mat_index_2][z];

                    ptiValue * const bmvals_row = sm_blocked_mvals + mode_i * R;
                    ptiValue * const blocked_times_mat_1_row = sm_blocked_times_mat_1 + tmp_i_1 * R;
                    ptiValue * const blocked_times_mat_2_row = sm_blocked_times_mat_2 + tmp_i_2 * R;

                    ptiValue tmp_val = 0;
                    for(ptiIndex l=0; l<num_loops_r; ++l) {
                        r = tidx + l * blockDim.x;
                        tmp_val = entry * blocked_times_mat_1_row[r] * blocked_times_mat_2_row[r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                    if(rest_loop > 0 && tidx < rest_loop) {
                        r = tidx + num_loops_r * blockDim.x;
                        tmp_val = entry * blocked_times_mat_1_row[r] * blocked_times_mat_2_row[r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                }   // End loop entries

                /* Store back mats[nmodes] from shared memory */
                if (tidy < sb_size) {
                    for(ptiIndex l=0; l<num_loops_r; ++l) {
                        r = tidx + l * blockDim.x;
                        atomicAdd( &(blocked_mvals[tidy * stride + r]),  sm_blocked_mvals[tidy * R + r] );
                    }
                    if(rest_loop > 0 && tidx < rest_loop) {
                        r = tidx + num_loops_r * blockDim.x;
                        atomicAdd( &(blocked_mvals[tidy * stride + r]),  sm_blocked_mvals[tidy * R + r] );
                    }
                }


            } else { /* Not enough matrix reuse */
                /* Thread level */
                z = tidy + bptr_begin;
                if(z < bptr_end) {
                    ptiValue const entry = dev_values[z];
                    ptiElementIndex const mode_i = dev_einds[mode][z];
                    ptiElementIndex const tmp_i_1 = dev_einds[times_mat_index_1][z];
                    ptiElementIndex const tmp_i_2 = dev_einds[times_mat_index_2][z];

                    ptiValue * const bmvals_row = blocked_mvals + mode_i * stride;

                    ptiValue tmp_val = 0;
                    for(ptiIndex l=0; l<num_loops_r; ++l) {
                        r = tidx + l * blockDim.x;
                        tmp_val = entry * blocked_times_mat_1[tmp_i_1 * stride + r] * blocked_times_mat_2[tmp_i_2 * stride + r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                    if(rest_loop > 0 && tidx < rest_loop) {
                        r = tidx + num_loops_r * blockDim.x;
                        tmp_val = entry * blocked_times_mat_1[tmp_i_1 * stride + r] * blocked_times_mat_2[tmp_i_2 * stride + r];
                        atomicAdd(&(bmvals_row[r]), tmp_val);
                    }

                }   // End loop entries
            }   // End if: block size

        }   // End if: block range
    }   // End loop blocks

}
