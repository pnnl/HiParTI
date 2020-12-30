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

#ifndef PARTI_MTTKRP_KERNELS_H
#define PARTI_MTTKRP_KERNELS_H

__device__ void lock(int* mutex);
__device__ void unlock(int* mutex);


/* impl_num = 01 */
__global__ void pti_MTTKRPKernelNnz3D(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats,
    ptiNnzIndex block_offset);


/* impl_num = 02 */
__global__ void pti_MTTKRPKernelNnzRank3D(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats,
    ptiNnzIndex block_offset);


/* impl_num = 03 */
__global__ void pti_MTTKRPKernelNnzRankSplit3D(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats,
    ptiNnzIndex block_offset);


/* impl_num = 04 */
__global__ void pti_MTTKRPKernelRankNnz3D(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats,
    ptiNnzIndex block_offset);


/* impl_num = 05 */
__global__ void pti_MTTKRPKernelRankSplitNnz3D(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats,
    ptiNnzIndex block_offset);


/* impl_num = 06 */
__global__ void pti_MTTKRPKernelRankSplitNnzRB3D(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats,
    ptiNnzIndex block_offset);


/* impl_num = 09, for arbitraty nmodes. Scratch is necessary for tensors with arbitrary modes. */
__global__ void pti_MTTKRPKernelScratch(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats,
    ptiValue * dev_scratch,
    ptiNnzIndex block_offset);



/**** impl_num = 1x: One GPU using one kernel ****/
/* impl_num = 11 */
__global__ void pti_MTTKRPKernelNnz3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

/* impl_num = 12 */
__global__ void pti_MTTKRPKernelRankNnz3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

/* impl_num = 15 */
__global__ void pti_MTTKRPKernelRankSplitNnz3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

/* impl_num = 16 */
__global__ void pti_MTTKRPKernelRankSplitNnzRB3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);



/**** impl_num = 2x: Stream One GPU: cache blocking ****/
/* impl_num = 21. */
__global__ void pti_MTTKRPKernelBlockNnz3D(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex * nnz,
    const ptiNnzIndex * dev_nnz_blk_begin,
    const ptiIndex R,
    const ptiIndex stride,
    ptiIndex * const inds_low_allblocks,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);


/* impl_num = 25 */
__global__ void pti_MTTKRPKernelBlockRankSplitNnz3D(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex * nnz,
    const ptiNnzIndex * dev_nnz_blk_begin,
    const ptiIndex R,
    const ptiIndex stride,
    ptiIndex * const inds_low_allblocks,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);


/* impl_num = 26 */
__global__ void pti_MTTKRPKernelBlockRankSplitNnzRB3D(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex * nnz,
    const ptiNnzIndex * dev_nnz_blk_begin,
    const ptiIndex R,
    const ptiIndex stride,
    ptiIndex * const inds_low_allblocks,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);



/**** impl_num = 3x: Stream One GPU: shared memory blocking for coarse grain ****/
/* impl_num = 35 */
__global__ void pti_MTTKRPKernelBlockRankSplitNnz3D_SMCoarse(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex * nnz,
    const ptiNnzIndex * dev_nnz_blk_begin,
    const ptiIndex R,
    const ptiIndex stride,
    ptiIndex * const inds_low_allblocks,
    ptiIndex ** const inds_low,
    ptiIndex ** const Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

/* impl_num = 36 */
__global__ void pti_MTTKRPKernelBlockRankSplitNnz3D_SMCoarseRB(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex * nnz,
    const ptiNnzIndex * dev_nnz_blk_begin,
    const ptiIndex R,
    const ptiIndex stride,
    ptiIndex * const inds_low_allblocks,
    ptiIndex ** const inds_low,
    ptiIndex ** const Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);


/**** impl_num = 4x: Stream One GPU: shared memory blocking for medium grain ****/
/* impl_num = 45 */
__global__ void pti_MTTKRPKernelBlockRankSplitNnz3D_SMMedium(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex * nnz,
    const ptiNnzIndex * dev_nnz_blk_begin,
    const ptiIndex R,
    const ptiIndex stride,
    ptiIndex * const inds_low_allblocks,
    ptiIndex ** const inds_low,
    ptiIndex ** const Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);


/* impl_num = 46 */
__global__ void pti_MTTKRPKernelBlockRankSplitNnz3D_SMMediumRB(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex * nnz,
    const ptiNnzIndex * dev_nnz_blk_begin,
    const ptiIndex R,
    const ptiIndex stride,
    ptiIndex * const inds_low_allblocks,
    ptiIndex ** const inds_low,
    ptiIndex ** const Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);


/**** impl_num = 5x: multiple GPUs ****/
/* impl_num = 59, only the interface is a bit different. */
__global__ void pti_MTTKRPKernelScratchDist(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    const ptiIndex * inds_low,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats,
    ptiValue * dev_scratch);



/* impl_num = 31 */
__global__ void pti_MTTKRPKernelNnz3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

__global__ void pti_MTTKRPKernelRankSplitNnz3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

#endif