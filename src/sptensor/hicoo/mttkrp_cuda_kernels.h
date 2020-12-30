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

/* impl_num = 01  Naive, 1-D */
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
    ptiValue ** const dev_mats);

/* impl_num = 02  Naive, 2-D */
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
    ptiValue ** const dev_mats);

/* impl_num = 03  Naive, 2-D */
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
    ptiValue ** const dev_mats);

/* impl_num = 04  Naive, 2-D */
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
    ptiValue ** const dev_mats);


/* impl_num = 14  Matrix Blocked, 2-D, with rank blocking. */
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
    ptiValue ** const dev_mats);

/* impl_num = 15  Matrix Blocked, 2-D, with rank blocking + switch according to block size, use shared memory for the output matrix. */
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
    ptiValue ** const dev_mats);

/* impl_num = 16  Matrix Blocked, 2-D, with rank blocking + switch according to block size, use shared memory for three matrices. */
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
    ptiValue ** const dev_mats);


#endif
