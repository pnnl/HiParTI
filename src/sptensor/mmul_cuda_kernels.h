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

#ifndef PARTI_MMUL_KERNELS_H
#define PARTI_MMUL_KERNELS_H

/* impl_num = 01 */
__global__ void pti_TTMNaiveKernel(
    ptiValue *Y_val, ptiIndex Y_stride, ptiNnzIndex Y_nnz,
    const ptiValue *X_val, ptiNnzIndex X_nnz, const ptiIndex *X_inds_m,
    const ptiNnzIndex *fiberidx_val, ptiNnzIndex fiberidx_len,
    const ptiValue *U_val, ptiIndex U_nrows, ptiIndex U_ncols, ptiIndex U_stride,
    ptiNnzIndex block_offset) ;

/* impl_num = 02 */
__global__ void pti_TTMKernel(
    ptiValue *Y_val, ptiIndex Y_stride, ptiNnzIndex Y_nnz,
    const ptiValue *X_val, ptiNnzIndex X_nnz, const ptiIndex *X_inds_m,
    const ptiNnzIndex *fiberidx_val, ptiNnzIndex fiberidx_len,
    const ptiValue *U_val, ptiIndex U_nrows, ptiIndex U_ncols, ptiIndex U_stride,
    ptiNnzIndex block_offset);



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
    ptiIndex U_stride);

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
    ptiIndex U_stride);

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
    ptiIndex U_stride);


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
    ptiIndex U_stride);

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
    ptiIndex U_stride);

#endif