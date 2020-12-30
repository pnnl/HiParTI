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
#include <omp.h>

#define CHUNKSIZE 1

int ptiOmpMTTKRPHiCOOKernels(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_3D(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOBlocks(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tb);
int ptiOmpMTTKRPHiCOOBlocks_3D(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tb);
int ptiOmpMTTKRPHiCOOKernelsBlocks(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb);
int ptiOmpMTTKRPHiCOOKernelsBlocks_3D(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb);


int ptiOmpMTTKRPHiCOOKernels_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Balanced(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Balanced(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Reduce(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Reduce_Balanced(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Reduce(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Reduce_Balanced(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Reduce_Two(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Reduce_Two(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRPHiCOOBlocks_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tb);
int ptiOmpMTTKRPHiCOOBlocks_3D_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tb);
int ptiOmpMTTKRPHiCOOKernelsBlocks_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb);
int ptiOmpMTTKRPHiCOOKernelsBlocks_3D_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb);


/**
 * Matriced sparse tensor in HiCOO format times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  hitsr    the HiCOO sparse tensor input
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products
 * @param[in]  mode   the mode on which the MTTKRP is performed
 * @param[in]  scratch an temporary array to store intermediate results, space assigned before this function
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 */
int ptiOmpMTTKRPHiCOO(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb)
{
    if(tk > 1 && tb == 1) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernels(hitsr, mats, mats_order, mode, tk) == 0);
    } else if(tk == 1 && tb > 1) {
        ptiAssert(ptiOmpMTTKRPHiCOOBlocks(hitsr, mats, mats_order, mode, tb) == 0);
    } else if(tk > 1 && tb > 1) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernelsBlocks(hitsr, mats, mats_order, mode, tk, tb) == 0);
    } else if(tk == 1 && tb == 1) {
        printf("Should specify sequetial MTTKRP.\n");
        return -1;
    }

    return 0;
}


int ptiOmpMTTKRPHiCOO_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb)
{
    if(tk > 1 && tb == 1) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernels_MatrixTiling(hitsr, mats, mats_order, mode, tk) == 0);
    } else if(tk == 1 && tb > 1) {
        ptiAssert(ptiOmpMTTKRPHiCOOBlocks_MatrixTiling(hitsr, mats, mats_order, mode, tb) == 0);
    } else if(tk > 1 && tb > 1) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernelsBlocks_MatrixTiling(hitsr, mats, mats_order, mode, tk, tb) == 0);
    } else if(tk == 1 && tb == 1) {
        printf("Should specify sequetial MTTKRP with -d -2.\n");
        return -1;
    }

    return 0;
}


int ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb,
    int balanced)
{
    if(tk > 1 && tb == 1) {
        if (balanced == 0)
            ptiAssert(ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled(hitsr, mats, mats_order, mode, tk) == 0);
        else if (balanced == 1)
            ptiAssert(ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Balanced(hitsr, mats, mats_order, mode, tk) == 0);
    } else {
        printf("Haven't support block parallelism.\n");
        return -1;
    }

    return 0;
}


int ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb,
    int balanced)
{
    if(tk > 1 && tb == 1) {
        if(balanced == 0)
            ptiAssert(ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Reduce(hitsr, mats, copy_mats, mats_order, mode, tk) == 0);
        else if (balanced == 1)
            ptiAssert(ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Reduce_Balanced(hitsr, mats, copy_mats, mats_order, mode, tk) == 0);
    } else {
        printf("Haven't support block parallelism.\n");
        return -1;
    }

    return 0;
}


int ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce_Two(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb)
{
    if(tk > 1 && tb == 1) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Reduce_Two(hitsr, mats, copy_mats, mats_order, mode, tk) == 0);
    } else {
        printf("Haven't support block parallelism.\n");
        return -1;
    }

    return 0;
}


int ptiOmpMTTKRPHiCOOKernels(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernels_3D(hitsr, mats, mats_order, mode, tk) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const vals = hitsr->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiMatrix * const M = mats[nmodes];
    ptiValue * const mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    // omp_lock_t lock;
    // omp_init_lock(&lock);

    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        /* Allocate thread-private data */
        ptiIndex * block_coord = (ptiIndex*)malloc(nmodes * sizeof(*block_coord));
        ptiIndex * ele_coord = (ptiIndex*)malloc(nmodes * sizeof(*ele_coord));
        ptiValueVector scratch; // Temporary array
        ptiNewValueVector(&scratch, R, R);

        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Block indices */
            for(ptiIndex m=0; m<nmodes; ++m)
                block_coord[m] = hitsr->binds[m].data[b];

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiNnzIndex z=bptr_begin; z<bptr_end; ++z) {
                /* Element indices */
                for(ptiIndex m=0; m<nmodes; ++m)
                    ele_coord[m] = (block_coord[m] << hitsr->sb_bits) + hitsr->einds[m].data[z];

                /* Multiply the 1st matrix */
                ptiIndex times_mat_index = mats_order[1];
                ptiMatrix * times_mat = mats[times_mat_index];
                ptiIndex tmp_i = ele_coord[times_mat_index];
                ptiValue const entry = vals[z];
                for(ptiIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(ptiIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    times_mat = mats[times_mat_index];
                    tmp_i = ele_coord[times_mat_index];
                    for(ptiIndex r=0; r<R; ++r) {
                        scratch.data[r] *= times_mat->values[tmp_i * stride + r];
                    }
                }

                ptiIndex const mode_i = ele_coord[mode];
                // omp_set_lock(&lock);
                for(ptiIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += scratch.data[r];
                }
                // omp_unset_lock(&lock);
            }   // End loop entries
        }   // End loop blocks

        /* Free thread-private space */
        free(block_coord);
        free(ele_coord);
        ptiFreeValueVector(&scratch);
    }   // End loop kernels

    // omp_destroy_lock(&lock);

    return 0;
}


int ptiOmpMTTKRPHiCOOKernels_3D(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiMatrix * restrict times_mat_2 = mats[times_mat_index_2];


    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {
            ptiBlockIndex block_coord_mode = hitsr->binds[mode].data[b];
            ptiBlockIndex block_coord_1 = hitsr->binds[times_mat_index_1].data[b];
            ptiBlockIndex block_coord_2 = hitsr->binds[times_mat_index_2].data[b];

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                
                ptiIndex mode_i = (block_coord_mode << hitsr->sb_bits) + hitsr->einds[mode].data[z];
                ptiIndex tmp_i_1 = (block_coord_1 << hitsr->sb_bits) + hitsr->einds[times_mat_index_1].data[z];
                ptiIndex tmp_i_2 = (block_coord_2 << hitsr->sb_bits) + hitsr->einds[times_mat_index_2].data[z];
                ptiValue entry = vals[z];
                for(ptiIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks
    }   // End loop kernels

    return 0;
}



int ptiOmpMTTKRPHiCOOKernels_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling(hitsr, mats, mats_order, mode, tk) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    /* Loop kernels */
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk)
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        /* Allocate thread-private data */
        ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));
        ptiValueVector scratch; // Temporary array
        ptiNewValueVector(&scratch, R, R);

        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Blocked matrices */
            for(ptiIndex m=0; m<nmodes; ++m)
                blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {

                /* Multiply the 1st matrix */
                ptiIndex times_mat_index = mats_order[1];
                ptiElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                ptiValue const entry = vals[z];
                #pragma omp simd
                for(ptiElementIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(ptiIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    tmp_i = hitsr->einds[times_mat_index].data[z];
                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        scratch.data[r] *= blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                    }
                }

                ptiElementIndex const mode_i = hitsr->einds[mode].data[z];
                ptiValue * const restrict bmvals_row = blocked_mvals + mode_i * stride;
                for(ptiElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    bmvals_row[r] += scratch.data[r];
                }
            }   // End loop entries
        }   // End loop blocks

        /* Free thread-private space */
        free(blocked_times_mat);
        ptiFreeValueVector(&scratch);
    }   // End loop kernels

    return 0;
}


int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];


    /* Loop kernels */
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk)
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                
                ptiElementIndex mode_i = hitsr->einds[mode].data[z];
                ptiElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                ptiElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                ptiValue entry = vals[z];
                ptiValue * const restrict bmvals_row = blocked_mvals + mode_i * stride;

                for(ptiElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    bmvals_row[r] += entry * 
                        blocked_times_mat_1[(ptiBlockMatrixIndex)tmp_i_1 * stride + r] *
                        blocked_times_mat_2[(ptiBlockMatrixIndex)tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}



int ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled(hitsr, mats, mats_order, mode, tk) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
    ptiIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    ptiIndexVector * restrict kschr_mode = hitsr->kschr[mode];

#ifdef NNZ_STATISTICS
    ptiNnzIndex * thread_nnzs = (ptiNnzIndex*)malloc(tk * sizeof(ptiNnzIndex));
    memset(thread_nnzs, 0, tk * sizeof(ptiNnzIndex));
#endif

    /* Loop parallel iterations */
    for(ptiIndex i=0; i<hitsr->nkiters[mode]; ++i) {
        /* Loop kernels */
#ifdef NNZ_STATISTICS
        #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
        #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk)
#endif
        for(ptiIndex k=0; k<num_kernel_dim; ++k) {
            int tid = omp_get_thread_num();

            if(i >= kschr_mode[k].len) continue;
            ptiIndex kptr_loc = kschr_mode[k].data[i];
            ptiNnzIndex kptr_begin = hitsr->kptr.data[kptr_loc];
            ptiNnzIndex kptr_end = hitsr->kptr.data[kptr_loc+1];

            /* Allocate thread-private data */
            ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));
            ptiValueVector scratch; // Temporary array
            ptiNewValueVector(&scratch, R, R);

            /* Loop blocks in a kernel */
            for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {
                /* Blocked matrices */
                for(ptiIndex m=0; m<nmodes; ++m)
                    blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
                ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

                ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
                ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
                thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

                /* Loop entries in a block */
                for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {

                    /* Multiply the 1st matrix */
                    ptiIndex times_mat_index = mats_order[1];
                    ptiElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                    ptiValue const entry = vals[z];
                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        scratch.data[r] = entry * blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                    }
                    /* Multiply the rest matrices */
                    for(ptiIndex m=2; m<nmodes; ++m) {
                        times_mat_index = mats_order[m];
                        tmp_i = hitsr->einds[times_mat_index].data[z];
                        #pragma omp simd
                        for(ptiElementIndex r=0; r<R; ++r) {
                            scratch.data[r] *= blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                        }
                    }

                    ptiElementIndex const mode_i = hitsr->einds[mode].data[z];
                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                    }
                }   // End loop entries
            }   // End loop blocks

            /* Free thread-private space */
            free(blocked_times_mat);
            ptiFreeValueVector(&scratch);
        }   // End loop kernels
    }   // End loop iterations

#ifdef NNZ_STATISTICS
    /* Calculate load balance of kernels */
    ptiNnzIndex sum_nnzs = 0, min_nnzs = hitsr->nnz, max_nnzs = 0;
    double std_nnzs = 0.0;
    double avg_nnzs = hitsr->nnz / (double)tk;
    // printf("thread_nnzs:\n");
    for(int i = 0; i < tk; ++i) {
        // printf("%"HIPARTI_PRI_NNZ_INDEX", ", thread_nnzs[i]);
        sum_nnzs += thread_nnzs[i];
        if(min_nnzs > thread_nnzs[i])
            min_nnzs = thread_nnzs[i];
        if(max_nnzs < thread_nnzs[i])
            max_nnzs = thread_nnzs[i];
        std_nnzs += (thread_nnzs[i] - avg_nnzs) * (thread_nnzs[i] - avg_nnzs);
    }
    // printf("\n");
    std_nnzs = sqrt(std_nnzs / tk);
    printf("min_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", max_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", avg_nnzs: %.1lf, std_nnzs: %.1lf\n", min_nnzs, max_nnzs, avg_nnzs, std_nnzs);
    ptiAssert(sum_nnzs == hitsr->nnz);

    free(thread_nnzs);
#endif

    return 0;
}


int ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Balanced(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Balanced(hitsr, mats, mats_order, mode, tk) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
    ptiIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    ptiIndexVector * restrict kschr_balanced_mode = hitsr->kschr_balanced[mode];
    ptiIndexVector * restrict kschr_balanced_pos_mode = hitsr->kschr_balanced_pos[mode];
    ptiIndex npars = hitsr->nkpars[mode];

#ifdef NNZ_STATISTICS
    ptiNnzIndex * thread_nnzs = (ptiNnzIndex*)malloc(tk * sizeof(ptiNnzIndex));
    memset(thread_nnzs, 0, tk * sizeof(ptiNnzIndex));
#endif

    /* Loop partitions */
    for(ptiIndex p=0; p<npars; ++p) {
        /* Loop kernels */
#ifdef NNZ_STATISTICS    
        #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
        #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
        for(ptiIndex i=0; i<num_kernel_dim; ++i) {
            if(p >= kschr_balanced_pos_mode[i].len - 1) continue;
            int tid = omp_get_thread_num();

            ptiIndex j_begin = kschr_balanced_pos_mode[i].data[p];
            ptiIndex j_end = kschr_balanced_pos_mode[i].data[p+1];

            /* Loop inside a partition */
            for(ptiIndex j = j_begin; j < j_end; ++j) {

                ptiIndex kernel_num = kschr_balanced_mode[i].data[j];
                ptiNnzIndex kptr_begin = hitsr->kptr.data[kernel_num];
                ptiNnzIndex kptr_end = hitsr->kptr.data[kernel_num+1];

                /* Allocate thread-private data */
                ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));
                ptiValueVector scratch; // Temporary array
                ptiNewValueVector(&scratch, R, R);

                /* Loop blocks in a kernel */
                for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {
                    /* Blocked matrices */
                    for(ptiIndex m=0; m<nmodes; ++m)
                        blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
                    ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

                    ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
                    ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
                    thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

                    /* Loop entries in a block */
                    for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {

                        /* Multiply the 1st matrix */
                        ptiIndex times_mat_index = mats_order[1];
                        ptiElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                        ptiValue const entry = vals[z];
                        #pragma omp simd
                        for(ptiElementIndex r=0; r<R; ++r) {
                            scratch.data[r] = entry * blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                        }
                        /* Multiply the rest matrices */
                        for(ptiIndex m=2; m<nmodes; ++m) {
                            times_mat_index = mats_order[m];
                            tmp_i = hitsr->einds[times_mat_index].data[z];
                            #pragma omp simd
                            for(ptiElementIndex r=0; r<R; ++r) {
                                scratch.data[r] *= blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                            }
                        }

                        ptiElementIndex const mode_i = hitsr->einds[mode].data[z];
                        #pragma omp simd
                        for(ptiElementIndex r=0; r<R; ++r) {
                            blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                        }
                    }   // End loop entries
                }   // End loop blocks

                /* Free thread-private space */
                free(blocked_times_mat);
                ptiFreeValueVector(&scratch);
            }   // End loop inside a partition
        }   // End loop kernels
    }   // End loop partitions


    /* Process using atomics */
#ifdef NNZ_STATISTICS
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
    for(ptiIndex k = 0; k < hitsr->kschr_rest[mode].len; ++k) {
        int tid = omp_get_thread_num();
        ptiIndex kernel_num = hitsr->kschr_rest[mode].data[k];
        ptiNnzIndex kptr_begin = hitsr->kptr.data[kernel_num];
        ptiNnzIndex kptr_end = hitsr->kptr.data[kernel_num+1];

        /* Allocate thread-private data */
        ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));
        ptiValueVector scratch; // Temporary array
        ptiNewValueVector(&scratch, R, R);

        /* Loop blocks in a kernel */
        for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Blocked matrices */
            for(ptiIndex m=0; m<nmodes; ++m)
                blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
            thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {

                /* Multiply the 1st matrix */
                ptiIndex times_mat_index = mats_order[1];
                ptiElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                ptiValue const entry = vals[z];
                #pragma omp simd
                for(ptiElementIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                }

                /* Multiply the rest matrices */
                for(ptiIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    tmp_i = hitsr->einds[times_mat_index].data[z];
                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        scratch.data[r] *= blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                    }
                }

                ptiElementIndex const mode_i = hitsr->einds[mode].data[z];
                ptiValue * const restrict bmvals_row = blocked_mvals + mode_i * stride;
                for(ptiElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    bmvals_row[r] += scratch.data[r];
                }
            }   // End loop entries
        }   // End loop blocks

        /* Free thread-private space */
        free(blocked_times_mat);
        ptiFreeValueVector(&scratch);
    }   // End loop kernels

#ifdef NNZ_STATISTICS
    /* Calculate load balance of kernels */
    ptiNnzIndex sum_nnzs = 0, min_nnzs = hitsr->nnz, max_nnzs = 0;
    double std_nnzs = 0.0;
    double avg_nnzs = hitsr->nnz / (double)tk;
    // printf("thread_nnzs:\n");
    for(int i = 0; i < tk; ++i) {
        // printf("%"HIPARTI_PRI_NNZ_INDEX", ", thread_nnzs[i]);
        sum_nnzs += thread_nnzs[i];
        if(min_nnzs > thread_nnzs[i])
            min_nnzs = thread_nnzs[i];
        if(max_nnzs < thread_nnzs[i])
            max_nnzs = thread_nnzs[i];
        std_nnzs += (thread_nnzs[i] - avg_nnzs) * (thread_nnzs[i] - avg_nnzs);
    }
    // printf("\n");
    std_nnzs = sqrt(std_nnzs / tk);
    printf("min_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", max_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", avg_nnzs: %.1lf, std_nnzs: %.1lf\n", min_nnzs, max_nnzs, avg_nnzs, std_nnzs);
    ptiAssert(sum_nnzs == hitsr->nnz);

    free(thread_nnzs);
#endif

    return 0;
}



int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];

    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
    ptiIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    ptiIndexVector * restrict kschr_mode = hitsr->kschr[mode];
    // printf("nkiters: %u, num_kernel_dim: %u\n", hitsr->nkiters[mode], num_kernel_dim);

#ifdef NNZ_STATISTICS
    ptiNnzIndex * thread_nnzs = (ptiNnzIndex*)malloc(tk * sizeof(ptiNnzIndex));
    memset(thread_nnzs, 0, tk * sizeof(ptiNnzIndex));
#endif

    /* Loop parallel iterations */
    for(ptiIndex i=0; i<hitsr->nkiters[mode]; ++i) {
        /* Loop kernels */
#ifdef NNZ_STATISTICS
        #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
        #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
        for(ptiIndex k=0; k<num_kernel_dim; ++k) {

            int tid = omp_get_thread_num();
            // printf("tid: %d, (i, k): (%u, %u)\n", tid, i, k);

            if(i >= kschr_mode[k].len) {
                // printf("i: %u, k: %u\n", i, k);
                continue;
            }

            ptiIndex kptr_loc = kschr_mode[k].data[i];
            ptiNnzIndex kptr_begin = hitsr->kptr.data[kptr_loc];
            ptiNnzIndex kptr_end = hitsr->kptr.data[kptr_loc+1];

            /* Loop blocks in a kernel */
            for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

                ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
                ptiValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
                ptiValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

                ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
                ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
                thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

                /* Loop entries in a block */
                for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                    
                    ptiElementIndex mode_i = hitsr->einds[mode].data[z];
                    ptiElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                    ptiElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                    ptiValue entry = vals[z];

                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += entry *
                            blocked_times_mat_1[(ptiBlockMatrixIndex)tmp_i_1 * stride + r] *
                            blocked_times_mat_2[(ptiBlockMatrixIndex)tmp_i_2 * stride + r];
                    }
                    
                }   // End loop entries
            }   // End loop blocks

        }   // End loop kernels
    }   // End loop iterations

#ifdef NNZ_STATISTICS
    /* Calculate load balance of kernels */
    ptiNnzIndex sum_nnzs = 0, min_nnzs = hitsr->nnz, max_nnzs = 0;
    double std_nnzs = 0.0;
    double avg_nnzs = hitsr->nnz / (double)tk;
    // printf("thread_nnzs:\n");
    for(int i = 0; i < tk; ++i) {
        // printf("%"HIPARTI_PRI_NNZ_INDEX", ", thread_nnzs[i]);
        sum_nnzs += thread_nnzs[i];
        if(min_nnzs > thread_nnzs[i])
            min_nnzs = thread_nnzs[i];
        if(max_nnzs < thread_nnzs[i])
            max_nnzs = thread_nnzs[i];
        std_nnzs += (thread_nnzs[i] - avg_nnzs) * (thread_nnzs[i] - avg_nnzs);
    }
    // printf("\n");
    std_nnzs = sqrt(std_nnzs / tk);
    printf("min_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", max_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", avg_nnzs: %.1lf, std_nnzs: %.1lf\n", min_nnzs, max_nnzs, avg_nnzs, std_nnzs);
    ptiAssert(sum_nnzs == hitsr->nnz);

    free(thread_nnzs);
#endif

    return 0;
}


int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Balanced(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];

    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
    ptiIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    ptiIndexVector * restrict kschr_balanced_mode = hitsr->kschr_balanced[mode];
    ptiIndexVector * restrict kschr_balanced_pos_mode = hitsr->kschr_balanced_pos[mode];
    ptiIndex npars = hitsr->nkpars[mode];
    // printf("nkiters: %u, num_kernel_dim: %u\n", hitsr->nkiters[mode], num_kernel_dim);

#ifdef NNZ_STATISTICS
    ptiNnzIndex * thread_nnzs = (ptiNnzIndex*)malloc(tk * sizeof(ptiNnzIndex));
    memset(thread_nnzs, 0, tk * sizeof(ptiNnzIndex));
#endif

    /* Loop partitions */
    for(ptiIndex p=0; p<npars; ++p) {
        /* Loop kernels */
#ifdef NNZ_STATISTICS
        #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
        #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
        for(ptiIndex i=0; i<num_kernel_dim; ++i) {
            if(p >= kschr_balanced_pos_mode[i].len - 1) continue;
            int tid = omp_get_thread_num();

            ptiIndex j_begin = kschr_balanced_pos_mode[i].data[p];
            ptiIndex j_end = kschr_balanced_pos_mode[i].data[p+1];

            /* Loop inside a partition */
            for(ptiIndex j = j_begin; j < j_end; ++j) {

                ptiIndex kernel_num = kschr_balanced_mode[i].data[j];
                // printf("tid: %d, (i, j): (%u, %u), kernel_num: %u\n", tid, i, j, kernel_num);
                ptiNnzIndex kptr_begin = hitsr->kptr.data[kernel_num];
                ptiNnzIndex kptr_end = hitsr->kptr.data[kernel_num+1];

                /* Loop blocks in a kernel */
                for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

                    ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
                    ptiValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
                    ptiValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

                    ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
                    ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
                    thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

                    /* Loop entries in a block */
                    for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                        
                        ptiElementIndex mode_i = hitsr->einds[mode].data[z];
                        ptiElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                        ptiElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                        ptiValue entry = vals[z];

                        #pragma omp simd
                        for(ptiElementIndex r=0; r<R; ++r) {
                            blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += entry *
                                blocked_times_mat_1[(ptiBlockMatrixIndex)tmp_i_1 * stride + r] *
                                blocked_times_mat_2[(ptiBlockMatrixIndex)tmp_i_2 * stride + r];
                        }
                        
                    }   // End loop entries
                }   // End loop blocks

            }   // End loop inside a partition
        }   // End loop kernels
    }   // End loop partitions

    /* Process using atomics */
#ifdef NNZ_STATISTICS
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
    for(ptiIndex k = 0; k < hitsr->kschr_rest[mode].len; ++k) {
        int tid = omp_get_thread_num();
        ptiIndex kernel_num = hitsr->kschr_rest[mode].data[k];
        ptiNnzIndex kptr_begin = hitsr->kptr.data[kernel_num];
        ptiNnzIndex kptr_end = hitsr->kptr.data[kernel_num+1];

        /* Loop blocks in a kernel */
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
            thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
     
                ptiElementIndex mode_i = hitsr->einds[mode].data[z];
                ptiElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                ptiElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                ptiValue entry = vals[z];
                ptiValue * const restrict bmvals_row = blocked_mvals + mode_i * stride;

                for(ptiElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    bmvals_row[r] += entry * 
                        blocked_times_mat_1[(ptiBlockMatrixIndex)tmp_i_1 * stride + r] *
                        blocked_times_mat_2[(ptiBlockMatrixIndex)tmp_i_2 * stride + r];
                }    
     
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels


#ifdef NNZ_STATISTICS
    /* Calculate load balance of kernels */
    ptiNnzIndex sum_nnzs = 0, min_nnzs = hitsr->nnz, max_nnzs = 0;
    double std_nnzs = 0.0;
    double avg_nnzs = hitsr->nnz / (double)tk;
    // printf("thread_nnzs:\n");
    for(int i = 0; i < tk; ++i) {
        // printf("%"HIPARTI_PRI_NNZ_INDEX", ", thread_nnzs[i]);
        sum_nnzs += thread_nnzs[i];
        if(min_nnzs > thread_nnzs[i])
            min_nnzs = thread_nnzs[i];
        if(max_nnzs < thread_nnzs[i])
            max_nnzs = thread_nnzs[i];
        std_nnzs += (thread_nnzs[i] - avg_nnzs) * (thread_nnzs[i] - avg_nnzs);
    }
    // printf("\n");
    std_nnzs = sqrt(std_nnzs / tk);
    printf("min_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", max_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", avg_nnzs: %.1lf, std_nnzs: %.1lf\n", min_nnzs, max_nnzs, avg_nnzs, std_nnzs);
    ptiAssert(sum_nnzs == hitsr->nnz);

    free(thread_nnzs);
#endif

    return 0;
}



int ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Reduce(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Reduce(hitsr, mats, copy_mats, mats_order, mode, tk) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));
    for(int t=0; t<tk; ++t) {
        memset(copy_mats[t]->values, 0, ndims[mode]*stride*sizeof(*(copy_mats[t]->values)));
    }

    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
    ptiIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    ptiIndexVector * restrict kschr_mode = hitsr->kschr[mode];

#ifdef NNZ_STATISTICS
    ptiNnzIndex * thread_nnzs = (ptiNnzIndex*)malloc(tk * sizeof(ptiNnzIndex));
    memset(thread_nnzs, 0, tk * sizeof(ptiNnzIndex));
#endif

    /* Loop parallel iterations */
#ifdef NNZ_STATISTICS
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
    for(ptiIndex i=0; i<hitsr->nkiters[mode]; ++i) {
        int tid = omp_get_thread_num();

        /* Loop kernels */
        for(ptiIndex k=0; k<num_kernel_dim; ++k) {

            if(i >= kschr_mode[k].len) continue;
            ptiIndex kptr_loc = kschr_mode[k].data[i];
            ptiNnzIndex kptr_begin = hitsr->kptr.data[kptr_loc];
            ptiNnzIndex kptr_end = hitsr->kptr.data[kptr_loc+1];

            /* Allocate thread-private data */
            ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));
            ptiValueVector scratch; // Temporary array
            ptiNewValueVector(&scratch, R, R);

            /* Loop blocks in a kernel */
            for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {
                /* Blocked matrices */
                for(ptiIndex m=0; m<nmodes; ++m)
                    blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
                ptiValue * blocked_mvals = copy_mats[tid]->values + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

                ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
                ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
                thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

                /* Loop entries in a block */
                for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {

                    /* Multiply the 1st matrix */
                    ptiIndex times_mat_index = mats_order[1];
                    ptiElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                    ptiValue const entry = vals[z];
                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        scratch.data[r] = entry * blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                    }
                    /* Multiply the rest matrices */
                    for(ptiIndex m=2; m<nmodes; ++m) {
                        times_mat_index = mats_order[m];
                        tmp_i = hitsr->einds[times_mat_index].data[z];
                        #pragma omp simd
                        for(ptiElementIndex r=0; r<R; ++r) {
                            scratch.data[r] *= blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                        }
                    }

                    ptiElementIndex const mode_i = hitsr->einds[mode].data[z];
                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                    }
                }   // End loop entries
            }   // End loop blocks

            /* Free thread-private space */
            free(blocked_times_mat);
            ptiFreeValueVector(&scratch);
        }   // End loop kernels
    }   // End loop iterations

    /* Reduction */
    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiIndex i=0; i<ndims[mode]; ++i) {
        for(int t=0; t<tk; ++t) {
            #pragma omp simd
            for(ptiElementIndex r=0; r<R; ++r) {
                mvals[i * stride + r] += copy_mats[t]->values[i * stride + r];
            }    
        }    
    }    

#ifdef NNZ_STATISTICS
    /* Calculate load balance of kernels */
    ptiNnzIndex sum_nnzs = 0, min_nnzs = hitsr->nnz, max_nnzs = 0;
    double std_nnzs = 0.0;
    double avg_nnzs = hitsr->nnz / (double)tk;
    // printf("thread_nnzs:\n");
    for(int i = 0; i < tk; ++i) {
        // printf("%"HIPARTI_PRI_NNZ_INDEX", ", thread_nnzs[i]);
        sum_nnzs += thread_nnzs[i];
        if(min_nnzs > thread_nnzs[i])
            min_nnzs = thread_nnzs[i];
        if(max_nnzs < thread_nnzs[i])
            max_nnzs = thread_nnzs[i];
        std_nnzs += (thread_nnzs[i] - avg_nnzs) * (thread_nnzs[i] - avg_nnzs);
    }
    // printf("\n");
    std_nnzs = sqrt(std_nnzs / tk);
    printf("min_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", max_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", avg_nnzs: %.1lf, std_nnzs: %.1lf\n", min_nnzs, max_nnzs, avg_nnzs, std_nnzs);
    ptiAssert(sum_nnzs == hitsr->nnz);

    free(thread_nnzs);
#endif

    return 0;
}


int ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Reduce_Balanced(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Reduce_Balanced(hitsr, mats, copy_mats, mats_order, mode, tk) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));
    for(int t=0; t<tk; ++t) {
        memset(copy_mats[t]->values, 0, ndims[mode]*stride*sizeof(*(copy_mats[t]->values)));
    }

    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
    ptiIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    ptiIndexVector * restrict kschr_balanced_mode = hitsr->kschr_balanced[mode];
    ptiIndexVector * restrict kschr_balanced_pos_mode = hitsr->kschr_balanced_pos[mode];
    ptiIndex npars = hitsr->nkpars[mode];

#ifdef NNZ_STATISTICS
    ptiNnzIndex * thread_nnzs = (ptiNnzIndex*)malloc(tk * sizeof(ptiNnzIndex));
    memset(thread_nnzs, 0, tk * sizeof(ptiNnzIndex));
#endif

    /* Loop parallel iterations */
#ifdef NNZ_STATISTICS
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
    for(ptiIndex p=0; p<npars; ++p) {
        int tid = omp_get_thread_num();

        /* Loop kernels */
        for(ptiIndex i=0; i<num_kernel_dim; ++i) {
            if(p >= kschr_balanced_pos_mode[i].len - 1) continue;
            ptiIndex j_begin = kschr_balanced_pos_mode[i].data[p];
            ptiIndex j_end = kschr_balanced_pos_mode[i].data[p+1];

            for(ptiIndex j=j_begin; j<j_end; ++j) {

                ptiIndex kernel_num = kschr_balanced_mode[i].data[j];
                ptiNnzIndex kptr_begin = hitsr->kptr.data[kernel_num];
                ptiNnzIndex kptr_end = hitsr->kptr.data[kernel_num+1];


                /* Allocate thread-private data */
                ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));
                ptiValueVector scratch; // Temporary array
                ptiNewValueVector(&scratch, R, R);

                /* Loop blocks in a kernel */
                for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {
                    /* Blocked matrices */
                    for(ptiIndex m=0; m<nmodes; ++m)
                        blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
                    ptiValue * blocked_mvals = copy_mats[tid]->values + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

                    ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
                    ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
                    thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

                    /* Loop entries in a block */
                    for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {

                        /* Multiply the 1st matrix */
                        ptiIndex times_mat_index = mats_order[1];
                        ptiElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                        ptiValue const entry = vals[z];
                        #pragma omp simd
                        for(ptiElementIndex r=0; r<R; ++r) {
                            scratch.data[r] = entry * blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                        }
                        /* Multiply the rest matrices */
                        for(ptiIndex m=2; m<nmodes; ++m) {
                            times_mat_index = mats_order[m];
                            tmp_i = hitsr->einds[times_mat_index].data[z];
                            #pragma omp simd
                            for(ptiElementIndex r=0; r<R; ++r) {
                                scratch.data[r] *= blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                            }
                        }

                        ptiElementIndex const mode_i = hitsr->einds[mode].data[z];
                        #pragma omp simd
                        for(ptiElementIndex r=0; r<R; ++r) {
                            blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                        }
                    }   // End loop entries
                }   // End loop blocks

                /* Free thread-private space */
                free(blocked_times_mat);
                ptiFreeValueVector(&scratch);
            }   // End kernels in a partition
        }   // End loop kernels
    }   // End loop iterations

   /* Process using atomics */
#ifdef NNZ_STATISTICS
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
    for(ptiIndex k = 0; k < hitsr->kschr_rest[mode].len; ++k) {
        int tid = omp_get_thread_num();
        ptiIndex kernel_num = hitsr->kschr_rest[mode].data[k];
        ptiNnzIndex kptr_begin = hitsr->kptr.data[kernel_num];
        ptiNnzIndex kptr_end = hitsr->kptr.data[kernel_num+1];

        /* Allocate thread-private data */
        ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));
        ptiValueVector scratch; // Temporary array
        ptiNewValueVector(&scratch, R, R);

        /* Loop blocks in a kernel */
        for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Blocked matrices */
            for(ptiIndex m=0; m<nmodes; ++m)
                blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
            /* Use copy_mats to reduce atomics */
            ptiValue * blocked_mvals = copy_mats[tid]->values + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
            thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {

                /* Multiply the 1st matrix */
                ptiIndex times_mat_index = mats_order[1];
                ptiElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                ptiValue const entry = vals[z];
                #pragma omp simd
                for(ptiElementIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                }

                /* Multiply the rest matrices */
                for(ptiIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    tmp_i = hitsr->einds[times_mat_index].data[z];
                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        scratch.data[r] *= blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                    }
                }

                ptiElementIndex const mode_i = hitsr->einds[mode].data[z];
                ptiValue * const restrict bmvals_row = blocked_mvals + mode_i * stride;
                for(ptiElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    bmvals_row[r] += scratch.data[r];
                }
            }   // End loop entries
        }   // End loop blocks

        /* Free thread-private space */
        free(blocked_times_mat);
        ptiFreeValueVector(&scratch);
    }   // End loop kernels

    /* Reduction */
    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiIndex i=0; i<ndims[mode]; ++i) {
        for(int t=0; t<tk; ++t) {
            #pragma omp simd
            for(ptiElementIndex r=0; r<R; ++r) {
                mvals[i * stride + r] += copy_mats[t]->values[i * stride + r];
            }    
        }    
    }    

#ifdef NNZ_STATISTICS
    /* Calculate load balance of kernels */
    ptiNnzIndex sum_nnzs = 0, min_nnzs = hitsr->nnz, max_nnzs = 0;
    double std_nnzs = 0.0;
    double avg_nnzs = hitsr->nnz / (double)tk;
    // printf("thread_nnzs:\n");
    for(int i = 0; i < tk; ++i) {
        // printf("%"HIPARTI_PRI_NNZ_INDEX", ", thread_nnzs[i]);
        sum_nnzs += thread_nnzs[i];
        if(min_nnzs > thread_nnzs[i])
            min_nnzs = thread_nnzs[i];
        if(max_nnzs < thread_nnzs[i])
            max_nnzs = thread_nnzs[i];
        std_nnzs += (thread_nnzs[i] - avg_nnzs) * (thread_nnzs[i] - avg_nnzs);
    }
    // printf("\n");
    std_nnzs = sqrt(std_nnzs / tk);
    printf("min_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", max_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", avg_nnzs: %.1lf, std_nnzs: %.1lf\n", min_nnzs, max_nnzs, avg_nnzs, std_nnzs);
    ptiAssert(sum_nnzs == hitsr->nnz);

    free(thread_nnzs);
#endif

    return 0;
}




int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Reduce(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk)
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));
    for(int t=0; t<tk; ++t) {
        memset(copy_mats[t]->values, 0, ndims[mode]*stride*sizeof(*(copy_mats[t]->values)));
    }

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];

    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
    ptiIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    ptiIndexVector * restrict kschr_mode = hitsr->kschr[mode];

#ifdef NNZ_STATISTICS
    ptiNnzIndex * thread_nnzs = (ptiNnzIndex*)malloc(tk * sizeof(ptiNnzIndex));
    memset(thread_nnzs, 0, tk * sizeof(ptiNnzIndex));
#endif

    /* Loop parallel iterations */
#ifdef NNZ_STATISTICS
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
    for(ptiIndex i=0; i<hitsr->nkiters[mode]; ++i) {
        int tid = omp_get_thread_num();

        /* Loop kernels */
        for(ptiIndex k=0; k<num_kernel_dim; ++k) {
            if(i >= kschr_mode[k].len) {
                // printf("i: %u, k: %u\n", i, k);
                continue;
            }
            ptiIndex kptr_loc = kschr_mode[k].data[i];
            ptiNnzIndex kptr_begin = hitsr->kptr.data[kptr_loc];
            ptiNnzIndex kptr_end = hitsr->kptr.data[kptr_loc+1];

            /* Loop blocks in a kernel */
            for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

                /* use copy_mats to store each thread's output */
                ptiValue * blocked_mvals = copy_mats[tid]->values + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
                ptiValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
                ptiValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

                ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
                ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
                thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

                /* Loop entries in a block */
                for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                    
                    ptiElementIndex mode_i = hitsr->einds[mode].data[z];
                    ptiElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                    ptiElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                    ptiValue entry = vals[z];

                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += entry *
                            blocked_times_mat_1[(ptiBlockMatrixIndex)tmp_i_1 * stride + r] *
                            blocked_times_mat_2[(ptiBlockMatrixIndex)tmp_i_2 * stride + r];
                    }
                    
                }   // End loop entries
            }   // End loop blocks

        }   // End loop kernels
    }   // End loop iterations

    /* Reduction */
    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiIndex i=0; i<ndims[mode]; ++i) {
        for(int t=0; t<tk; ++t) {
            #pragma omp simd
            for(ptiElementIndex r=0; r<R; ++r) {
                mvals[i * stride + r] += copy_mats[t]->values[i * stride + r];
            }
        }
    }

#ifdef NNZ_STATISTICS
    /* Calculate load balance of kernels */
    ptiNnzIndex sum_nnzs = 0, min_nnzs = hitsr->nnz, max_nnzs = 0;
    double std_nnzs = 0.0;
    double avg_nnzs = hitsr->nnz / (double)tk;
    // printf("thread_nnzs:\n");
    for(int i = 0; i < tk; ++i) {
        // printf("%"HIPARTI_PRI_NNZ_INDEX", ", thread_nnzs[i]);
        sum_nnzs += thread_nnzs[i];
        if(min_nnzs > thread_nnzs[i])
            min_nnzs = thread_nnzs[i];
        if(max_nnzs < thread_nnzs[i])
            max_nnzs = thread_nnzs[i];
        std_nnzs += (thread_nnzs[i] - avg_nnzs) * (thread_nnzs[i] - avg_nnzs);
    }
    // printf("\n");
    std_nnzs = sqrt(std_nnzs / tk);
    printf("min_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", max_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", avg_nnzs: %.1lf, std_nnzs: %.1lf\n", min_nnzs, max_nnzs, avg_nnzs, std_nnzs);
    ptiAssert(sum_nnzs == hitsr->nnz);

    free(thread_nnzs);
#endif

    return 0;
}


int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Reduce_Balanced(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk)
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));
    for(int t=0; t<tk; ++t) {
        memset(copy_mats[t]->values, 0, ndims[mode]*stride*sizeof(*(copy_mats[t]->values)));
    }

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];

    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
    ptiIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    ptiIndexVector * restrict kschr_balanced_mode = hitsr->kschr_balanced[mode];
    ptiIndexVector * restrict kschr_balanced_pos_mode = hitsr->kschr_balanced_pos[mode];
    ptiIndex npars = hitsr->nkpars[mode];

#ifdef NNZ_STATISTICS
    ptiNnzIndex * thread_nnzs = (ptiNnzIndex*)malloc(tk * sizeof(ptiNnzIndex));
    memset(thread_nnzs, 0, tk * sizeof(ptiNnzIndex));
#endif

    /* Loop parallel iterations */
#ifdef NNZ_STATISTICS
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
    for(ptiIndex p=0; p<npars; ++p) {
        int tid = omp_get_thread_num();

        /* Loop kernels */
        for(ptiIndex i=0; i<num_kernel_dim; ++i) {
            if(p >= kschr_balanced_pos_mode[i].len - 1) continue;
            ptiIndex j_begin = kschr_balanced_pos_mode[i].data[p];
            ptiIndex j_end = kschr_balanced_pos_mode[i].data[p+1];

            for(ptiIndex j=j_begin; j<j_end; ++j) {

                ptiIndex kernel_num = kschr_balanced_mode[i].data[j];
                ptiNnzIndex kptr_begin = hitsr->kptr.data[kernel_num];
                ptiNnzIndex kptr_end = hitsr->kptr.data[kernel_num+1];

                /* Loop blocks in a kernel */
                for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

                    /* use copy_mats to store each thread's output */
                    ptiValue * blocked_mvals = copy_mats[tid]->values + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
                    ptiValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
                    ptiValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

                    ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
                    ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
                    thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

                    /* Loop entries in a block */
                    for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                        
                        ptiElementIndex mode_i = hitsr->einds[mode].data[z];
                        ptiElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                        ptiElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                        ptiValue entry = vals[z];

                        #pragma omp simd
                        for(ptiElementIndex r=0; r<R; ++r) {
                            blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += entry *
                                blocked_times_mat_1[(ptiBlockMatrixIndex)tmp_i_1 * stride + r] *
                                blocked_times_mat_2[(ptiBlockMatrixIndex)tmp_i_2 * stride + r];
                        }
                        
                    }   // End loop entries
                }   // End loop blocks
            }   // End kernels in a partition

        }   // End loop kernels
    }   // End loop partitions


    /* Process using atomics */
#ifdef NNZ_STATISTICS
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) shared(thread_nnzs)
#else
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk) 
#endif
    for(ptiIndex k = 0; k < hitsr->kschr_rest[mode].len; ++k) {
        int tid = omp_get_thread_num();
        ptiIndex kernel_num = hitsr->kschr_rest[mode].data[k];
        ptiNnzIndex kptr_begin = hitsr->kptr.data[kernel_num];
        ptiNnzIndex kptr_end = hitsr->kptr.data[kernel_num+1];

        /* Loop blocks in a kernel */
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Use copy_mats to reduce atomics */
            ptiValue * blocked_mvals = copy_mats[tid]->values + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
#ifdef NNZ_STATISTICS
            thread_nnzs[tid] += (bptr_end - bptr_begin);
#endif

            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                 
                ptiElementIndex mode_i = hitsr->einds[mode].data[z];
                ptiElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                ptiElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                ptiValue entry = vals[z];
                ptiValue * const restrict bmvals_row = blocked_mvals + mode_i * stride;

                for(ptiElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update 
                    bmvals_row[r] += entry *  
                        blocked_times_mat_1[(ptiBlockMatrixIndex)tmp_i_1 * stride + r] *
                        blocked_times_mat_2[(ptiBlockMatrixIndex)tmp_i_2 * stride + r];
                }       
                
            }   // End loop entries
        }   // End loop blocks
        
    }   // End loop kernels

    /* Reduction */
    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiIndex i=0; i<ndims[mode]; ++i) {
        for(int t=0; t<tk; ++t) {
            #pragma omp simd
            for(ptiElementIndex r=0; r<R; ++r) {
                mvals[i * stride + r] += copy_mats[t]->values[i * stride + r];
            }
        }
    }

    /* Calculate load balance of kernels */
#ifdef NNZ_STATISTICS
    ptiNnzIndex sum_nnzs = 0, min_nnzs = hitsr->nnz, max_nnzs = 0;
    double std_nnzs = 0.0;
    double avg_nnzs = hitsr->nnz / (double)tk;
    // printf("thread_nnzs:\n");
    for(int i = 0; i < tk; ++i) {
        // printf("%"HIPARTI_PRI_NNZ_INDEX", ", thread_nnzs[i]);
        sum_nnzs += thread_nnzs[i];
        if(min_nnzs > thread_nnzs[i])
            min_nnzs = thread_nnzs[i];
        if(max_nnzs < thread_nnzs[i])
            max_nnzs = thread_nnzs[i];
        std_nnzs += (thread_nnzs[i] - avg_nnzs) * (thread_nnzs[i] - avg_nnzs);
    }
    // printf("\n");
    std_nnzs = sqrt(std_nnzs / tk);
    printf("min_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", max_nnzs: %"HIPARTI_PRI_NNZ_INDEX ", avg_nnzs: %.1lf, std_nnzs: %.1lf\n", min_nnzs, max_nnzs, avg_nnzs, std_nnzs);
    ptiAssert(sum_nnzs == hitsr->nnz);

    free(thread_nnzs);
#endif

    return 0;
}


int ptiOmpMTTKRPHiCOOKernels_MatrixTiling_Scheduled_Reduce_Two(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Reduce_Two(hitsr, mats, copy_mats, mats_order, mode, tk) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));
    for(int t=0; t<tk; ++t) {
        memset(copy_mats[t]->values, 0, ndims[mode]*stride*sizeof(*(copy_mats[t]->values)));
    }

    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
    ptiIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    ptiIndexVector * restrict kschr_mode = hitsr->kschr[mode];

    int tk2 = 2;
    /* Loop parallel iterations */
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk/tk2)
    for(ptiIndex i=0; i<hitsr->nkiters[mode]; ++i) {
        int tid = omp_get_thread_num();

        /* Loop kernels */
        // TODO: cannot compile using icc
        // #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk2)
        for(ptiIndex k=0; k<num_kernel_dim; ++k) {

            if(i >= kschr_mode[k].len) continue;
            ptiIndex kptr_loc = kschr_mode[k].data[i];
            ptiNnzIndex kptr_begin = hitsr->kptr.data[kptr_loc];
            ptiNnzIndex kptr_end = hitsr->kptr.data[kptr_loc+1];

            /* Allocate thread-private data */
            ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));
            ptiValueVector scratch; // Temporary array
            ptiNewValueVector(&scratch, R, R);

            /* Loop blocks in a kernel */
            for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {
                /* Blocked matrices */
                for(ptiIndex m=0; m<nmodes; ++m)
                    blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
                ptiValue * blocked_mvals = copy_mats[tid]->values + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

                ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
                ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
                /* Loop entries in a block */
                for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {

                    /* Multiply the 1st matrix */
                    ptiIndex times_mat_index = mats_order[1];
                    ptiElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                    ptiValue const entry = vals[z];
                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        scratch.data[r] = entry * blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                    }
                    /* Multiply the rest matrices */
                    for(ptiIndex m=2; m<nmodes; ++m) {
                        times_mat_index = mats_order[m];
                        tmp_i = hitsr->einds[times_mat_index].data[z];
                        #pragma omp simd
                        for(ptiElementIndex r=0; r<R; ++r) {
                            scratch.data[r] *= blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                        }
                    }

                    ptiElementIndex const mode_i = hitsr->einds[mode].data[z];
                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                    }
                }   // End loop entries
            }   // End loop blocks

            /* Free thread-private space */
            free(blocked_times_mat);
            ptiFreeValueVector(&scratch);
        }   // End loop kernels
    }   // End loop iterations

    /* Reduction */
    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiIndex i=0; i<ndims[mode]; ++i) {
        for(int t=0; t<tk/tk2; ++t) {
            #pragma omp simd
            for(ptiElementIndex r=0; r<R; ++r) {
                mvals[i * stride + r] += copy_mats[t]->values[i * stride + r];
            }    
        }    
    }    

    return 0;
}


int ptiOmpMTTKRPHiCOOKernels_3D_MatrixTiling_Scheduled_Reduce_Two(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiRankMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk)
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));
    for(int t=0; t<tk; ++t) {
        memset(copy_mats[t]->values, 0, ndims[mode]*stride*sizeof(*(copy_mats[t]->values)));
    }

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];

    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
    ptiIndex num_kernel_dim = (ndims[mode] + sk - 1) / sk;
    ptiIndexVector * restrict kschr_mode = hitsr->kschr[mode];

    int tk2 = 2;
    /* Loop parallel iterations */
    #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk/tk2)
    for(ptiIndex i=0; i<hitsr->nkiters[mode]; ++i) {
        int tid = omp_get_thread_num();

        /* Loop kernels */
        // Cannot compile using icc
        // #pragma omp parallel for schedule(dynamic, CHUNKSIZE) num_threads(tk2)
        for(ptiIndex k=0; k<num_kernel_dim; ++k) {
            if(i >= kschr_mode[k].len) {
                // printf("i: %u, k: %u\n", i, k);
                continue;
            }
            ptiIndex kptr_loc = kschr_mode[k].data[i];
            ptiNnzIndex kptr_begin = hitsr->kptr.data[kptr_loc];
            ptiNnzIndex kptr_end = hitsr->kptr.data[kptr_loc+1];

            /* Loop blocks in a kernel */
            for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

                /* use copy_mats to store each thread's output */
                ptiValue * blocked_mvals = copy_mats[tid]->values + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
                ptiValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
                ptiValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

                ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
                ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
                /* Loop entries in a block */
                for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                    
                    ptiElementIndex mode_i = hitsr->einds[mode].data[z];
                    ptiElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                    ptiElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                    ptiValue entry = vals[z];

                    #pragma omp simd
                    for(ptiElementIndex r=0; r<R; ++r) {
                        blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += entry *
                            blocked_times_mat_1[(ptiBlockMatrixIndex)tmp_i_1 * stride + r] *
                            blocked_times_mat_2[(ptiBlockMatrixIndex)tmp_i_2 * stride + r];
                    }
                    
                }   // End loop entries
            }   // End loop blocks

        }   // End loop kernels
    }   // End loop iterations

    /* Reduction */
    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiIndex i=0; i<ndims[mode]; ++i) {
        for(int t=0; t<tk/tk2; ++t) {
            #pragma omp simd
            for(ptiElementIndex r=0; r<R; ++r) {
                mvals[i * stride + r] += copy_mats[t]->values[i * stride + r];
            }
        }
    }

    return 0;
}



int ptiOmpMTTKRPHiCOOBlocks(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tb) 
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOBlocks_3D(hitsr, mats, mats_order, mode, tb) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const vals = hitsr->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiMatrix * const M = mats[nmodes];
    ptiValue * const mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    // omp_lock_t lock;
    // omp_init_lock(&lock);

    /* Loop kernels */
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Allocate thread-private data */
            ptiIndex * block_coord = (ptiIndex*)malloc(nmodes * sizeof(*block_coord));
            ptiIndex * ele_coord = (ptiIndex*)malloc(nmodes * sizeof(*ele_coord));
            ptiValueVector scratch; // Temporary array
            ptiNewValueVector(&scratch, R, R);

            /* Block indices */
            for(ptiIndex m=0; m<nmodes; ++m)
                block_coord[m] = hitsr->binds[m].data[b];

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                /* Element indices */
                for(ptiIndex m=0; m<nmodes; ++m)
                    ele_coord[m] = (block_coord[m] << hitsr->sb_bits) + hitsr->einds[m].data[z];

                /* Multiply the 1st matrix */
                ptiIndex times_mat_index = mats_order[1];
                ptiMatrix * times_mat = mats[times_mat_index];
                ptiIndex tmp_i = ele_coord[times_mat_index];
                ptiValue const entry = vals[z];
                for(ptiIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(ptiIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    times_mat = mats[times_mat_index];
                    tmp_i = ele_coord[times_mat_index];
                    for(ptiIndex r=0; r<R; ++r) {
                        scratch.data[r] *= times_mat->values[tmp_i * stride + r];
                    }
                }

                ptiIndex const mode_i = ele_coord[mode];
                // omp_set_lock(&lock);
                for(ptiIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += scratch.data[r];
                }
                // omp_unset_lock(&lock);
            }   // End loop entries

            /* Free thread-private space */
            free(block_coord);
            free(ele_coord);
            ptiFreeValueVector(&scratch);
        }   // End loop blocks

    }   // End loop kernels

    // omp_destroy_lock(&lock);

    return 0;
}


int ptiOmpMTTKRPHiCOOBlocks_3D(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tb) 
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiMatrix * restrict times_mat_2 = mats[times_mat_index_2];


    /* Loop kernels */
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            ptiBlockIndex block_coord_mode = hitsr->binds[mode].data[b];
            ptiBlockIndex block_coord_1 = hitsr->binds[times_mat_index_1].data[b];
            ptiBlockIndex block_coord_2 = hitsr->binds[times_mat_index_2].data[b];

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                
                ptiIndex mode_i = (block_coord_mode << hitsr->sb_bits) + hitsr->einds[mode].data[z];
                ptiIndex tmp_i_1 = (block_coord_1 << hitsr->sb_bits) + hitsr->einds[times_mat_index_1].data[z];
                ptiIndex tmp_i_2 = (block_coord_2 << hitsr->sb_bits) + hitsr->einds[times_mat_index_2].data[z];
                ptiValue entry = vals[z];
                for(ptiIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks
    }   // End loop kernels

    return 0;
}



int ptiOmpMTTKRPHiCOOBlocks_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tb) 
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOBlocks_3D_MatrixTiling(hitsr, mats, mats_order, mode, tb) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    /* Loop kernels */
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Allocate thread-private data */
            ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));
            ptiValueVector scratch; // Temporary array
            ptiNewValueVector(&scratch, R, R);

            /* Blocked matrices */
            for(ptiIndex m=0; m<nmodes; ++m)
                blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {

                /* Multiply the 1st matrix */
                ptiIndex times_mat_index = mats_order[1];
                ptiElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                ptiValue const entry = vals[z];
                for(ptiElementIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(ptiIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    tmp_i = hitsr->einds[times_mat_index].data[z];
                    for(ptiElementIndex r=0; r<R; ++r) {
                        scratch.data[r] *= blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                    }
                }

                ptiElementIndex const mode_i = hitsr->einds[mode].data[z];
                for(ptiElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                }
            }   // End loop entries

            /* Free thread-private space */
            free(blocked_times_mat);
            ptiFreeValueVector(&scratch);
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}


int ptiOmpMTTKRPHiCOOBlocks_3D_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tb) 
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];


    /* Loop kernels */
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Allocate thread-private data */
            ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                
                ptiElementIndex mode_i = hitsr->einds[mode].data[z];
                ptiElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                ptiElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                ptiValue entry = vals[z];

                for(ptiElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += entry *
                        blocked_times_mat_1[(ptiBlockMatrixIndex)tmp_i_1 * stride + r] *
                        blocked_times_mat_2[(ptiBlockMatrixIndex)tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}




int ptiOmpMTTKRPHiCOOKernelsBlocks(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb) 
{
    omp_set_nested(1);
    omp_set_dynamic(0);

    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernelsBlocks_3D(hitsr, mats, mats_order, mode, tk, tb) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const vals = hitsr->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiMatrix * const M = mats[nmodes];
    ptiValue * const mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    // omp_lock_t lock;
    // omp_init_lock(&lock);

    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Allocate thread-private data */
            ptiIndex * block_coord = (ptiIndex*)malloc(nmodes * sizeof(*block_coord));
            ptiIndex * ele_coord = (ptiIndex*)malloc(nmodes * sizeof(*ele_coord));
            ptiValueVector scratch; // Temporary array
            ptiNewValueVector(&scratch, R, R);

            /* Block indices */
            for(ptiIndex m=0; m<nmodes; ++m)
                block_coord[m] = hitsr->binds[m].data[b];

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                /* Element indices */
                for(ptiIndex m=0; m<nmodes; ++m)
                    ele_coord[m] = (block_coord[m] << hitsr->sb_bits) + hitsr->einds[m].data[z];

                /* Multiply the 1st matrix */
                ptiIndex times_mat_index = mats_order[1];
                ptiMatrix * times_mat = mats[times_mat_index];
                ptiIndex tmp_i = ele_coord[times_mat_index];
                ptiValue const entry = vals[z];
                for(ptiIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(ptiIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    times_mat = mats[times_mat_index];
                    tmp_i = ele_coord[times_mat_index];
                    for(ptiIndex r=0; r<R; ++r) {
                        scratch.data[r] *= times_mat->values[tmp_i * stride + r];
                    }
                }

                ptiIndex const mode_i = ele_coord[mode];
                // omp_set_lock(&lock);
                for(ptiIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += scratch.data[r];
                }
                // omp_unset_lock(&lock);
            }   // End loop entries

            /* Free thread-private space */
            free(block_coord);
            free(ele_coord);
            ptiFreeValueVector(&scratch);
        }   // End loop blocks

    }   // End loop kernels

    // omp_destroy_lock(&lock);

    return 0;
}


int ptiOmpMTTKRPHiCOOKernelsBlocks_3D(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb) 
{
    omp_set_nested(1);
    omp_set_dynamic(0);

    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiMatrix * restrict times_mat_2 = mats[times_mat_index_2];


    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            ptiBlockIndex block_coord_mode = hitsr->binds[mode].data[b];
            ptiBlockIndex block_coord_1 = hitsr->binds[times_mat_index_1].data[b];
            ptiBlockIndex block_coord_2 = hitsr->binds[times_mat_index_2].data[b];

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                
                ptiIndex mode_i = (block_coord_mode << hitsr->sb_bits) + hitsr->einds[mode].data[z];
                ptiIndex tmp_i_1 = (block_coord_1 << hitsr->sb_bits) + hitsr->einds[times_mat_index_1].data[z];
                ptiIndex tmp_i_2 = (block_coord_2 << hitsr->sb_bits) + hitsr->einds[times_mat_index_2].data[z];
                ptiValue entry = vals[z];
                for(ptiIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks
    }   // End loop kernels

    return 0;
}



int ptiOmpMTTKRPHiCOOKernelsBlocks_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb) 
{
    ptiIndex const nmodes = hitsr->nmodes;
    omp_set_nested(1);
    omp_set_dynamic(0);

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRPHiCOOKernelsBlocks_3D_MatrixTiling(hitsr, mats, mats_order, mode, tk, tb) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {

        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(ptiNnzIndex b=kptr_begin; b<kptr_end; ++b) {

            /* Allocate thread-private data */
            ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));
            ptiValueVector scratch; // Temporary array
            ptiNewValueVector(&scratch, R, R);

            /* Blocked matrices */
            for(ptiIndex m=0; m<nmodes; ++m)
                blocked_times_mat[m] = mats[m]->values + (hitsr->binds[m].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {

                /* Multiply the 1st matrix */
                ptiIndex times_mat_index = mats_order[1];
                ptiElementIndex tmp_i = hitsr->einds[times_mat_index].data[z];
                ptiValue const entry = vals[z];
                for(ptiElementIndex r=0; r<R; ++r) {
                    scratch.data[r] = entry * blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                }
                /* Multiply the rest matrices */
                for(ptiIndex m=2; m<nmodes; ++m) {
                    times_mat_index = mats_order[m];
                    tmp_i = hitsr->einds[times_mat_index].data[z];
                    for(ptiElementIndex r=0; r<R; ++r) {
                        scratch.data[r] *= blocked_times_mat[times_mat_index][(ptiBlockMatrixIndex)tmp_i * stride + r];
                    }
                }

                ptiElementIndex const mode_i = hitsr->einds[mode].data[z];
                for(ptiElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                }
            }   // End loop entries

            /* Free thread-private space */
            free(blocked_times_mat);
            ptiFreeValueVector(&scratch);
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}


int ptiOmpMTTKRPHiCOOKernelsBlocks_3D_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    const int tb) 
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  HiCOO SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiElementIndex const R = mats[mode]->ncols;
    ptiRankMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(*mvals));

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiRankMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiRankMatrix * restrict times_mat_2 = mats[times_mat_index_2];


    /* Loop kernels */
    #pragma omp parallel for num_threads(tk)
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        #pragma omp parallel for num_threads(tb)
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {
            
            /* Allocate thread-private data */
            ptiValue * blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            ptiValue * blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                
                ptiElementIndex mode_i = hitsr->einds[mode].data[z];
                ptiElementIndex tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                ptiElementIndex tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                ptiValue entry = vals[z];

                for(ptiElementIndex r=0; r<R; ++r) {
                    #pragma omp atomic update
                    blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += entry *
                        blocked_times_mat_1[(ptiBlockMatrixIndex)tmp_i_1 * stride + r] *
                        blocked_times_mat_2[(ptiBlockMatrixIndex)tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}


