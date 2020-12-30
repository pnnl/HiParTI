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

int ptiMTTKRPHiCOO_3D(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode);
int ptiMTTKRPHiCOO_3D_Blocked(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode);
int ptiMTTKRPHiCOO_3D_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode);
int ptiMTTKRPHiCOO_4D_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode);
int ptiMTTKRPHiCOO_3D_MatrixTiling_init(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode);

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
int ptiMTTKRPHiCOO(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode)
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiMTTKRPHiCOO_3D_Blocked(hitsr, mats, mats_order, mode) == 0);
        return 0;
    }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiIndex const stride = mats[0]->stride;
    ptiValueVector scratch;  // Temporary array

    /* Check the mats. */
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
    ptiNewValueVector(&scratch, R, R);

    ptiIndex * block_coord = (ptiIndex*)malloc(nmodes * sizeof(*block_coord));
    ptiIndex * ele_coord = (ptiIndex*)malloc(nmodes * sizeof(*ele_coord));


    /* Loop kernels */
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {
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
                for(ptiIndex r=0; r<R; ++r) {
                    mvals[mode_i * stride + r] += scratch.data[r];
                }
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels


    free(block_coord);
    free(ele_coord);
    ptiFreeValueVector(&scratch);

    return 0;
}


/* Very slow version! Slower than COO in Morton order. */
int ptiMTTKRPHiCOO_3D(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode)
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

    /* block_coord is reused, no need to store ele_coord for 3D tensors */
    ptiBlockIndex * block_coord = (ptiBlockIndex*)malloc(nmodes * sizeof(*block_coord));

    ptiIndex mode_i;
    ptiIndex tmp_i_1, tmp_i_2;
    ptiValue entry;

    /* Loop kernels */
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Block indices */
            for(ptiIndex m=0; m<nmodes; ++m)
                block_coord[m] = hitsr->binds[m].data[b];

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                
                mode_i = (block_coord[mode] << hitsr->sb_bits) + hitsr->einds[mode].data[z];
                tmp_i_1 = (block_coord[times_mat_index_1] << hitsr->sb_bits) + hitsr->einds[times_mat_index_1].data[z];
                tmp_i_2 = (block_coord[times_mat_index_2] << hitsr->sb_bits) + hitsr->einds[times_mat_index_2].data[z];
                entry = vals[z];
                for(ptiIndex r=0; r<R; ++r) {
                    mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels


    free(block_coord);

    return 0;
}


int ptiMTTKRPHiCOO_3D_Blocked(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode)
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

    ptiElementIndex mode_i;
    ptiElementIndex tmp_i_1, tmp_i_2;
    ptiValue entry;
    ptiValue * restrict blocked_mvals;
    ptiValue * restrict blocked_times_mat_1;
    ptiValue * restrict blocked_times_mat_2;

    /* Loop kernels */
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                
                mode_i = hitsr->einds[mode].data[z];
                tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                entry = vals[z];

                ptiValue * const restrict bmvals_row = blocked_mvals + mode_i * stride;
                ptiValue * const restrict blocked_times_mat_1_row = blocked_times_mat_1 + tmp_i_1 * stride;
                ptiValue * const restrict blocked_times_mat_2_row = blocked_times_mat_2 + tmp_i_2 * stride;

                for(ptiIndex r=0; r<R; ++r) {
                    bmvals_row[r] += entry *
                        blocked_times_mat_1_row[r]
                        * blocked_times_mat_2_row[r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}


int ptiMTTKRPHiCOO_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode)
{
    ptiIndex const nmodes = hitsr->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiMTTKRPHiCOO_3D_MatrixTiling(hitsr, mats, mats_order, mode) == 0);
        return 0;
    } 
    // else if(nmodes == 4) {
    //     ptiAssert(ptiMTTKRPHiCOO_4D_MatrixTiling(hitsr, mats, mats_order, mode) == 0);
    //     return 0;
    // }

    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;
    ptiValueVector scratch;  // Temporary array

    /* Check the mats. */
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
    ptiNewValueVector(&scratch, R, R);

    ptiValue ** blocked_times_mat = (ptiValue**)malloc(nmodes * sizeof(*blocked_times_mat));

    /* Loop kernels */
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {
            /* Block indices */
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
                #pragma omp simd
                for(ptiElementIndex r=0; r<R; ++r) {
                    blocked_mvals[(ptiBlockMatrixIndex)mode_i * stride + r] += scratch.data[r];
                }
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels


    free(blocked_times_mat);
    ptiFreeValueVector(&scratch);

    return 0;
}

int ptiMTTKRPHiCOO_3D_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode)
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

    ptiElementIndex mode_i;
    ptiElementIndex tmp_i_1, tmp_i_2;
    ptiValue entry;
    ptiValue * restrict blocked_mvals;
    ptiValue * restrict blocked_times_mat_1;
    ptiValue * restrict blocked_times_mat_2;

    /* Loop kernels */
    // ptiTimer loop_timer, kernel_timer, block_timer, element_timer, elementmat_timer, blockmat_timer;
    // double loop_etime = 0, kernel_etime = 0, block_etime = 0, element_etime = 0, elementmat_etime = 0, blockmat_etime = 0;
    // ptiNewTimer(&loop_timer, 0);
    // ptiNewTimer(&kernel_timer, 0);
    // ptiNewTimer(&block_timer, 0);
    // ptiNewTimer(&element_timer, 0);
    // ptiNewTimer(&elementmat_timer, 0);
    // ptiNewTimer(&blockmat_timer, 0);

    // ptiStartTimer(loop_timer);

    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        // printf("kptr_begin: %"HIPARTI_PRI_NNZ_INDEX", kptr_end: %"HIPARTI_PRI_NNZ_INDEX"\n", kptr_begin, kptr_end); 
        // ptiStartTimer(kernel_timer);
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            // ptiStartTimer(blockmat_timer);
            blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            // ptiStopTimer(blockmat_timer);
            // blockmat_etime += ptiElapsedTime(blockmat_timer);
            // ptiPrintElapsedTime(blockmat_timer, "===Blockmat Timer");

            /* Loop entries in a block */
            // printf("bptr_begin: %"HIPARTI_PRI_INDEX", bptr_end: %"HIPARTI_PRI_INDEX"\n", bptr_begin, bptr_end); 
            // ptiStartTimer(block_timer);
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                // ptiStartTimer(elementmat_timer);
                mode_i = hitsr->einds[mode].data[z];
                tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                // mode_i = (ptiBlockMatrixIndex)hitsr->einds[mode].data[z];
                // tmp_i_1 = (ptiBlockMatrixIndex)hitsr->einds[times_mat_index_1].data[z];
                // tmp_i_2 = (ptiBlockMatrixIndex)hitsr->einds[times_mat_index_2].data[z];
                entry = vals[z];
                ptiValue * const restrict bmvals_row = blocked_mvals + mode_i * stride;
                ptiValue * const restrict blocked_times_mat_1_row = blocked_times_mat_1 + tmp_i_1 * stride;
                ptiValue * const restrict blocked_times_mat_2_row = blocked_times_mat_2 + tmp_i_2 * stride;
                // ptiStopTimer(elementmat_timer);
                // elementmat_etime += ptiElapsedTime(elementmat_timer);
                // ptiPrintElapsedTime(elementmat_timer, "===Elementmat Timer");

                // ptiStartTimer(element_timer);
                #pragma omp simd
                for(ptiElementIndex r=0; r<R; ++r) {
                    // blocked_mvals[mode_i * stride + r] += entry * 
                    //     blocked_times_mat_1[tmp_i_1 * stride + r] * 
                    //     blocked_times_mat_2[tmp_i_2 * stride + r];
                    bmvals_row[r] += entry * 
                        blocked_times_mat_1_row[r]
                        * blocked_times_mat_2_row[r];
                }
                // ptiStopTimer(element_timer);
                // element_etime += ptiElapsedTime(element_timer);
                // ptiPrintElapsedTime(element_timer, "===Element Timer");
                
            }   // End loop entries
            // ptiStopTimer(block_timer);
            // block_etime += ptiElapsedTime(block_timer);
            // ptiPrintElapsedTime(block_timer, "==Block Timer");

        }   // End loop blocks
        // ptiStopTimer(kernel_timer);
        // kernel_etime += ptiElapsedTime(kernel_timer);
        // ptiPrintElapsedTime(kernel_timer, "=Kernel Timer");

    }   // End loop kernels

    // ptiStopTimer(loop_timer);
    // loop_etime += ptiElapsedTime(loop_timer);
    // ptiPrintElapsedTime(loop_timer, "=Loop Timer");

    // printf("\nTotal Elementmat Time: %lf\n", elementmat_etime);
    // printf("Total Element Time: %lf\n", element_etime);
    // printf("Total Blockmat Time: %lf\n", blockmat_etime);
    // printf("Total Block Time: %lf\n", block_etime);
    // printf("Total Kernel Time: %lf\n", kernel_etime);
    // printf("Total Loop Time: %lf\n\n", loop_etime);

    // ptiFreeTimer(loop_timer);
    // ptiFreeTimer(kernel_timer);
    // ptiFreeTimer(block_timer);
    // ptiFreeTimer(element_timer);
    // ptiFreeTimer(elementmat_timer);
    // ptiFreeTimer(blockmat_timer);

    return 0;
}



int ptiMTTKRPHiCOO_4D_MatrixTiling(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiRankMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode)
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiValue const * const restrict vals = hitsr->values.data;
    ptiElementIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes == 4);
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
    ptiIndex times_mat_index_3 = mats_order[3];
    ptiRankMatrix * restrict times_mat_3 = mats[times_mat_index_3];

    ptiElementIndex mode_i;
    ptiElementIndex tmp_i_1, tmp_i_2, tmp_i_3;
    ptiValue entry;
    ptiValue * restrict blocked_mvals;
    ptiValue * restrict blocked_times_mat_1;
    ptiValue * restrict blocked_times_mat_2;
    ptiValue * restrict blocked_times_mat_3;

    /* Loop kernels */
    // ptiTimer loop_timer, kernel_timer, block_timer, element_timer, elementmat_timer, blockmat_timer;
    // double loop_etime = 0, kernel_etime = 0, block_etime = 0, element_etime = 0, elementmat_etime = 0, blockmat_etime = 0;
    // ptiNewTimer(&loop_timer, 0);
    // ptiNewTimer(&kernel_timer, 0);
    // ptiNewTimer(&block_timer, 0);
    // ptiNewTimer(&element_timer, 0);
    // ptiNewTimer(&elementmat_timer, 0);
    // ptiNewTimer(&blockmat_timer, 0);

    // ptiStartTimer(loop_timer);

    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        // printf("kptr_begin: %"HIPARTI_PRI_NNZ_INDEX", kptr_end: %"HIPARTI_PRI_NNZ_INDEX"\n", kptr_begin, kptr_end); 
        // ptiStartTimer(kernel_timer);
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            // ptiStartTimer(blockmat_timer);
            blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_3 = times_mat_3->values + (hitsr->binds[times_mat_index_3].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            // ptiStopTimer(blockmat_timer);
            // blockmat_etime += ptiElapsedTime(blockmat_timer);
            // ptiPrintElapsedTime(blockmat_timer, "===Blockmat Timer");

            /* Loop entries in a block */
            // printf("bptr_begin: %"HIPARTI_PRI_INDEX", bptr_end: %"HIPARTI_PRI_INDEX"\n", bptr_begin, bptr_end); 
            // ptiStartTimer(block_timer);
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                // ptiStartTimer(elementmat_timer);
                mode_i = hitsr->einds[mode].data[z];
                tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                tmp_i_3 = hitsr->einds[times_mat_index_3].data[z];
                // mode_i = (ptiBlockMatrixIndex)hitsr->einds[mode].data[z];
                // tmp_i_1 = (ptiBlockMatrixIndex)hitsr->einds[times_mat_index_1].data[z];
                // tmp_i_2 = (ptiBlockMatrixIndex)hitsr->einds[times_mat_index_2].data[z];
                // tmp_i_3 = (ptiBlockMatrixIndex)hitsr->einds[times_mat_index_3].data[z];
                entry = vals[z];
                ptiValue * const restrict bmvals_row = blocked_mvals + mode_i * stride;
                ptiValue * const restrict blocked_times_mat_1_row = blocked_times_mat_1 + tmp_i_1 * stride;
                ptiValue * const restrict blocked_times_mat_2_row = blocked_times_mat_2 + tmp_i_2 * stride;
                ptiValue * const restrict blocked_times_mat_3_row = blocked_times_mat_3 + tmp_i_3 * stride;
                // ptiStopTimer(elementmat_timer);
                // elementmat_etime += ptiElapsedTime(elementmat_timer);
                // ptiPrintElapsedTime(elementmat_timer, "===Elementmat Timer");

                // ptiStartTimer(element_timer);
                #pragma omp simd
                for(ptiElementIndex r=0; r<R; ++r) {
                    // blocked_mvals[mode_i * stride + r] += entry * 
                    //     blocked_times_mat_1[tmp_i_1 * stride + r] * 
                    //     blocked_times_mat_2[tmp_i_2 * stride + r] * 
                    //     blocked_times_mat_3[tmp_i_3 * stride + r];
                    bmvals_row[r] += entry * 
                        blocked_times_mat_1_row[r]
                        * blocked_times_mat_2_row[r]
                        * blocked_times_mat_3_row[r];
                }
                // ptiStopTimer(element_timer);
                // element_etime += ptiElapsedTime(element_timer);
                // ptiPrintElapsedTime(element_timer, "===Element Timer");
                
            }   // End loop entries
            // ptiStopTimer(block_timer);
            // block_etime += ptiElapsedTime(block_timer);
            // ptiPrintElapsedTime(block_timer, "==Block Timer");

        }   // End loop blocks
        // ptiStopTimer(kernel_timer);
        // kernel_etime += ptiElapsedTime(kernel_timer);
        // ptiPrintElapsedTime(kernel_timer, "=Kernel Timer");

    }   // End loop kernels

    // ptiStopTimer(loop_timer);
    // loop_etime += ptiElapsedTime(loop_timer);
    // ptiPrintElapsedTime(loop_timer, "=Loop Timer");

    // printf("\nTotal Elementmat Time: %lf\n", elementmat_etime);
    // printf("Total Element Time: %lf\n", element_etime);
    // printf("Total Blockmat Time: %lf\n", blockmat_etime);
    // printf("Total Block Time: %lf\n", block_etime);
    // printf("Total Kernel Time: %lf\n", kernel_etime);
    // printf("Total Loop Time: %lf\n\n", loop_etime);

    // ptiFreeTimer(loop_timer);
    // ptiFreeTimer(kernel_timer);
    // ptiFreeTimer(block_timer);
    // ptiFreeTimer(element_timer);
    // ptiFreeTimer(elementmat_timer);
    // ptiFreeTimer(blockmat_timer);

    return 0;
}



int ptiMTTKRPHiCOO_3D_MatrixTiling_init(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode)
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

    ptiElementIndex mode_i;
    ptiElementIndex tmp_i_1, tmp_i_2;
    ptiValue entry;
    ptiValue * blocked_mvals;
    ptiValue * blocked_times_mat_1;
    ptiValue * blocked_times_mat_2;

    /* Loop kernels */
    for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
        ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];

        /* Loop blocks in a kernel */
        for(ptiIndex b=kptr_begin; b<kptr_end; ++b) {

            blocked_mvals = mvals + (hitsr->binds[mode].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_1 = times_mat_1->values + (hitsr->binds[times_mat_index_1].data[b] << hitsr->sb_bits) * stride;
            blocked_times_mat_2 = times_mat_2->values + (hitsr->binds[times_mat_index_2].data[b] << hitsr->sb_bits) * stride;

            ptiNnzIndex bptr_begin = hitsr->bptr.data[b];
            ptiNnzIndex bptr_end = hitsr->bptr.data[b+1];
            /* Loop entries in a block */
            for(ptiIndex z=bptr_begin; z<bptr_end; ++z) {
                
                mode_i = hitsr->einds[mode].data[z];
                tmp_i_1 = hitsr->einds[times_mat_index_1].data[z];
                tmp_i_2 = hitsr->einds[times_mat_index_2].data[z];
                entry = vals[z];

                for(ptiIndex r=0; r<R; ++r) {
                    blocked_mvals[mode_i * stride + r] += entry * 
                        blocked_times_mat_1[tmp_i_1 * stride + r] * 
                        blocked_times_mat_2[tmp_i_2 * stride + r];
                }
                
            }   // End loop entries
        }   // End loop blocks

    }   // End loop kernels

    return 0;
}
