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
#include "sptensor.h"

int ptiOmpMTTKRP_3D(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRP_3D_Reduce(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk);
int ptiOmpMTTKRP_3D_Lock(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    ptiMutexPool * lock_pool);

/**
 * OpenMP parallelized Matriced sparse tensor times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  X    the sparse tensor input X
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products
 * @param[in]  mode   the mode on which the MTTKRP is performed
 * @param[in]  scratch an temporary array to store intermediate results, space assigned before this function
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 * In this version, a large scratch is used to maximize parallelism. (To be optimized)
 */
int ptiOmpMTTKRP_Init(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode)
{

    ptiIndex const nmodes = X->nmodes;

    ptiNnzIndex const nnz = X->nnz;
    ptiIndex const * const ndims = X->ndims;
    ptiValue const * const vals = X->values.data;
    ptiIndex const stride = mats[0]->stride;
    ptiValueVector scratch;  // Temporary array
    ptiNewValueVector(&scratch, nnz * stride, nnz * stride);
    ptiConstantValueVector(&scratch, 0);

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiIndex const * const mode_ind = X->inds[mode].data;
    ptiMatrix * const M = mats[nmodes];
    ptiValue * const mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(ptiValue));

    #pragma omp parallel for
    for(ptiNnzIndex x=0; x<nnz; ++x) {

        ptiIndex times_mat_index = mats_order[1];
        ptiMatrix * times_mat = mats[times_mat_index];
        ptiIndex * times_inds = X->inds[times_mat_index].data;
        ptiIndex tmp_i = times_inds[x];
        ptiValue const entry = vals[x];
        for(ptiIndex r=0; r<R; ++r) {
            scratch.data[x * stride + r] = entry * times_mat->values[tmp_i * stride + r];
        }

        for(ptiIndex i=2; i<nmodes; ++i) {
            times_mat_index = mats_order[i];
            times_mat = mats[times_mat_index];
            times_inds = X->inds[times_mat_index].data;
            tmp_i = times_inds[x];

            for(ptiIndex r=0; r<R; ++r) {
                scratch.data[x * stride + r] *= times_mat->values[tmp_i * stride + r];
            }
        }

    }

    for(ptiNnzIndex x=0; x<nnz; ++x) {
        ptiIndex const mode_i = mode_ind[x];
        for(ptiIndex r=0; r<R; ++r) {
            mvals[mode_i * stride + r] += scratch.data[x * stride + r];
        }
    }

    ptiFreeValueVector(&scratch);

    return 0;
}


int ptiOmpMTTKRP(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk)
{
    ptiIndex const nmodes = X->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRP_3D(X, mats, mats_order, mode, tk) == 0);
        return 0;
    }

    ptiNnzIndex const nnz = X->nnz;
    ptiIndex const * const ndims = X->ndims;
    ptiValue const * const vals = X->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiIndex const * const mode_ind = X->inds[mode].data;
    ptiValue * const restrict mvals = mats[nmodes]->values;
    memset(mvals, 0, tmpI*stride*sizeof(ptiValue));

    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiNnzIndex x=0; x<nnz; ++x) {
        ptiValueVector scratch;  // Temporary array
        ptiNewValueVector(&scratch, R, R);
        ptiConstantValueVector(&scratch, 0);

        ptiIndex times_mat_index = mats_order[1];
        ptiMatrix * times_mat = mats[times_mat_index];
        ptiIndex * times_inds = X->inds[times_mat_index].data;
        ptiIndex tmp_i = times_inds[x];
        ptiValue const entry = vals[x];
        #pragma omp simd
        for(ptiIndex r=0; r<R; ++r) {
            scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
        }

        for(ptiIndex i=2; i<nmodes; ++i) {
            times_mat_index = mats_order[i];
            times_mat = mats[times_mat_index];
            times_inds = X->inds[times_mat_index].data;
            tmp_i = times_inds[x];

            #pragma omp simd
            for(ptiIndex r=0; r<R; ++r) {
                scratch.data[r] *= times_mat->values[tmp_i * stride + r];
            }
        }

        ptiIndex const mode_i = mode_ind[x];
        ptiValue * const restrict mvals_row = mvals + mode_i * stride;
        for(ptiIndex r=0; r<R; ++r) {
            #pragma omp atomic update
            mvals_row[r] += scratch.data[r];
        }

        ptiFreeValueVector(&scratch);
    }   // End loop nnzs

    return 0;
}


int ptiOmpMTTKRP_3D(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = X->nmodes;
    ptiNnzIndex const nnz = X->nnz;
    ptiIndex const * const ndims = X->ndims;
    ptiValue const * const restrict vals = X->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiIndex const * const restrict mode_ind = X->inds[mode].data;
    ptiValue * const restrict mvals = mats[nmodes]->values;
    memset(mvals, 0, tmpI*stride*sizeof(ptiValue));

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    ptiIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;

    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiNnzIndex x=0; x<nnz; ++x) {
        ptiIndex mode_i = mode_ind[x];
        ptiValue * const restrict mvals_row = mvals + mode_i * stride;
        ptiIndex tmp_i_1 = times_inds_1[x];
        ptiIndex tmp_i_2 = times_inds_2[x];
        ptiValue entry = vals[x];

        for(ptiIndex r=0; r<R; ++r) {
            #pragma omp atomic update
            mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
        }
    }

    return 0;
}


int ptiOmpMTTKRP_Lock(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    ptiMutexPool * lock_pool)
{
    ptiIndex const nmodes = X->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRP_3D_Lock(X, mats, mats_order, mode, tk, lock_pool) == 0);
        return 0;
    }

    ptiNnzIndex const nnz = X->nnz;
    ptiIndex const * const ndims = X->ndims;
    ptiValue const * const vals = X->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiIndex const * const mode_ind = X->inds[mode].data;
    ptiValue * const mvals = mats[nmodes]->values;
    memset(mvals, 0, tmpI*stride*sizeof(ptiValue));

    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiNnzIndex x=0; x<nnz; ++x) {
        ptiValueVector scratch;  // Temporary array
        ptiNewValueVector(&scratch, R, R);
        ptiConstantValueVector(&scratch, 0);

        ptiIndex times_mat_index = mats_order[1];
        ptiMatrix * times_mat = mats[times_mat_index];
        ptiIndex * times_inds = X->inds[times_mat_index].data;
        ptiIndex tmp_i = times_inds[x];
        ptiValue const entry = vals[x];
        for(ptiIndex r=0; r<R; ++r) {
            scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
        }

        for(ptiIndex i=2; i<nmodes; ++i) {
            times_mat_index = mats_order[i];
            times_mat = mats[times_mat_index];
            times_inds = X->inds[times_mat_index].data;
            tmp_i = times_inds[x];

            for(ptiIndex r=0; r<R; ++r) {
                scratch.data[r] *= times_mat->values[tmp_i * stride + r];
            }
        }

        ptiIndex const mode_i = mode_ind[x];
        ptiValue * const restrict mvals_row = mvals + mode_i * stride;

        ptiMutexSetLock(lock_pool, mode_i);
        for(ptiIndex r=0; r<R; ++r) {
            mvals_row[r] += scratch.data[r];
        }
        ptiMutexUnsetLock(lock_pool, mode_i);

        ptiFreeValueVector(&scratch);
    }   // End loop nnzs

    return 0;
}



int ptiOmpMTTKRP_3D_Lock(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk,
    ptiMutexPool * lock_pool)
{
    ptiIndex const nmodes = X->nmodes;
    ptiNnzIndex const nnz = X->nnz;
    ptiIndex const * const ndims = X->ndims;
    ptiValue const * const restrict vals = X->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiIndex const * const restrict mode_ind = X->inds[mode].data;
    ptiValue * const restrict mvals = mats[nmodes]->values;
    memset(mvals, 0, tmpI*stride*sizeof(ptiValue));

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    ptiIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;

    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiNnzIndex x=0; x<nnz; ++x) {
        ptiValueVector scratch;  // Temporary array
        ptiNewValueVector(&scratch, R, R);
        ptiConstantValueVector(&scratch, 0);

        ptiIndex mode_i = mode_ind[x];
        ptiValue * const restrict mvals_row = mvals + mode_i * stride;
        ptiIndex tmp_i_1 = times_inds_1[x];
        ptiIndex tmp_i_2 = times_inds_2[x];
        ptiValue entry = vals[x];

        for(ptiIndex r=0; r<R; ++r) {
            scratch.data[r] = entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
        }

        ptiMutexSetLock(lock_pool, mode_i);
        for(ptiIndex r=0; r<R; ++r) {
            mvals_row[r] += scratch.data[r];
        }
        ptiMutexUnsetLock(lock_pool, mode_i);

        ptiFreeValueVector(&scratch);
    }

    return 0;
}




int ptiOmpMTTKRP_Reduce(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = X->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiOmpMTTKRP_3D_Reduce(X, mats, copy_mats, mats_order, mode, tk) == 0);
        return 0;
    }

    ptiNnzIndex const nnz = X->nnz;
    ptiIndex const * const ndims = X->ndims;
    ptiValue const * const vals = X->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiIndex const * const mode_ind = X->inds[mode].data;
    ptiMatrix * const M = mats[nmodes];
    ptiValue * const mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(ptiValue));
    for(int t=0; t<tk; ++t) {
        memset(copy_mats[t]->values, 0, ndims[mode]*stride*sizeof(*(copy_mats[t]->values)));
    }

    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiNnzIndex x=0; x<nnz; ++x) {
        int tid = omp_get_thread_num();

        ptiValueVector scratch;  // Temporary array
        ptiNewValueVector(&scratch, R, R);
        ptiConstantValueVector(&scratch, 0);

        ptiIndex times_mat_index = mats_order[1];
        ptiMatrix * times_mat = mats[times_mat_index];
        ptiIndex * times_inds = X->inds[times_mat_index].data;
        ptiIndex tmp_i = times_inds[x];
        ptiValue const entry = vals[x];
        #pragma omp simd
        for(ptiIndex r=0; r<R; ++r) {
            scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
        }

        for(ptiIndex i=2; i<nmodes; ++i) {
            times_mat_index = mats_order[i];
            times_mat = mats[times_mat_index];
            times_inds = X->inds[times_mat_index].data;
            tmp_i = times_inds[x];

            #pragma omp simd
            for(ptiIndex r=0; r<R; ++r) {
                scratch.data[r] *= times_mat->values[tmp_i * stride + r];
            }
        }

        ptiIndex const mode_i = mode_ind[x];
        #pragma omp simd
        for(ptiIndex r=0; r<R; ++r) {
            copy_mats[tid]->values[mode_i * stride + r] += scratch.data[r];
        }

        ptiFreeValueVector(&scratch);
    }   // End loop nnzs

    /* Reduction */
    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiIndex i=0; i<ndims[mode]; ++i) {
        for(int t=0; t<tk; ++t) {
            #pragma omp simd
            for(ptiIndex r=0; r<R; ++r) {
                mvals[i * stride + r] += copy_mats[t]->values[i * stride + r];
            }
        }
    }

    return 0;
}



int ptiOmpMTTKRP_3D_Reduce(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiMatrix * copy_mats[],    // temporary matrices for reduction
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode,
    const int tk) 
{
    ptiIndex const nmodes = X->nmodes;
    ptiNnzIndex const nnz = X->nnz;
    ptiIndex const * const ndims = X->ndims;
    ptiValue const * const restrict vals = X->values.data;
    ptiIndex const stride = mats[0]->stride;

    /* Check the mats. */
    ptiAssert(nmodes ==3);
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }

    ptiIndex const tmpI = mats[mode]->nrows;
    ptiIndex const R = mats[mode]->ncols;
    ptiIndex const * const restrict mode_ind = X->inds[mode].data;
    ptiMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(ptiValue));
    for(int t=0; t<tk; ++t) {
        memset(copy_mats[t]->values, 0, ndims[mode]*stride*sizeof(*(copy_mats[t]->values)));
    }

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    ptiIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;

    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiNnzIndex x=0; x<nnz; ++x) {
        int tid = omp_get_thread_num();

        ptiIndex mode_i = mode_ind[x];
        ptiIndex tmp_i_1 = times_inds_1[x];
        ptiIndex tmp_i_2 = times_inds_2[x];
        ptiValue entry = vals[x];

        #pragma omp simd
        for(ptiIndex r=0; r<R; ++r) {
            copy_mats[tid]->values[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
        }
    }

    /* Reduction */
    #pragma omp parallel for schedule(static) num_threads(tk)
    for(ptiIndex i=0; i<ndims[mode]; ++i) {
        for(int t=0; t<tk; ++t) {
            #pragma omp simd
            for(ptiIndex r=0; r<R; ++r) {
                mvals[i * stride + r] += copy_mats[t]->values[i * stride + r];
            }
        }
    }

    return 0;
}