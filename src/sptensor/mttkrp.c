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

int ptiMTTKRP_3D(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode);

/**
 * Matriced sparse tensor times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  X    the sparse tensor input X
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products
 * @param[in]  mode   the mode on which the MTTKRP is performed
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 */
int ptiMTTKRP(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode) {

    ptiIndex const nmodes = X->nmodes;

    if(nmodes == 3) {
        ptiAssert(ptiMTTKRP_3D(X, mats, mats_order, mode) == 0);
        return 0;
    }

    ptiNnzIndex const nnz = X->nnz;
    ptiIndex const * const ndims = X->ndims;
    ptiValue const * const restrict vals = X->values.data;
    ptiIndex const stride = mats[0]->stride;
    ptiValueVector scratch;  // Temporary array

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
    ptiIndex const * const restrict mode_ind = X->inds[mode].data;
    ptiMatrix * const restrict M = mats[nmodes];
    ptiValue * const restrict mvals = M->values;
    memset(mvals, 0, tmpI*stride*sizeof(ptiValue));
    ptiNewValueVector(&scratch, R, R);
    ptiConstantValueVector(&scratch, 0);


    for(ptiNnzIndex x=0; x<nnz; ++x) {

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
        for(ptiIndex r=0; r<R; ++r) {
            mvals[mode_i * stride + r] += scratch.data[r];
        }
    }

    ptiFreeValueVector(&scratch);

    return 0;
}


int ptiMTTKRP_3D(ptiSparseTensor const * const X,
    ptiMatrix * mats[],     // mats[nmodes] as temporary space.
    ptiIndex const mats_order[],    // Correspond to the mode order of X.
    ptiIndex const mode)
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

    ptiIndex times_mat_index_1 = mats_order[1];
    ptiMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    ptiIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    ptiIndex times_mat_index_2 = mats_order[2];
    ptiMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    ptiIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;

    ptiIndex mode_i;
    ptiIndex tmp_i_1, tmp_i_2;
    ptiValue entry;
    for(ptiNnzIndex x=0; x<nnz; ++x) {
        mode_i = mode_ind[x];
        tmp_i_1 = times_inds_1[x];
        tmp_i_2 = times_inds_2[x];
        entry = vals[x];

        for(ptiIndex r=0; r<R; ++r) {
            mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
        }
    }

    return 0;
}
