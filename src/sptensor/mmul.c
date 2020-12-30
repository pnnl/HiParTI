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
#include <stdlib.h>
#include "sptensor.h"

/**
 * Sparse tensor times a dense matrix (SpTTM)
 * @param[out] Y    the result of X*U, should be uninitialized
 * @param[in]  X    the sparse tensor input X
 * @param[in]  U    the dense matrix input U
 * @param      mode the mode on which the multiplication is done on
 *
 * This function will sort Y with `ptiSparseTensorSortIndexAtMode`
 * automatically, this operation can be undone with `ptiSparseTensorSortIndex`
 * if you need to access raw data.
 * Anyway, you do not have to take this side-effect into consideration if you
 * do not need to access raw data.
 */
int ptiSparseTensorMulMatrix(ptiSemiSparseTensor *Y, ptiSparseTensor * const X, ptiMatrix *const U, ptiIndex mode)
{
    int result;
    ptiIndex *ind_buf;
    ptiIndex m;
    ptiNnzIndex i;
    ptiNnzIndexVector fiberidx;
    if(mode >= X->nmodes) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns * Mtx", "shape mismatch");
    }
    if(X->ndims[mode] != U->nrows) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns * Mtx", "shape mismatch");
    }
    ptiSparseTensorSortIndexAtMode(X, mode, 0);
    // jli: try to avoid malloc in all operation functions.
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    pti_CheckOSError(!ind_buf, "CPU  SpTns * Mtx");
    for(m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = U->ncols;
    // jli: use pre-processing to allocate Y size outside this function.
    result = ptiNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    free(ind_buf);
    pti_CheckError(result, "CPU  SpTns * Mtx", NULL);
    ptiSemiSparseTensorSetIndices(Y, &fiberidx, X);

    ptiTimer timer;
    ptiNewTimer(&timer, 0);
    ptiStartTimer(timer);

    for(i = 0; i < Y->nnz; ++i) {
        ptiNnzIndex inz_begin = fiberidx.data[i];
        ptiNnzIndex inz_end = fiberidx.data[i+1];
        // jli: exchange the two loops
        for(ptiNnzIndex j = inz_begin; j < inz_end; ++j) {
            ptiIndex r = X->inds[mode].data[j];
            for(ptiIndex k = 0; k < U->ncols; ++k) {
                Y->values.values[i*Y->stride + k] += X->values.data[j] * U->values[r*U->stride + k];
            }
        }
    }

    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "CPU  SpTns * Mtx");
    ptiFreeTimer(timer);

    ptiFreeNnzIndexVector(&fiberidx);
    return 0;
}
