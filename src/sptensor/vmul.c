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
 * Sparse tensor times a vector (SpTTV)
 */
int ptiSparseTensorMulVector(ptiSemiSparseTensor *Y, ptiSparseTensor * const X, ptiValueVector * const V, ptiIndex mode)
{
    int result;
    ptiIndex *ind_buf;
    ptiNnzIndexVector fiberidx;
    if(mode >= X->nmodes) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns * Vec", "shape mismatch");
    }
    if(X->ndims[mode] != V->len) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "CPU  SpTns * Vec", "shape mismatch");
    }
    ptiSparseTensorSortIndexAtMode(X, mode, 0);
    // jli: try to avoid malloc in all operation functions.
    ind_buf = malloc(X->nmodes * sizeof *ind_buf);
    pti_CheckOSError(!ind_buf, "CPU  SpTns * Vec");
    for(ptiIndex m = 0; m < X->nmodes; ++m) {
        ind_buf[m] = X->ndims[m];
    }
    ind_buf[mode] = 1;
    // jli: use pre-processing to allocate Y size outside this function.
    result = ptiNewSemiSparseTensor(Y, X->nmodes, mode, ind_buf);
    free(ind_buf);
    pti_CheckError(result, "CPU  SpTns * Vec", NULL);
    ptiSemiSparseTensorSetIndices(Y, &fiberidx, X);

    ptiTimer timer;
    ptiNewTimer(&timer, 0);
    ptiStartTimer(timer);

    for(ptiNnzIndex i = 0; i < Y->nnz; ++i) {
        ptiNnzIndex inz_begin = fiberidx.data[i];
        ptiNnzIndex inz_end = fiberidx.data[i+1];
        ptiNnzIndex j;
        // jli: exchange the two loops
        for(j = inz_begin; j < inz_end; ++j) {
            ptiIndex r = X->inds[mode].data[j];
            Y->values.values[i*Y->stride] += X->values.data[j] * V->data[r];
        }
    }

    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "CPU  SpTns * Vec");
    ptiFreeTimer(timer);

    ptiFreeNnzIndexVector(&fiberidx);
    return 0;
}
