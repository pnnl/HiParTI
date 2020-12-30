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
#include <stdlib.h>
#include <string.h>

/* jli: (TODO) Keep this function, but add another Khatri-Rao product for two dense matrices. */
/* jli: (Future TODO) Add Khatri-Rao product for two sparse matrices. */

int ptiSparseTensorKhatriRaoMul(ptiSparseTensor *Y, const ptiSparseTensor *A, const ptiSparseTensor *B) {
    ptiIndex nmodes;
    ptiIndex mode;
    ptiIndex *inds;
    ptiNnzIndex i, j;
    int result;
    if(A->nmodes != B->nmodes) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "Khatri-Rao", "shape mismatch");
    }
    nmodes = A->nmodes;
    if(nmodes == 0) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "Khatri-Rao", "shape mismatch");
    }
    if(A->ndims[nmodes-1] != B->ndims[nmodes-1]) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "Khatri-Rao", "shape mismatch");
    }
    inds = malloc(nmodes * sizeof *inds);
    pti_CheckOSError(!inds, "Khatri-Rao");
    for(mode = 0; mode < nmodes-1; ++mode) {
        inds[mode] = A->ndims[mode] * B->ndims[mode];
    }
    inds[nmodes-1] = A->ndims[mode];
    result = ptiNewSparseTensor(Y, nmodes, inds);
    free(inds);
    pti_CheckError(result, "Khatri-Rao", NULL);
    /* For each element in A and B */
    for(i = 0; i < A->nnz; ++i) {
        for(j = 0; j < B->nnz; ++j) {
            if(A->inds[nmodes-1].data[i] == B->inds[nmodes-1].data[j]) {
                /*
                    Y[f(i0,j0), ..., f(i(N-2), j(N-2))] = a[i10 ..., i(N-2)] * b[j0, ..., j(N-2)]
                    where f(in, jn) = jn + in * Jn
                */
                for(mode = 0; mode < nmodes-1; ++mode) {
                    ptiAppendIndexVector(&Y->inds[mode], A->inds[mode].data[i] * B->ndims[mode] + B->inds[mode].data[j]);
                }
                ptiAppendIndexVector(&Y->inds[nmodes-1], A->inds[nmodes-1].data[i]);
                ptiAppendValueVector(&Y->values, A->values.data[i] * B->values.data[j]);
                ++Y->nnz;
            }
        }
    }
    ptiSparseTensorSortIndex(Y, 1, 1);
    return 0;
}
