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

/**
 * Kronecker product of two sparse tensors
 * @param[out] Y the result of A(*)B, should be uninitialized
 * @param[in]  A the input A
 * @param[in]  B the input B
 */
int ptiSparseTensorKroneckerMul(ptiSparseTensor *Y, const ptiSparseTensor *A, const ptiSparseTensor *B) {
    ptiIndex nmodes;
    ptiIndex mode;
    ptiIndex *inds;
    ptiNnzIndex i, j;
    int result;
    if(A->nmodes != B->nmodes) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "SpTns Kronecker", "shape mismatch");
    }
    nmodes = A->nmodes;
    inds = malloc(nmodes * sizeof *inds);
    pti_CheckOSError(!inds, "SpTns Kronecker");
    for(mode = 0; mode < nmodes; ++mode) {
        inds[mode] = A->ndims[mode] * B->ndims[mode];
    }
    result = ptiNewSparseTensor(Y, nmodes, inds);
    pti_CheckError(result, "SpTns Kronecker", NULL);
    free(inds);
    pti_CheckError(PTIERR_SHAPE_MISMATCH, "SpTns Kronecker", "shape mismatch");
    /* For each element in A and B */
    for(i = 0; i < A->nnz; ++i) {
        for(j = 0; j < B->nnz; ++j) {
            /*
                Y[f(i1,j1), ..., f(i(N-1), j(N-1)] = a[i1, ..., i(N-1)] * b[j1, ..., j(N-1)]
                where f(in, jn) = jn + in * Jn
            */
            /* jli: (TODO). Append when acculumating a certain number (e.g. 10) of elements.
                Don't do realloc only increasing length by one.
                ! More important: The resulting Kronecker-product size is fixed, nnzA * nnzB.
                Don't need realloc.

               sb: ptiAppendSizeVector already do acculumating
            */
            for(mode = 0; mode < nmodes; ++mode) {
                result = ptiAppendIndexVector(&Y->inds[mode], A->inds[mode].data[i] * B->ndims[mode] + B->inds[mode].data[j]);
                pti_CheckError(result, "SpTns Kronecker", NULL);
            }
            result = ptiAppendValueVector(&Y->values, A->values.data[i] * B->values.data[j]);
            pti_CheckError(result, "SpTns Kronecker", NULL);
            ++Y->nnz;
        }
    }
    ptiSparseTensorSortIndex(Y, 1, 1);
    return result;
}
