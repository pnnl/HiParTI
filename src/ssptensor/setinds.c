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

#include <assert.h>
#include <HiParTI.h>
#include "ssptensor.h"
#include "../sptensor/sptensor.h"

static int pti_SparseTensorCompareExceptMode(const ptiSparseTensor *tsr1, ptiNnzIndex ind1, const ptiSparseTensor *tsr2, ptiNnzIndex ind2, ptiIndex mode);

/**
 * Convert a sparse tensor into a semi sparse tensor, but only set the indices
 * without setting any actual data
 * @param[out] dest     a pointer to an uninitialized semi sparse tensor
 * @param[out] fiberidx a vector to store the starting position of each fiber, should be uninitialized
 * @param[in]  ref      a pointer to a valid sparse tensor
 */
int ptiSemiSparseTensorSetIndices(
    ptiSemiSparseTensor *dest,
    ptiNnzIndexVector *fiberidx,
    ptiSparseTensor *ref
) {
    ptiNnzIndex lastidx = ref->nnz;
    ptiNnzIndex i;
    ptiIndex m;
    int result;
    assert(dest->nmodes == ref->nmodes);
    ptiSparseTensorSortIndexAtMode(ref, dest->mode, 0);
    result = ptiNewNnzIndexVector(fiberidx, 0, 0);
    pti_CheckError(result, "SspTns SetIndices", NULL);
    dest->nnz = 0;
    for(i = 0; i < ref->nnz; ++i) {
        if(lastidx == ref->nnz || pti_SparseTensorCompareExceptMode(ref, lastidx, ref, i, dest->mode) != 0) {
            for(m = 0; m < dest->nmodes; ++m) {
                if(m != dest->mode) {
                    result = ptiAppendIndexVector(&dest->inds[m], ref->inds[m].data[i]);
                    pti_CheckError(result, "SspTns SetIndices", NULL);
                }
            }
            lastidx = i;
            ++dest->nnz;
            if(fiberidx != NULL) {
                result = ptiAppendNnzIndexVector(fiberidx, i);
                pti_CheckError(result, "SspTns SetIndices", NULL);
            }
        }
    }
    if(fiberidx != NULL) {
        result = ptiAppendNnzIndexVector(fiberidx, ref->nnz);
        pti_CheckError(result, "SspTns SetIndices", NULL);
    }
    result = ptiResizeMatrix(&dest->values, dest->nnz);
    pti_CheckError(result, "SspTns SetIndices", NULL);
    memset(dest->values.values, 0, dest->nnz * dest->stride * sizeof (ptiValue));
    return 0;
}

static int pti_SparseTensorCompareExceptMode(const ptiSparseTensor *tsr1, ptiNnzIndex ind1, const ptiSparseTensor *tsr2, ptiNnzIndex ind2, ptiIndex mode) {
    ptiIndex i;
    ptiIndex eleind1, eleind2;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes; ++i) {
        if(i != mode) {
            eleind1 = tsr1->inds[i].data[ind1];
            eleind2 = tsr2->inds[i].data[ind2];
            if(eleind1 < eleind2) {
                return -1;
            } else if(eleind1 > eleind2) {
                return 1;
            }
        }
    }
    return 0;
}
