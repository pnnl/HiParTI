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
#include "sptensor.h"

/**
 * Convert a sparse tensor into a semi sparse tensor, but only set the indices
 * without setting any actual data
 * @param[out] dest     a pointer to an uninitialized semi sparse tensor
 * @param[out] fiberidx a vector to store the starting position of each fiber, should be uninitialized
 * @param[in]  ref      a pointer to a valid sparse tensor
 */
int ptiSparseTensorSetIndices(
    ptiSparseTensor *ref,
    ptiIndex * mode_order,
    ptiIndex num_cmodes,
    ptiNnzIndexVector *fiberidx)
{
    int result;
    result = ptiNewNnzIndexVector(fiberidx, 0, 0);
    pti_CheckError(result, "SspTns SetIndices", NULL);

    ptiNnzIndex lastidx = ref->nnz;
    for(ptiNnzIndex i = 0; i < ref->nnz; ++i) {
        if(lastidx == ref->nnz || pti_SparseTensorCompareIndicesCustomize(ref, lastidx, mode_order, ref, i, mode_order, num_cmodes) != 0) {
            lastidx = i;
            if(fiberidx != NULL) {
                result = ptiAppendNnzIndexVector(fiberidx, i);
                pti_CheckError(result, "SpTns SetIndices", NULL);
            }
        }
    }
    if(fiberidx != NULL) {
        result = ptiAppendNnzIndexVector(fiberidx, ref->nnz);
        pti_CheckError(result, "SspTns SetIndices", NULL);
    }
      
    return 0;
}

