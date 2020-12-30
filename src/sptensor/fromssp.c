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
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Convert a semi sparse tensor into a sparse tensor
 * @param[out] dest    a pointer to an uninitialized sparse tensor
 * @param[in]  src     a pointer to a valid semi sparse tensor
 * @param      epsilon a small positive value, usually 1e-6, which is considered approximately equal to zero
 */
int ptiSemiSparseTensorToSparseTensor(ptiSparseTensor *dest, const ptiSemiSparseTensor *src, ptiValue epsilon) {
    ptiNnzIndex i;
    int result;
    ptiIndex nmodes = src->nmodes;
    assert(epsilon > 0);
    dest->nmodes = nmodes;
    dest->sortorder = malloc(nmodes * sizeof dest->sortorder[0]);
    dest->ndims = malloc(nmodes * sizeof *dest->ndims);
    pti_CheckOSError(!dest->ndims, "SspTns -> SpTns");
    memcpy(dest->ndims, src->ndims, nmodes * sizeof *dest->ndims);
    dest->nnz = 0;
    dest->inds = malloc(nmodes * sizeof *dest->inds);
    pti_CheckOSError(!dest->inds, "SspTns -> SpTns");
    for(i = 0; i < nmodes; ++i) {
        result = ptiNewIndexVector(&dest->inds[i], 0, src->nnz);
        pti_CheckError(result, "SspTns -> SpTns", NULL);
    }
    result = ptiNewValueVector(&dest->values, 0, src->nnz);
    pti_CheckError(result, "SspTns -> SpTns", NULL);
    for(i = 0; i < src->nnz; ++i) {
        ptiIndex j;
        for(j = 0; j < src->ndims[src->mode]; ++j) {
            ptiValue data = src->values.values[i*src->stride + j];
            int data_class = fpclassify(data);
            if(
                data_class == FP_NAN ||
                data_class == FP_INFINITE ||
                (data_class == FP_NORMAL && !(data < epsilon && data > -epsilon))
            ) {
                ptiIndex m;
                for(m = 0; m < nmodes; ++m) {
                    if(m != src->mode) {
                        result = ptiAppendIndexVector(&dest->inds[m], src->inds[m].data[i]);
                    } else {
                        result = ptiAppendIndexVector(&dest->inds[src->mode], j);
                    }
                    pti_CheckError(result, "SspTns -> SpTns", NULL);
                }
                result = ptiAppendValueVector(&dest->values, data);
                pti_CheckError(result, "SspTns -> SpTns", NULL);
                ++dest->nnz;
            }
        }
    }
    ptiSparseTensorSortIndex(dest, 1, 1);
    return 0;
}
