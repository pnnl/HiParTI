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

/**
 * Element wise divide two sparse tensors
 * @param[out] Z the result of X/Y, should be uninitialized
 * @param[in]  X the input X
 * @param[in]  Y the input Y
 *
 * The name "DotDiv" comes from the MATLAB operator "./".
 */
int ptiSparseTensorDotDiv(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y)
{
    ptiNnzIndex i, j;
    int result;
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "SpTns DotDiv", "shape mismatch");
    }
    for(i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "SpTns DotDiv", "shape mismatch");
        }
    }

    ptiNewSparseTensor(Z, X->nmodes, X->ndims);

    /* Multiply elements one by one, assume indices are ordered */
    i = 0;
    j = 0;
    while(i < X->nnz && j < Y->nnz) {
        int compare = pti_SparseTensorCompareIndices(X, i, Y, j);

        if(compare > 0) {
            ++j;
        } else if(compare < 0) {
            ++i;
        } else {
            for(ptiIndex mode = 0; mode < X->nmodes; ++mode) {
                result = ptiAppendIndexVector(&Z->inds[mode], X->inds[mode].data[i] * Y->inds[mode].data[j]);
                pti_CheckError(result, "SpTns DotDiv", NULL);
            }
            result = ptiAppendValueVector(&Z->values, X->values.data[i] / Y->values.data[j]);
            pti_CheckError(result, "SpTns DotDiv", NULL);

            ++Z->nnz;
            ++i;
            ++j;
        }
    }

    return 0;
}
