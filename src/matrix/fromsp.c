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
#include <string.h>

/**
 * Convert sparse matrix (in a tensor data structure) to dense matrix
 *
 * @param dest pointer to an uninitialized matrix
 * @param src  pointer to a valid sparse tensor
 */
int ptiSparseTensorToMatrix(ptiMatrix *dest, const ptiSparseTensor *src) {
    ptiNnzIndex i;
    int result;
    if(src->nmodes != 2) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "SpTns -> Mtx", "shape mismatch");
    }
    result = ptiNewMatrix(dest, src->ndims[0], src->ndims[1]);
    pti_CheckError(result, "SpTns -> Mtx", NULL);
    memset(dest->values, 0, dest->nrows * dest->stride * sizeof (ptiValue));
    for(i = 0; i < src->nnz; ++i) {
        dest->values[src->inds[0].data[i] * dest->stride + src->inds[1].data[i]] = src->values.data[i];
    }
    return 0;
}
