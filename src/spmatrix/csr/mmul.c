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
#include <stdio.h>

/**
  * SpMV, y = Ax
  */
int ptiSparseMatrixMulMatrixCSR(ptiMatrix * C, ptiSparseMatrixCSR *csrmtx, ptiMatrix * B)
{
    for(ptiIndex i = 0; i < csrmtx->nrows; ++i) {
        for(ptiNnzIndex z = csrmtx->rowptr.data[i]; z < csrmtx->rowptr.data[i+1]; ++z) {
            ptiIndex col = csrmtx->colind.data[z];
            ptiValue val = csrmtx->values.data[z];
            for(ptiNnzIndex c = 0; c < B->ncols; ++c) {
                C->values[i * C->stride + c] += val * B->values[col * B->stride + c];
            }
        }
    }
    return 0;
}

#ifdef HIPARTI_USE_OPENMP
int ptiOmpSparseMatrixMulMatrixCSR(ptiMatrix * C, ptiSparseMatrixCSR *csrmtx, ptiMatrix * B)
{
    #pragma omp parallel for
    for(ptiIndex i = 0; i < csrmtx->nrows; ++i) {
        for(ptiNnzIndex z = csrmtx->rowptr.data[i]; z < csrmtx->rowptr.data[i+1]; ++z) {
            ptiIndex col = csrmtx->colind.data[z];
            ptiValue val = csrmtx->values.data[z];
            for(ptiNnzIndex c = 0; c < B->ncols; ++c) {
                C->values[i * C->stride + c] += val * B->values[col * B->stride + c];
            }
        }
    }
    return 0;
}
#endif