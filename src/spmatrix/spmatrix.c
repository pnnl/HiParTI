/*
    This file is part of HiParTI!.

    HiParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    HiParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with HiParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <HiParTI.h>
#include <stdlib.h>
#include <string.h>


int ptiNewSparseMatrix(ptiSparseMatrix *mtx, ptiIndex nrows, ptiIndex ncols, ptiIndex nnz)
{
    int result;
    mtx->nrows = nrows;
    mtx->ncols = ncols;
    mtx->nnz = nnz;
    result = ptiNewIndexVector(&mtx->rowind, nnz, nnz);
    pti_CheckError(result, "SpMtx New", NULL);
    result = ptiNewIndexVector(&mtx->colind, nnz, nnz);
    pti_CheckError(result, "SpMtx New", NULL);
    result = ptiNewValueVector(&mtx->values, nnz, nnz);
    pti_CheckError(result, "SpMtx New", NULL);

    return 0;
}

int ptiCopySparseMatrix(ptiSparseMatrix *dest, const ptiSparseMatrix *src, int const nt) {
    int result;
    dest->nrows = src->nrows;
    dest->ncols = src->ncols;
    dest->nnz = src->nnz;
    result = ptiCopyIndexVector(&dest->rowind, &src->rowind, nt);
    pti_CheckError(result, "SpMtx Copy", NULL);
    result = ptiCopyIndexVector(&dest->colind, &src->colind, nt);
    pti_CheckError(result, "SpMtx Copy", NULL);
    result = ptiCopyValueVector(&dest->values, &src->values, nt);
    pti_CheckError(result, "SpMtx Copy", NULL);
    return 0;
}

void ptiSparseMatrixShuffleIndices(ptiSparseMatrix *mtx, ptiIndex ** map_inds) {
    /* Renumber nonzero elements */
    ptiIndex tmp_ind;
    for(ptiNnzIndex z = 0; z < mtx->nnz; ++z) {
        tmp_ind = mtx->rowind.data[z];
        mtx->rowind.data[z] = map_inds[0][tmp_ind];
        tmp_ind = mtx->colind.data[z];
        mtx->colind.data[z] = map_inds[1][tmp_ind];
    }
    
}

void ptiFreeSparseMatrix(ptiSparseMatrix *mtx) {
    ptiFreeIndexVector(&mtx->rowind);
    ptiFreeIndexVector(&mtx->colind);
    ptiFreeValueVector(&mtx->values);
    mtx->nrows = 0;
    mtx->ncols = 0;
    mtx->nnz = 0;
}