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


int ptiNewSparseMatrixCSR(ptiSparseMatrixCSR *csrmtx, ptiIndex nrows, ptiIndex ncols, ptiIndex nnz)
{
    int result;
    csrmtx->nrows = nrows;
    csrmtx->ncols = ncols;
    csrmtx->nnz = nnz;
    result = ptiNewNnzIndexVector(&csrmtx->rowptr, (nrows+1), (nrows+1));
    pti_CheckError(result, "SpMtx New", NULL);
    result = ptiNewIndexVector(&csrmtx->colind, nnz, nnz);
    pti_CheckError(result, "SpMtx New", NULL);
    result = ptiNewValueVector(&csrmtx->values, nnz, nnz);
    pti_CheckError(result, "SpMtx New", NULL);

    return 0;
}

void ptiFreeSparseMatrixCSR(ptiSparseMatrixCSR *csrmtx) {
    ptiFreeNnzIndexVector(&csrmtx->rowptr);
    ptiFreeIndexVector(&csrmtx->colind);
    ptiFreeValueVector(&csrmtx->values);
    csrmtx->nrows = 0;
    csrmtx->ncols = 0;
    csrmtx->nnz = 0;
}