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
#include <stdio.h>


int ptiDumpSparseMatrixCSR(ptiSparseMatrixCSR * const csrmtx, FILE *fp)
{
    int iores;

    iores = fprintf(fp, "%"HIPARTI_PRI_INDEX "x%"HIPARTI_PRI_INDEX ", ", csrmtx->nrows, csrmtx->ncols);
    pti_CheckOSError(iores < 0, "SpMtx Dump");
    iores = fprintf(fp, "%"HIPARTI_PRI_NNZ_INDEX "\n", csrmtx->nnz);
    pti_CheckOSError(iores < 0, "SpMtx Dump");

    fprintf(fp, "rowptr:\n");
    ptiDumpNnzIndexVector(&csrmtx->rowptr, fp);
    fprintf(fp, "colind:\n");
    ptiDumpIndexVector(&csrmtx->colind, fp);
    fprintf(fp, "values:\n");
    ptiDumpValueVector(&csrmtx->values, fp);

    return 0;
}
