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

int ptiDumpSparseMatrix(const ptiSparseMatrix *mtx, ptiIndex start_index, FILE *fp)
{
    int iores;
    ptiNnzIndex i;
    iores = fprintf(fp, "%"HIPARTI_PRI_INDEX, mtx->nrows);
    pti_CheckOSError(iores < 0, "SpMtx Dump");
    iores = fprintf(fp, " %"HIPARTI_PRI_INDEX, mtx->ncols);
    pti_CheckOSError(iores < 0, "SpMtx Dump");
    iores = fprintf(fp, " %"HIPARTI_PRI_NNZ_INDEX "\n", mtx->nnz);
    pti_CheckOSError(iores < 0, "SpMtx Dump");

    for(i = 0; i < mtx->nnz; ++i) {
        iores = fprintf(fp, "%"HIPARTI_PRI_INDEX "\t", mtx->rowind.data[i] + start_index);
        pti_CheckOSError(iores < 0, "SpMtx Dump");
        iores = fprintf(fp, "%"HIPARTI_PRI_INDEX "\t", mtx->colind.data[i] + start_index);
        pti_CheckOSError(iores < 0, "SpMtx Dump");
        iores = fprintf(fp, "%"HIPARTI_PRI_VALUE "\n", (double) mtx->values.data[i]);
        pti_CheckOSError(iores < 0, "SpMtx Dump");
    }
    return 0;
}
