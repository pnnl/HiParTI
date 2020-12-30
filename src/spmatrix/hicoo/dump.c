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


int ptiDumpSparseMatrixHiCOO(ptiSparseMatrixHiCOO * const himtx, FILE *fp)
{
    int iores;

    iores = fprintf(fp, "%"HIPARTI_PRI_INDEX "x%"HIPARTI_PRI_INDEX ", ", himtx->nrows, himtx->ncols);
    pti_CheckOSError(iores < 0, "SpMtx Dump");
    iores = fprintf(fp, "%"HIPARTI_PRI_NNZ_INDEX "\n", himtx->nnz);
    pti_CheckOSError(iores < 0, "SpMtx Dump");

    fprintf(fp, "bptr:\n");
    ptiDumpNnzIndexVector(&himtx->bptr, fp);
    fprintf(fp, "bindI:\n");
    ptiDumpBlockIndexVector(&himtx->bindI, fp);
    fprintf(fp, "bindJ:\n");
    ptiDumpBlockIndexVector(&himtx->bindJ, fp);
    fprintf(fp, "eindI:\n");
    ptiDumpElementIndexVector(&himtx->eindI, fp);
    fprintf(fp, "eindJ:\n");
    ptiDumpElementIndexVector(&himtx->eindJ, fp);
    fprintf(fp, "values:\n");
    ptiDumpValueVector(&himtx->values, fp);

    ptiIndex sk = (ptiIndex)pow(2, himtx->sk_bits);
    fprintf(fp, "Superblock scheduler:\n");
    fprintf(fp, "nkiters: %"HIPARTI_PRI_INDEX"\n", himtx->nkiters);
    fprintf(fp, "kschr:\n");
    for(ptiIndex i=0; i<(himtx->nrows + sk - 1)/sk; ++i) {
        ptiDumpIndexVector(&himtx->kschr[i], fp);
    }
    fprintf(fp, "kptr:\n");
    ptiDumpNnzIndexVector(&himtx->kptr, fp);

    return 0;
}
