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

int ptiNewSparseMatrixHiCOO(
    ptiSparseMatrixHiCOO *himtx,
    const ptiIndex nrows,
    const ptiIndex ncols,
    const ptiNnzIndex nnz,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits)
{
    int result;

    himtx->nrows = nrows;
    himtx->ncols = ncols;
    himtx->nnz = nnz;

    /* Parameters */
    himtx->sb_bits = sb_bits; // block size by nnz
    himtx->sk_bits = sk_bits; // superblock size by nnz

    result = ptiNewNnzIndexVector(&himtx->bptr, 0, 0);
    pti_CheckError(result, "HiSpMtx New", NULL);
    result = ptiNewBlockIndexVector(&himtx->bindI, 0, 0);
    pti_CheckError(result, "HiSpMtx New", NULL);
    result = ptiNewBlockIndexVector(&himtx->bindJ, 0, 0);
    pti_CheckError(result, "HiSpMtx New", NULL);

    result = ptiNewElementIndexVector(&himtx->eindI, 0, 0);
    pti_CheckError(result, "HiSpMtx New", NULL);
    result = ptiNewElementIndexVector(&himtx->eindJ, 0, 0);
    pti_CheckError(result, "HiSpMtx New", NULL);

    result = ptiNewValueVector(&himtx->values, 0, 0);
    pti_CheckError(result, "HiSpMtx New", NULL);

    /* Allocate superblock scheduler */
    ptiIndex sk = (ptiIndex)pow(2, sk_bits);
    ptiIndex kernel_ndim = (nrows + sk - 1)/sk;
    himtx->kschr = (ptiIndexVector*)malloc(kernel_ndim * sizeof(*(himtx->kschr)));
    pti_CheckOSError(!himtx->kschr, "HiSpTns New");
    for(ptiIndex i = 0; i < kernel_ndim; ++i) {
        result = ptiNewIndexVector(&(himtx->kschr[i]), 0, 0);
        pti_CheckError(result, "HiSpTns New", NULL);
    }
    himtx->nkiters = 0;

    result = ptiNewNnzIndexVector(&himtx->kptr, 0, 0);
    pti_CheckError(result, "HiSpTns New", NULL);

    return 0;
}


void ptiFreeSparseMatrixHiCOO(ptiSparseMatrixHiCOO *himtx)
{
    ptiFreeNnzIndexVector(&himtx->bptr);
    ptiFreeBlockIndexVector(&himtx->bindI);
    ptiFreeBlockIndexVector(&himtx->bindJ);
    ptiFreeElementIndexVector(&himtx->eindI);
    ptiFreeElementIndexVector(&himtx->eindJ);
    ptiFreeValueVector(&himtx->values);

    ptiFreeNnzIndexVector(&himtx->kptr);
    ptiIndex sk = (ptiIndex)pow(2, himtx->sk_bits);
    ptiIndex kernel_ndim = (himtx->nrows + sk - 1)/sk;
    for(ptiIndex i = 0; i < kernel_ndim; ++i) {
        ptiFreeIndexVector(&(himtx->kschr[i]));
    }
    free(himtx->kschr);

    himtx->nnz = 0;
    himtx->nrows = 0;
    himtx->ncols = 0;
    himtx->sb_bits = 0;
    himtx->sk_bits = 0;
    himtx->nkiters = 0;
}
