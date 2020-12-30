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
#include "hicoo.h"

/**
 * Create a new sparse tensor in HiCOO format
 * @param hitsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 * @param nnz number of nonzeros the tensor will have
 */
int ptiNewSparseTensorHiCOO(
    ptiSparseTensorHiCOO *hitsr,
    const ptiIndex nmodes,
    const ptiIndex ndims[],
    const ptiNnzIndex nnz,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits,
    const ptiElementIndex sc_bits)
{
    ptiIndex i;
    int result;

    hitsr->nmodes = nmodes;
    hitsr->sortorder = malloc(nmodes * sizeof hitsr->sortorder[0]);
    for(i = 0; i < nmodes; ++i) {
        hitsr->sortorder[i] = i;
    }
    hitsr->ndims = malloc(nmodes * sizeof *hitsr->ndims);
    pti_CheckOSError(!hitsr->ndims, "HiSpTns New");
    memcpy(hitsr->ndims, ndims, nmodes * sizeof *hitsr->ndims);
    hitsr->nnz = nnz;

    /* Parameters */
    hitsr->sb_bits = sb_bits; // block size by nnz
    hitsr->sk_bits = sk_bits; // kernel size by nnz
    hitsr->sc_bits = sc_bits; // chunk size by blocks
    ptiIndex sk = (ptiIndex)pow(2, sk_bits);

    hitsr->kschr = (ptiIndexVector**)malloc(nmodes * sizeof *hitsr->kschr);
    pti_CheckOSError(!hitsr->kschr, "HiSpTns New");
    for(ptiIndex m = 0; m < nmodes; ++m) {
        ptiIndex kernel_ndim = (ndims[m] + sk - 1)/sk;
        hitsr->kschr[m] = (ptiIndexVector*)malloc(kernel_ndim * sizeof(*(hitsr->kschr[m])));
        pti_CheckOSError(!hitsr->kschr[m], "HiSpTns New");
        for(ptiIndex i = 0; i < kernel_ndim; ++i) {
            result = ptiNewIndexVector(&(hitsr->kschr[m][i]), 0, 0);
            pti_CheckError(result, "HiSpTns New", NULL);
        }
    }
    hitsr->nkiters = (ptiIndex*)malloc(nmodes * sizeof *hitsr->nkiters);

    result = ptiNewNnzIndexVector(&hitsr->kptr, 0, 0);
    pti_CheckError(result, "HiSpTns New", NULL);
    result = ptiNewNnzIndexVector(&hitsr->cptr, 0, 0);
    pti_CheckError(result, "HiSpTns New", NULL);

    /* Balanced structures */
    hitsr->kschr_balanced = (ptiIndexVector**)malloc(nmodes * sizeof *hitsr->kschr_balanced);
    pti_CheckOSError(!hitsr->kschr_balanced, "HiSpTns New");
    for(ptiIndex m = 0; m < nmodes; ++m) {
        ptiIndex kernel_ndim = (ndims[m] + sk - 1)/sk;
        hitsr->kschr_balanced[m] = (ptiIndexVector*)malloc(kernel_ndim * sizeof(*(hitsr->kschr_balanced[m])));
        pti_CheckOSError(!hitsr->kschr_balanced[m], "HiSpTns New");
        for(ptiIndex i = 0; i < kernel_ndim; ++i) {
            result = ptiNewIndexVector(&(hitsr->kschr_balanced[m][i]), 0, 0);
            pti_CheckError(result, "HiSpTns New", NULL);
        }
    }
    hitsr->kschr_balanced_pos = (ptiIndexVector**)malloc(nmodes * sizeof *hitsr->kschr_balanced_pos);
    pti_CheckOSError(!hitsr->kschr_balanced_pos, "HiSpTns New");
    for(ptiIndex m = 0; m < nmodes; ++m) {
        ptiIndex kernel_ndim = (ndims[m] + sk - 1)/sk;
        hitsr->kschr_balanced_pos[m] = (ptiIndexVector*)malloc(kernel_ndim * sizeof(*(hitsr->kschr_balanced_pos[m])));
        pti_CheckOSError(!hitsr->kschr_balanced_pos[m], "HiSpTns New");
        for(ptiIndex i = 0; i < kernel_ndim; ++i) {
            result = ptiNewIndexVector(&(hitsr->kschr_balanced_pos[m][i]), 0, 0);
            pti_CheckError(result, "HiSpTns New", NULL);
        }
    }
    hitsr->nkpars = (ptiIndex*)malloc(nmodes * sizeof(ptiIndex));
    pti_CheckOSError(!hitsr->nkpars, "HiSpTns New");
    hitsr->kschr_rest = (ptiIndexVector*)malloc(nmodes * sizeof *hitsr->kschr_rest);
    pti_CheckOSError(!hitsr->kschr_rest, "HiSpTns New");
    for(ptiIndex m = 0; m < nmodes; ++m) {
        result = ptiNewIndexVector(&(hitsr->kschr_rest[m]), 0, 0);
        pti_CheckError(result, "HiSpTns New", NULL);
    }
    result = ptiNewNnzIndexVector(&hitsr->knnzs, 0, 0);
    pti_CheckError(result, "HiSpTns New", NULL);

    result = ptiNewNnzIndexVector(&hitsr->bptr, 0, 0);
    pti_CheckError(result, "HiSpTns New", NULL);
    hitsr->binds = malloc(nmodes * sizeof *hitsr->binds);
    pti_CheckOSError(!hitsr->binds, "HiSpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = ptiNewBlockIndexVector(&hitsr->binds[i], 0, 0);
        pti_CheckError(result, "HiSpTns New", NULL);
    }

    hitsr->einds = malloc(nmodes * sizeof *hitsr->einds);
    pti_CheckOSError(!hitsr->einds, "HiSpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = ptiNewElementIndexVector(&hitsr->einds[i], 0, 0);
        pti_CheckError(result, "HiSpTns New", NULL);
    }
    result = ptiNewValueVector(&hitsr->values, 0, 0);
    pti_CheckError(result, "HiSpTns New", NULL);


    return 0;
}


/**
 * Create a new sparse tensor in HiCOO format
 * @param hitsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int ptiNewSparseTensorHiCOO_NoNnz(
    ptiSparseTensorHiCOO *hitsr,
    const ptiIndex nmodes,
    const ptiIndex ndims[],
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits,
    const ptiElementIndex sc_bits)
{
    ptiIndex i;
    int result;

    hitsr->nmodes = nmodes;
    hitsr->sortorder = malloc(nmodes * sizeof hitsr->sortorder[0]);
    for(i = 0; i < nmodes; ++i) {
        hitsr->sortorder[i] = i;
    }
    hitsr->ndims = malloc(nmodes * sizeof *hitsr->ndims);
    pti_CheckOSError(!hitsr->ndims, "HiSpTns New");
    memcpy(hitsr->ndims, ndims, nmodes * sizeof *hitsr->ndims);

    /* Parameters */
    hitsr->sb_bits = sb_bits; // block size by nnz
    hitsr->sk_bits = sk_bits; // kernel size by nnz
    hitsr->sc_bits = sc_bits; // chunk size by blocks
    ptiIndex sk = (ptiIndex)pow(2, sk_bits);

    hitsr->kschr = (ptiIndexVector**)malloc(nmodes * sizeof *hitsr->kschr);
    pti_CheckOSError(!hitsr->kschr, "HiSpTns New");
    for(ptiIndex m = 0; m < nmodes; ++m) {
        ptiIndex kernel_ndim = (ndims[m] + sk - 1)/sk;
        hitsr->kschr[m] = (ptiIndexVector*)malloc(kernel_ndim * sizeof(*(hitsr->kschr[m])));
        pti_CheckOSError(!hitsr->kschr[m], "HiSpTns New");
        for(ptiIndex i = 0; i < kernel_ndim; ++i) {
            result = ptiNewIndexVector(&(hitsr->kschr[m][i]), 0, 0);
            pti_CheckError(result, "HiSpTns New", NULL);
        }
    }
    hitsr->nkiters = (ptiIndex*)malloc(nmodes * sizeof *hitsr->nkiters);

    result = ptiNewNnzIndexVector(&hitsr->kptr, 0, 0);
    pti_CheckError(result, "HiSpTns New", NULL);
    result = ptiNewNnzIndexVector(&hitsr->cptr, 0, 0);
    pti_CheckError(result, "HiSpTns New", NULL);

    result = ptiNewNnzIndexVector(&hitsr->bptr, 0, 0);
    pti_CheckError(result, "HiSpTns New", NULL);
    hitsr->binds = malloc(nmodes * sizeof *hitsr->binds);
    pti_CheckOSError(!hitsr->binds, "HiSpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = ptiNewBlockIndexVector(&hitsr->binds[i], 0, 0);
        pti_CheckError(result, "HiSpTns New", NULL);
    }

    hitsr->einds = malloc(nmodes * sizeof *hitsr->einds);
    pti_CheckOSError(!hitsr->einds, "HiSpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = ptiNewElementIndexVector(&hitsr->einds[i], 0, 0);
        pti_CheckError(result, "HiSpTns New", NULL);
    }
    result = ptiNewValueVector(&hitsr->values, 0, 0);
    pti_CheckError(result, "HiSpTns New", NULL);


    return 0;
}


/**
 * Release any memory the HiCOO sparse tensor is holding
 * @param hitsr the tensor to release
 */
void ptiFreeSparseTensorHiCOO(ptiSparseTensorHiCOO *hitsr)
{
    ptiIndex i;
    ptiIndex nmodes = hitsr->nmodes;
    ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);

    for(ptiIndex m = 0; m < nmodes; ++m) {
        ptiIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
        for(i = 0; i < kernel_ndim; ++i) {
            ptiFreeIndexVector(&(hitsr->kschr[m][i]));
        }
        free(hitsr->kschr[m]);
    }
    free(hitsr->kschr);
    free(hitsr->nkiters);

    ptiFreeNnzIndexVector(&hitsr->kptr);
    ptiFreeNnzIndexVector(&hitsr->cptr);

    ptiFreeNnzIndexVector(&hitsr->bptr);
    for(i = 0; i < nmodes; ++i) {
        ptiFreeBlockIndexVector(&hitsr->binds[i]);
        ptiFreeElementIndexVector(&hitsr->einds[i]);
    }
    free(hitsr->binds);
    free(hitsr->einds);
    ptiFreeValueVector(&hitsr->values);

    hitsr->nmodes = 0;
    hitsr->nnz = 0;
    hitsr->sb_bits = 0;
    hitsr->sk_bits = 0;
    hitsr->sc_bits = 0;

    free(hitsr->sortorder);
    free(hitsr->ndims);
}


double SparseTensorFrobeniusNormSquaredHiCOO(ptiSparseTensorHiCOO const * const hitsr)
{
  double norm = 0;
  ptiValue const * const restrict vals = hitsr->values.data;

#ifdef HIPARTI_USE_OPENMP
  #pragma omp parallel for reduction(+:norm)
#endif
  for(size_t n=0; n < hitsr->nnz; ++n) {
    norm += vals[n] * vals[n];
  }
  return norm;
}