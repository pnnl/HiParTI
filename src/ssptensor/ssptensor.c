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
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "ssptensor.h"

/**
 * Create a new semi sparse tensor
 * @param tsr    a pointer to an uninitialized semi sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param mode   the mode which will be stored in dense format
 * @param ndims  the dimension of each mode the tensor will have
 */
int ptiNewSemiSparseTensor(ptiSemiSparseTensor *tsr, ptiIndex nmodes, ptiIndex mode, const ptiIndex ndims[]) {
    ptiIndex i;
    int result;
    if(nmodes < 2) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "SspTns New", "nmodes < 2");
    }
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    pti_CheckOSError(!tsr->ndims, "SspTns New");
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->mode = mode;
    tsr->nnz = 0;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    pti_CheckOSError(!tsr->inds, "SspTns New");
    for(i = 0; i < nmodes; ++i) {
        result = ptiNewIndexVector(&tsr->inds[i], 0, 0);
        pti_CheckError(result, "SspTns New", NULL);
    }
    tsr->stride = ((ndims[mode]-1)/8+1)*8;
    result = ptiNewMatrix(&tsr->values, 0, tsr->stride);
    pti_CheckError(result, "SspTns New", NULL);
    return 0;
}

/**
 * Copy a semi sparse tensor
 * @param[out] dest a pointer to an uninitialized semi sparse tensor
 * @param[in]  src  a pointer to a valid semi sparse tensor
 */
int ptiCopySemiSparseTensor(ptiSemiSparseTensor *dest, const ptiSemiSparseTensor *src) {
    ptiIndex i;
    int result;
    assert(src->nmodes >= 2);
    dest->nmodes = src->nmodes;
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    pti_CheckOSError(!dest->ndims, "SspTns Copy");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->mode = src->mode;
    dest->nnz = src->nnz;
    dest->inds = malloc(dest->nmodes * sizeof *dest->inds);
    pti_CheckOSError(!dest->inds, "SspTns Copy");
    for(i = 0; i < dest->nmodes; ++i) {
        result = ptiCopyIndexVector(&dest->inds[i], &src->inds[i], 1);
        pti_CheckError(result, "SspTns Copy", NULL);
    }
    dest->stride = src->stride;
    result = ptiCopyMatrix(&dest->values, &src->values);
    pti_CheckError(result, "SspTns Copy", NULL);
    return 0;
}

/**
 * Release any memory the semi sparse tensor is holding
 * @param tsr the tensor to release
 */
void ptiFreeSemiSparseTensor(ptiSemiSparseTensor *tsr) {
    ptiIndex i;
    for(i = 0; i < tsr->nmodes; ++i) {
        ptiFreeIndexVector(&tsr->inds[i]);
    }
    free(tsr->ndims);
    free(tsr->inds);
    ptiFreeMatrix(&tsr->values);
}



/**
 * Create a new semi sparse tensor
 * @param tsr    a pointer to an uninitialized semi sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param mode   the mode which will be stored in dense format
 * @param ndims  the dimension of each mode the tensor will have
 */
int ptiNewSemiSparseTensorGeneral(ptiSemiSparseTensorGeneral *tsr, ptiIndex nmodes, const ptiIndex ndims[], ptiIndex ndmodes, const ptiIndex dmodes[]) {
    ptiIndex i;
    int result;
    if(nmodes < 2) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "SspTns New", "nmodes < 2");
    }
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    pti_CheckOSError(!tsr->ndims, "SspTns New");
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);

    tsr->ndmodes = ndmodes;
    tsr->dmodes = malloc(nmodes * sizeof *tsr->dmodes);
    pti_CheckOSError(!tsr->dmodes, "SspTns New");
    memcpy(tsr->dmodes, dmodes, nmodes * sizeof *tsr->dmodes);

    ptiIndex nsmodes = nmodes - ndmodes;
    tsr->nnz = 0;
    tsr->inds = malloc(nsmodes * sizeof *tsr->inds);
    pti_CheckOSError(!tsr->inds, "SspTns New");
    for(i = 0; i < nsmodes; ++i) {
        result = ptiNewIndexVector(&tsr->inds[i], 0, 0);
        pti_CheckError(result, "SspTns New", NULL);
    }
    tsr->strides = malloc(ndmodes * sizeof *tsr->strides);
    for(i = 0; i < ndmodes; ++i) {
        tsr->strides[i] = ((ndims[dmodes[i]]-1)/8+1)*8;
    }
    result = ptiNewMatrix(&tsr->values, 0, tsr->strides[0]);
    pti_CheckError(result, "SspTns New", NULL);
    return 0;
}

/**
 * Release any memory the semi sparse tensor is holding
 * @param tsr the tensor to release
 */
void ptiFreeSemiSparseTensorGeneral(ptiSemiSparseTensorGeneral *tsr) {
    ptiIndex i;
    for(i = 0; i < (tsr->nmodes - tsr->ndmodes); ++i) {
        ptiFreeIndexVector(&tsr->inds[i]);
    }
    free(tsr->ndims);
    free(tsr->dmodes);
    free(tsr->strides);
    free(tsr->inds);
    ptiFreeMatrix(&tsr->values);
}