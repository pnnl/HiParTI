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
#include "sptensor.h"
#include <stdlib.h>
#include <string.h>

/**
 * Create a new sparse tensor
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int ptiNewSparseTensor(ptiSparseTensor *tsr, ptiIndex nmodes, const ptiIndex ndims[]) {
    ptiIndex i;
    int result;
    tsr->nmodes = nmodes;
    tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    for(i = 0; i < nmodes; ++i) {
        tsr->sortorder[i] = i;
    }
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    pti_CheckOSError(!tsr->ndims, "SpTns New");
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->nnz = 0;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    pti_CheckOSError(!tsr->inds, "SpTns New");
    for(i = 0; i < nmodes; ++i) {
        result = ptiNewIndexVector(&tsr->inds[i], 0, 0);
        pti_CheckError(result, "SpTns New", NULL);
    }
    result = ptiNewValueVector(&tsr->values, 0, 0);
    pti_CheckError(result, "SpTns New", NULL);
    return 0;
}

/**
 * Copy a sparse tensor
 * @param[out] dest a pointer to an uninitialized sparse tensor
 * @param[in]  src  a pointer to a valid sparse tensor
 */
int ptiCopySparseTensor(ptiSparseTensor *dest, const ptiSparseTensor *src, int const nt) {
    ptiIndex i;
    int result;
    dest->nmodes = src->nmodes;
    dest->sortorder = malloc(src->nmodes * sizeof src->sortorder[0]);
    memcpy(dest->sortorder, src->sortorder, src->nmodes * sizeof src->sortorder[0]);
    dest->ndims = malloc(dest->nmodes * sizeof *dest->ndims);
    pti_CheckOSError(!dest->ndims, "SpTns Copy");
    memcpy(dest->ndims, src->ndims, src->nmodes * sizeof *src->ndims);
    dest->nnz = src->nnz;
    dest->inds = malloc(dest->nmodes * sizeof *dest->inds);
    pti_CheckOSError(!dest->inds, "SpTns Copy");
    for(i = 0; i < dest->nmodes; ++i) {
        result = ptiCopyIndexVector(&dest->inds[i], &src->inds[i], nt);
        pti_CheckError(result, "SpTns Copy", NULL);
    }
    result = ptiCopyValueVector(&dest->values, &src->values, nt);
    pti_CheckError(result, "SpTns Copy", NULL);
    return 0;
}

/**
 * Release any memory the sparse tensor is holding
 * @param tsr the tensor to release
 */
void ptiFreeSparseTensor(ptiSparseTensor *tsr) {
    ptiIndex i;
    for(i = 0; i < tsr->nmodes; ++i) {
        ptiFreeIndexVector(&tsr->inds[i]);
    }
    free(tsr->sortorder);
    free(tsr->ndims);
    free(tsr->inds);
    ptiFreeValueVector(&tsr->values);
    tsr->nmodes = 0;
    tsr->nmodes = 0;
}


double SparseTensorFrobeniusNormSquared(ptiSparseTensor const * const ptien)
{
  double norm = 0;
  ptiValue const * const restrict vals = ptien->values.data;
  
#ifdef HIPARTI_USE_OPENMP
  #pragma omp parallel for reduction(+:norm)
#endif
  for(ptiNnzIndex n=0; n < ptien->nnz; ++n) {
    norm += vals[n] * vals[n];
  }
  return norm;
}


int pti_DistSparseTensor(ptiSparseTensor * tsr,
    int const nthreads,
    ptiNnzIndex * const dist_nnzs,
    ptiIndex * dist_nrows) {

    ptiNnzIndex global_nnz = tsr->nnz;
    ptiNnzIndex aver_nnz = global_nnz / nthreads;
    memset(dist_nnzs, 0, nthreads*sizeof(ptiNnzIndex));
    memset(dist_nrows, 0, nthreads*sizeof(ptiIndex));

    ptiSparseTensorSortIndex(tsr, 0, 1);
    ptiIndex * ind0 = tsr->inds[0].data;

    int ti = 0;
    dist_nnzs[0] = 1;
    dist_nrows[0] = 1;
    for(ptiNnzIndex x=1; x<global_nnz; ++x) {
        if(ind0[x] == ind0[x-1]) {
            ++ dist_nnzs[ti];
        } else if (ind0[x] > ind0[x-1]) {
            if(dist_nnzs[ti] < aver_nnz || ti == nthreads-1) {
                ++ dist_nnzs[ti];
                ++ dist_nrows[ti];
            } else {
                ++ ti;
                ++ dist_nnzs[ti];
                ++ dist_nrows[ti];
            }
        } else {
            pti_CheckError(PTIERR_VALUE_ERROR, "SpTns Dist", "tensor unsorted on mode-0");
        }
    }

    return 0;

}


int pti_DistSparseTensorFixed(ptiSparseTensor * tsr,
    int const nthreads,
    ptiNnzIndex * const dist_nnzs,
    ptiNnzIndex * dist_nrows) {

    ptiNnzIndex global_nnz = tsr->nnz;
    ptiNnzIndex aver_nnz = global_nnz / nthreads;
    memset(dist_nnzs, 0, nthreads*sizeof(ptiNnzIndex));

    ptiSparseTensorSortIndex(tsr, 0, 1);
    ptiIndex * ind0 = tsr->inds[0].data;

    int ti = 0;
    dist_nnzs[0] = 1;
    for(ptiNnzIndex x=1; x<global_nnz; ++x) {
        if(ind0[x] == ind0[x-1]) {
            ++ dist_nnzs[ti];
        } else if (ind0[x] > ind0[x-1]) {
            if(dist_nnzs[ti] < aver_nnz || ti == nthreads-1) {
                ++ dist_nnzs[ti];
            } else {
                ++ ti;
                ++ dist_nnzs[ti];
            }
        } else {
            pti_CheckError(PTIERR_VALUE_ERROR, "SpTns Dist", "tensor unsorted on mode-0");
        }
    }

    return 0;
}


int pti_SparseTensorDumpAllSplits(const pti_SplitResult * splits, ptiIndex const nsplits, FILE *fp) {
    ptiIndex i = 0;
    for(i=0; i<nsplits; ++i) {
    // while(split_i) {
        const pti_SplitResult *split_i = splits + i;
        printf("Printing split #%"HIPARTI_PRI_INDEX " of %"HIPARTI_PRI_INDEX "lu:\n", i + 1, nsplits);
        printf("Index: \n");
        ptiDumpIndexArray(split_i->inds_low, split_i->tensor.nmodes, fp);
        printf(" .. \n");
        ptiDumpIndexArray(split_i->inds_high, split_i->tensor.nmodes, fp);
        ptiDumpSparseTensor(&split_i->tensor, 0, fp);
        printf("\n");
        fflush(fp);
        // ++ i;
        // split_i = split_i->next;
    }
    return 0;
}


/**
 * Shuffle all indices.
 *
 * @param[in] tsr tensor to be shuffled
 * @param[out] map_inds is the renumbering mapping
 *
 */
void ptiSparseTensorShuffleIndices(ptiSparseTensor *tsr, ptiIndex ** map_inds) {
    /* Renumber nonzero elements */
    ptiIndex tmp_ind;
    for(ptiNnzIndex z = 0; z < tsr->nnz; ++z) {
        for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
            tmp_ind = tsr->inds[m].data[z];
            tsr->inds[m].data[z] = map_inds[m][tmp_ind];
        }
    }
    
}


/**
 * Inverse-Shuffle all indices.
 *
 * @param[in] tsr tensor to be shuffled
 * @param[out] map_inds is the renumbering mapping
 *
 */
void ptiSparseTensorInvMap(ptiSparseTensor *tsr, ptiIndex ** map_inds)
{
    ptiIndex ** tmp_map_inds = (ptiIndex **)malloc(tsr->nmodes * sizeof(ptiIndex**));
    pti_CheckOSError(!tmp_map_inds, "ptiSparseTensorInvMap");
    for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
        tmp_map_inds[m] = (ptiIndex *)malloc(tsr->ndims[m] * sizeof (ptiIndex));
        pti_CheckError(!tmp_map_inds[m], "ptiSparseTensorInvMap", NULL);
    }

    ptiIndex tmp_ind;
    for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
        ptiIndex loc = 0;
        for(ptiNnzIndex i = 0; i < tsr->ndims[m]; ++i) {
            tmp_ind = map_inds[m][i];
            tmp_map_inds[m][tmp_ind] = loc;
            ++ loc;
        }
    }

    for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
        for(ptiNnzIndex i = 0; i < tsr->ndims[m]; ++i) {
            map_inds[m][i] = tmp_map_inds[m][i];
        }
    }

    for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
        free(tmp_map_inds[m]);
    }
    free(tmp_map_inds);
    
}


/**
 * Shuffle tensor modes (only changing pointers).
 *
 * @param[in] tsr tensor to be shuffled
 * @param[out] mode_order is the new order of modes
 *
 */
void ptiSparseTensorShuffleModes(ptiSparseTensor *tsr, ptiIndex * mode_order)
{
    /// Use temporary space to save the original indices and ndims, ensuring the correct shuffling.
    ptiIndex ** tmp_ind = (ptiIndex **)malloc(tsr->nmodes * sizeof(ptiIndex *));
    ptiIndex * tmp_ndims = (ptiIndex*)malloc(tsr->nmodes * sizeof(ptiIndex));
    for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
        tmp_ind[m] = tsr->inds[m].data;
        tmp_ndims[m] = tsr->ndims[m];
    }
    
    for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
        ptiIndex pm = mode_order[m];
        
        tsr->ndims[m] = tmp_ndims[pm];
        tsr->inds[m].data = tmp_ind[m];

        tsr->sortorder[m] = m;
    }
    
    free(tmp_ind);
    free(tmp_ndims);
}