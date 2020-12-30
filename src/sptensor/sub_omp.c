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

/* TODO: bug. */
int ptiSparseTensorSubOMP(ptiSparseTensor *Y, ptiSparseTensor *X, int const nthreads) {
    /* Ensure X and Y are in same shape */
    if(Y->nmodes != X->nmodes) {
        pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP SpTns Sub", "shape mismatch");
    }
    for(ptiIndex i = 0; i < X->nmodes; ++i) {
        if(Y->ndims[i] != X->ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "OMP SpTns Sub", "shape mismatch");
        }
    }

    /* Determine partationing strategy. */
    ptiNnzIndex * dist_nnzs_X = (ptiNnzIndex*)malloc(nthreads*sizeof(ptiNnzIndex));
    ptiNnzIndex * dist_nnzs_Y = (ptiNnzIndex*)malloc(nthreads*sizeof(ptiNnzIndex));
    ptiIndex * dist_nrows_Y = (ptiIndex*)malloc(nthreads*sizeof(ptiIndex));

    pti_DistSparseTensor(Y, nthreads, dist_nnzs_Y, dist_nrows_Y);
    pti_DistSparseTensorFixed(X, nthreads, dist_nnzs_X, dist_nnzs_Y);
    free(dist_nrows_Y);

    printf("dist_nnzs_Y:\n");
    for(int i=0; i<nthreads; ++i) {
        printf("%" HIPARTI_PRI_NNZ_INDEX " ", dist_nnzs_Y[i]);
    }
    printf("\n");
    printf("dist_nnzs_X:\n");
    for(int i=0; i<nthreads; ++i) {
        printf("%" HIPARTI_PRI_NNZ_INDEX " ", dist_nnzs_X[i]);
    }
    printf("\n");
    fflush(stdout);


    /* Build a private arrays to append values. */
    ptiNnzIndex nnz_gap = llabs((long long) Y->nnz - (long long) X->nnz);
    ptiNnzIndex increase_size = 0;
    if(nnz_gap == 0) increase_size = 10;
    else increase_size = nnz_gap;

    ptiIndexVector **local_inds = (ptiIndexVector**)malloc(nthreads* sizeof *local_inds);
    for(int k=0; k<nthreads; ++k) {
        local_inds[k] = (ptiIndexVector*)malloc(Y->nmodes* sizeof *(local_inds[k]));
        for(ptiIndex m=0; m<Y->nmodes; ++m) {
            ptiNewIndexVector(&(local_inds[k][m]), 0, increase_size);
        }
    }

    ptiValueVector *local_vals = (ptiValueVector*)malloc(nthreads* sizeof *local_vals);
    for(int k=0; k<nthreads; ++k) {
        ptiNewValueVector(&(local_vals[k]), 0, increase_size);
    }


    /* Add elements one by one, assume indices are ordered */
    ptiNnzIndex Ynnz = 0;
    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);
    #pragma omp parallel reduction(+:Ynnz)
    {
        int tid = omp_get_thread_num();
        ptiNnzIndex i=0, j=0;
        Ynnz = dist_nnzs_Y[tid];
        while(i < dist_nnzs_X[tid] && j < dist_nnzs_Y[tid]) {
            int compare = pti_SparseTensorCompareIndices(X, i, Y, j);
            if(compare > 0) {
                ++j;
            } else if(compare < 0) {
                ptiIndex mode;
                int result;
                for(mode = 0; mode < X->nmodes; ++mode) {
                    result = ptiAppendIndexVector(&(local_inds[tid][mode]), X->inds[mode].data[i]);
                    pti_CheckOmpError(result, "OMP SpTns Sub", NULL);
                }
                result = ptiAppendValueVector(&(local_vals[tid]), -X->values.data[i]);
                pti_CheckOmpError(result, "OMP SpTns Sub", NULL);
                ++Ynnz;
                ++i;
            } else {
                Y->values.data[j] -= X->values.data[i];
                ++i;
                ++j;
            }
        }
        /* Append remaining elements of X to Y */
        while(i < dist_nnzs_X[tid]) {
            ptiIndex mode;
            int result;
            for(mode = 0; mode < X->nmodes; ++mode) {
                result = ptiAppendIndexVector(&(local_inds[tid][mode]), X->inds[mode].data[i]);
                pti_CheckOmpError(result, "OMP SpTns Sub", NULL);
            }
            result = ptiAppendValueVector(&(local_vals[tid]), -X->values.data[i]);
            pti_CheckOmpError(result, "OMP SpTns Sub", NULL);
            ++Ynnz;
            ++i;
        }

    }

    /* Append all the local arrays to Y. */
    for(int k=0; k<nthreads; ++k) {
        for(ptiIndex m=0; m<Y->nmodes; ++m) {
            ptiAppendIndexVectorWithVector(&(Y->inds[m]), &(local_inds[k][m]));
        }
        ptiAppendValueVectorWithVector(&(Y->values), &(local_vals[k]));
    }

    for(int k=0; k<nthreads; ++k) {
        for(ptiIndex m=0; m<Y->nmodes; ++m) {
            ptiFreeIndexVector(&(local_inds[k][m]));
        }
        free(local_inds[k]);
        ptiFreeValueVector(&(local_vals[k]));
    }
    free(local_inds);
    free(local_vals);
    free(dist_nnzs_X);
    free(dist_nnzs_Y);

    /* Check whether elements become zero after adding.
       If so, fill the gap with the [nnz-1]'th element.
    */
    pti_SparseTensorCollectZeros(Y);
    /* Sort the indices */
    ptiSparseTensorSortIndex(Y, 1, nthreads);

    return 0;
}
