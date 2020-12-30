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

/**
  * SpMV, y = Ax
  */
int ptiSparseMatrixMulVectorCSR(ptiValueVector * y, ptiSparseMatrixCSR *csrmtx, ptiValueVector * x)
{
#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for
#endif
    for(ptiIndex i = 0; i < csrmtx->nrows; ++i) {
        for(ptiNnzIndex z = csrmtx->rowptr.data[i]; z < csrmtx->rowptr.data[i+1]; ++z) {
            ptiIndex col = csrmtx->colind.data[z];
            y->data[i] += csrmtx->values.data[z] * x->data[col];
        }
    }
    return 0;
}

#ifdef HIPARTI_USE_OPENMP
int ptiOmpSparseMatrixMulVectorCSR(ptiValueVector * y, ptiSparseMatrixCSR *csrmtx, ptiValueVector * x)
{
    #pragma omp parallel for
    for(ptiIndex i = 0; i < csrmtx->nrows; ++i) {
        for(ptiNnzIndex z = csrmtx->rowptr.data[i]; z < csrmtx->rowptr.data[i+1]; ++z) {
            ptiIndex col = csrmtx->colind.data[z];
            y->data[i] += csrmtx->values.data[z] * x->data[col];
        }
    }
    return 0;
}
int ptiOmpSparseMatrixMulVectorCSRReduce(ptiValueVector * y, const ptiSparseMatrixCSR *mtx, ptiValueVector * x){
	ptiValueVector * ybufs;
	int nthreads =1;
	#pragma omp parallel
	nthreads=omp_get_num_threads();

	ybufs = (ptiValueVector *) malloc(nthreads * sizeof(ptiValueVector));
        for(int t=0; t<nthreads; ++t) {
            ptiNewValueVector(&ybufs[t], mtx->nrows, mtx->nrows);
            ptiConstantValueVector(&ybufs[t], 0);
        }
	ptiOmpSparseMatrixMulVectorCSR_Reduce(y, ybufs, mtx, x);
	for(int t=0; t<nthreads; ++t) {
            ptiFreeValueVector(&ybufs[t]);
        }
        free(ybufs);
	return 0;
}

int ptiOmpSparseMatrixMulVectorCSR_Reduce(ptiValueVector *y, ptiValueVector * ybufs, const ptiSparseMatrixCSR *csrmtx, ptiValueVector * x)
{
	int nthreads;
	#pragma omp parallel
		nthreads=omp_get_num_threads();

	#pragma omp parallel for // schedule(static)
	for(ptiIndex i = 0; i < csrmtx->nrows; ++i) {
        	for(ptiNnzIndex z = csrmtx->rowptr.data[i]; z < csrmtx->rowptr.data[i+1]; ++z) {
			int tid = omp_get_thread_num();
           		ptiIndex col = csrmtx->colind.data[z];
            		ybufs[tid].data[i] += csrmtx->values.data[z] * x->data[col];
       		}
    	}

	/*Reduction*/
	#pragma omp parallel for schedule(static)
	for(ptiIndex r=0; r<y->len; ++r){
		for (int t=0; t<nthreads; ++t){
			y->data[r] +=ybufs[t].data[r];
		}
	}

	return 0;
}

#endif
