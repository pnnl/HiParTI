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

int ptiSparseMatrixMulVectorHiCOO(ptiValueVector * y, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x)
{
    ptiNnzIndex nb = himtx->bptr.len - 1;
    ptiElementIndex sb_bits = himtx->sb_bits;

    for(ptiNnzIndex b = 0; b < nb; ++b) {   // Loop blocks
        ptiNnzIndex bptr_begin = himtx->bptr.data[b];
        ptiNnzIndex bptr_end = himtx->bptr.data[b+1];
        ptiValue * restrict blocked_yvals = y->data + (himtx->bindI.data[b] << sb_bits);
        ptiValue * restrict blocked_xvals = x->data + (himtx->bindJ.data[b] << sb_bits);

        for(ptiNnzIndex z=bptr_begin; z<bptr_end; ++z) {   // Loop entries
            ptiElementIndex ei = himtx->eindI.data[z];
            ptiElementIndex ej = himtx->eindJ.data[z];
            blocked_yvals[ei] += himtx->values.data[z] * blocked_xvals[ej];            
        }
    }

    return 0;
}

#ifdef HIPARTI_USE_OPENMP
int ptiOmpSparseMatrixMulVectorHiCOO(ptiValueVector * y, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x)
{
    ptiNnzIndex nb = himtx->bptr.len - 1;
    ptiElementIndex sb_bits = himtx->sb_bits;

    /* No loop for kernels */
    #pragma omp parallel for
    for(ptiNnzIndex b = 0; b < nb; ++b) {   // Loop blocks
        ptiNnzIndex bptr_begin = himtx->bptr.data[b];
        ptiNnzIndex bptr_end = himtx->bptr.data[b+1];
        ptiValue * restrict blocked_yvals = y->data + (himtx->bindI.data[b] << sb_bits);
        ptiValue * restrict blocked_xvals = x->data + (himtx->bindJ.data[b] << sb_bits);

        for(ptiNnzIndex z=bptr_begin; z<bptr_end; ++z) {   // Loop entries
            ptiElementIndex ei = himtx->eindI.data[z];
            ptiElementIndex ej = himtx->eindJ.data[z];
            #pragma omp atomic update
            blocked_yvals[ei] += himtx->values.data[z] * blocked_xvals[ej];            
        }
    }

    return 0;
}

int ptiOmpSparseMatrixMulVectorHiCOO_Schedule(ptiValueVector * y, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x)
{
    ptiElementIndex sb_bits = himtx->sb_bits;
    ptiIndex sk = (ptiIndex)pow(2, himtx->sk_bits);
    ptiIndex num_kernel_dim = (himtx->nrows + sk - 1) / sk;

    /* Loop parallel iterations */
    for(ptiIndex i=0; i<himtx->nkiters; ++i) {  // Loop schedule columns

        #pragma omp parallel for schedule(dynamic, 1) 
        for(ptiIndex k=0; k<num_kernel_dim; ++k) {  // Loop schedule rows
            if(i >= himtx->kschr[k].len) {
                continue;
            }

            ptiIndex kptr_loc = himtx->kschr[k].data[i];
            ptiNnzIndex kptr_begin = himtx->kptr.data[kptr_loc];
            ptiNnzIndex kptr_end = himtx->kptr.data[kptr_loc+1];

            /* Loop blocks in a kernel */
            for(ptiNnzIndex b = kptr_begin; b < kptr_end; ++b) {   // Loop blocks
                ptiNnzIndex bptr_begin = himtx->bptr.data[b];
                ptiNnzIndex bptr_end = himtx->bptr.data[b+1];
                ptiValue * restrict blocked_yvals = y->data + (himtx->bindI.data[b] << sb_bits);
                ptiValue * restrict blocked_xvals = x->data + (himtx->bindJ.data[b] << sb_bits);

                for(ptiNnzIndex z=bptr_begin; z<bptr_end; ++z) {   // Loop entries
                    ptiElementIndex ei = himtx->eindI.data[z];
                    ptiElementIndex ej = himtx->eindJ.data[z];
                    blocked_yvals[ei] += himtx->values.data[z] * blocked_xvals[ej];            
                }
            }
        }
    }

    return 0;
}

int ptiOmpSparseMatrixMulVectorHiCOOReduce(ptiValueVector * y, const ptiSparseMatrixHiCOO *mtx, ptiValueVector * x){
	ptiValueVector * ybufs;
	int nthreads =1;
	#pragma omp parallel
	nthreads=omp_get_num_threads();
	ybufs = (ptiValueVector *) malloc(nthreads * sizeof(ptiValueVector));
        for(int t=0; t<nthreads; ++t) {
            ptiNewValueVector(&ybufs[t], mtx->nrows, mtx->nrows);
            ptiConstantValueVector(&ybufs[t], 0);
	}
	ptiOmpSparseMatrixMulVectorHiCOO_Schedule_Reduce(y, ybufs, mtx, x);
	for(int t=0; t<nthreads; ++t) {
            ptiFreeValueVector(&ybufs[t]);
        }
        free(ybufs);
	return 0;
}

int ptiOmpSparseMatrixMulVectorHiCOO_Schedule_Reduce(ptiValueVector * y, ptiValueVector * ybufs, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x)
{
    ptiElementIndex sb_bits = himtx->sb_bits;
    ptiIndex sk = (ptiIndex)pow(2, himtx->sk_bits);
    ptiIndex num_kernel_dim = (himtx->nrows + sk - 1) / sk;
    int nthreads;
    #pragma omp parallel
        nthreads = omp_get_num_threads();

    /* Loop parallel iterations */
    #pragma omp parallel for schedule(dynamic, 1) 
    for(ptiIndex i=0; i<himtx->nkiters; ++i) {  // Loop schedule columns
        // int tid = omp_get_thread_num();

        for(ptiIndex k=0; k<num_kernel_dim; ++k) {  // Loop schedule rows
            if(i >= himtx->kschr[k].len) {
                continue;
            }

            ptiIndex kptr_loc = himtx->kschr[k].data[i];
            ptiNnzIndex kptr_begin = himtx->kptr.data[kptr_loc];
            ptiNnzIndex kptr_end = himtx->kptr.data[kptr_loc+1];

            /* Loop blocks in a kernel */
            for(ptiNnzIndex b = kptr_begin; b < kptr_end; ++b) {   // Loop blocks
                ptiNnzIndex bptr_begin = himtx->bptr.data[b];
                ptiNnzIndex bptr_end = himtx->bptr.data[b+1];
                ptiValue * restrict blocked_yvals = y->data + (himtx->bindI.data[b] << sb_bits);
                ptiValue * restrict blocked_xvals = x->data + (himtx->bindJ.data[b] << sb_bits);

                for(ptiNnzIndex z=bptr_begin; z<bptr_end; ++z) {   // Loop entries
                    ptiElementIndex ei = himtx->eindI.data[z];
                    ptiElementIndex ej = himtx->eindJ.data[z];
                    blocked_yvals[ei] += himtx->values.data[z] * blocked_xvals[ej];            
                }
            }   // End loop b
        }   // End loop k
    }   // End loop i

    /* Reduction */
    #pragma omp parallel for schedule(static)
    for(ptiIndex r=0; r<himtx->nrows; ++r) {
        for(int t=0; t<nthreads; ++t) {
            y->data[r] += ybufs[t].data[r];
        }
    }

    return 0;
}
#endif