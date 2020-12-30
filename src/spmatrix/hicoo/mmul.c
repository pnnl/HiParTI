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
#include <stdio.h>

int ptiSparseMatrixMulMatrixHiCOO(ptiMatrix * C, const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B)
{
    ptiNnzIndex nb = himtx->bptr.len - 1;
    ptiElementIndex sb_bits = himtx->sb_bits;

    /* No loop for kernels */
    for(ptiNnzIndex b = 0; b < nb; ++b) {   // Loop blocks
        ptiNnzIndex bptr_begin = himtx->bptr.data[b];
        ptiNnzIndex bptr_end = himtx->bptr.data[b+1];
        ptiValue * restrict blocked_Cvals = C->values + (himtx->bindI.data[b] << sb_bits) * C->stride;
        ptiValue * restrict blocked_Bvals = B->values + (himtx->bindJ.data[b] << sb_bits) * B->stride;

        for(ptiNnzIndex z=bptr_begin; z<bptr_end; ++z) {   // Loop entries
            ptiElementIndex ei = himtx->eindI.data[z];
            ptiElementIndex ej = himtx->eindJ.data[z];
            ptiValue val = himtx->values.data[z];
            for(ptiNnzIndex c = 0; c < B->ncols; ++c) {
                blocked_Cvals[ei * C->stride + c] += val * blocked_Bvals[ej * B->stride + c];
            }
        }
    }

    return 0;
}

#ifdef HIPARTI_USE_OPENMP
int ptiOmpSparseMatrixMulMatrixHiCOO(ptiMatrix * C, const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B)
{
    ptiNnzIndex nb = himtx->bptr.len - 1;
    ptiElementIndex sb_bits = himtx->sb_bits;

    /* No loop for kernels */
    #pragma omp parallel for
    for(ptiNnzIndex b = 0; b < nb; ++b) {   // Loop blocks
        ptiNnzIndex bptr_begin = himtx->bptr.data[b];
        ptiNnzIndex bptr_end = himtx->bptr.data[b+1];
        ptiValue * restrict blocked_Cvals = C->values + (himtx->bindI.data[b] << sb_bits) * C->stride;
        ptiValue * restrict blocked_Bvals = B->values + (himtx->bindJ.data[b] << sb_bits) * B->stride;

        for(ptiNnzIndex z=bptr_begin; z<bptr_end; ++z) {   // Loop entries
            ptiElementIndex ei = himtx->eindI.data[z];
            ptiElementIndex ej = himtx->eindJ.data[z];
            ptiValue val = himtx->values.data[z];
            ptiValue * restrict blocked_cval_row = blocked_Cvals + ei * C->stride;
            for(ptiNnzIndex c = 0; c < B->ncols; ++c) {
                #pragma omp atomic update
                blocked_cval_row[c] += val * blocked_Bvals[ej * B->stride + c];
            }
        }
    }

    return 0;
}


int ptiOmpSparseMatrixMulMatrixHiCOO_Schedule(ptiMatrix * C, const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B)
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
                ptiValue * restrict blocked_Cvals = C->values + (himtx->bindI.data[b] << sb_bits) * C->stride;
                ptiValue * restrict blocked_Bvals = B->values + (himtx->bindJ.data[b] << sb_bits) * B->stride;

                for(ptiNnzIndex z=bptr_begin; z<bptr_end; ++z) {   // Loop entries
                    ptiElementIndex ei = himtx->eindI.data[z];
                    ptiElementIndex ej = himtx->eindJ.data[z];
                    ptiValue val = himtx->values.data[z];

                    #pragma omp simd
                    for(ptiNnzIndex c = 0; c < B->ncols; ++c) {
                        blocked_Cvals[ei * C->stride + c] += val * blocked_Bvals[ej * B->stride + c];
                    }
                }
            }   // End loop b
        }   // End loop k
    }   // End loop i

    return 0;
}


int ptiOmpSparseMatrixMulMatrixHiCOO_Schedule_Reduce(ptiMatrix * C, ptiMatrix * Cbufs, const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B)
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
        int tid = omp_get_thread_num();

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
                ptiValue * restrict blocked_Cvals = Cbufs[tid].values + (himtx->bindI.data[b] << sb_bits) * C->stride;
                ptiValue * restrict blocked_Bvals = B->values + (himtx->bindJ.data[b] << sb_bits) * B->stride;

                for(ptiNnzIndex z=bptr_begin; z<bptr_end; ++z) {   // Loop entries
                    ptiElementIndex ei = himtx->eindI.data[z];
                    ptiElementIndex ej = himtx->eindJ.data[z];
                    ptiValue val = himtx->values.data[z];

                    #pragma omp simd
                    for(ptiNnzIndex c = 0; c < B->ncols; ++c) {
                        blocked_Cvals[ei * C->stride + c] += val * blocked_Bvals[ej * B->stride + c];
                    }
                }
            }   // End loop b
        }   // End loop k
    }   // End loop i

    /* Reduction */
    #pragma omp parallel for schedule(static)
    for(ptiIndex r=0; r<C->nrows; ++r) {
        for(int t=0; t<nthreads; ++t) {
            #pragma omp simd
            for(ptiIndex c=0; c<C->ncols; ++c) {
                C->values[r * C->stride + c] += Cbufs[t].values[r * C->stride + c];
            }
        }
    }

    return 0;
}

#endif