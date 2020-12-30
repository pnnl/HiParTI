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

/**
  * SpMM, C = AB, A is sparse, B, C are dense
  */
int ptiSparseMatrixMulMatrix(ptiMatrix * C, const ptiSparseMatrix *spA, ptiMatrix * B)
{
    for(ptiNnzIndex z = 0; z < spA->nnz; ++z) {
        ptiIndex row = spA->rowind.data[z]; // C[row,:]
        ptiIndex col = spA->colind.data[z]; // B[col,:]
        ptiValue val = spA->values.data[z];
        for(ptiNnzIndex c = 0; c < B->ncols; ++c) {
            C->values[row * C->stride + c] += val * B->values[col * B->stride + c];
        }
    }
    return 0;
}

#ifdef HIPARTI_USE_OPENMP
int ptiOmpSparseMatrixMulMatrix(ptiMatrix * C, const ptiSparseMatrix *spA, ptiMatrix * B)
{
    #pragma omp parallel for // schedule(static)
    for(ptiNnzIndex z = 0; z < spA->nnz; ++z) {
        ptiIndex row = spA->rowind.data[z]; // C[row,:]
        ptiIndex col = spA->colind.data[z]; // B[col,:]
        ptiValue val = spA->values.data[z];
        ptiValue * restrict cval_row = C->values + row * C->stride;
        for(ptiIndex c = 0; c < B->ncols; ++c) {
            #pragma omp atomic update
            cval_row[c] += val * B->values[col * B->stride + c];
            // C->values[row * C->stride+c] += val * B->values[col * B->stride + c];    // slower
        }
    }
    return 0;
}


int ptiOmpSparseMatrixMulMatrix_Reduce(ptiMatrix * C, ptiMatrix * Cbufs, const ptiSparseMatrix *spA, ptiMatrix * B)
{
    int nthreads;
    #pragma omp parallel
        nthreads = omp_get_num_threads();

    #pragma omp parallel for // schedule(static)
    for(ptiNnzIndex z = 0; z < spA->nnz; ++z) {
        int tid = omp_get_thread_num();
        ptiIndex row = spA->rowind.data[z]; // C[row,:]
        ptiIndex col = spA->colind.data[z]; // B[col,:]
        ptiValue val = spA->values.data[z];
        #pragma omp simd
        for(ptiIndex c = 0; c < B->ncols; ++c) {
            Cbufs[tid].values[row * C->stride + c] += val * B->values[col * B->stride + c];    // slower
        }
    }

    /* Reduction */
    #pragma omp parallel for schedule(static)
    for(ptiIndex r=0; r<C->nrows; ++r) {
        for(int t=0; t<nthreads; ++t) {
            #pragma omp simd
            for(ptiIndex c = 0; c < C->ncols; ++c) {
                C->values[r * C->stride + c] += Cbufs[t].values[r * C->stride + c];
            }   
        }
    }

    return 0;
}
#endif
