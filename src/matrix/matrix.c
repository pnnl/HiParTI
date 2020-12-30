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
#include <time.h>
#include <math.h>

/**
 * Initialize a new dense matrix
 *
 * @param mtx   a valid pointer to an uninitialized ptiMatrix variable
 * @param nrows the number of rows
 * @param ncols the number of columns
 *
 * The memory layout of this dense matrix is a flat 2D array, with `ncols`
 * rounded up to multiples of 8
 */
int ptiNewMatrix(ptiMatrix *mtx, ptiIndex const nrows, ptiIndex const ncols) {
    mtx->nrows = nrows;
    mtx->ncols = ncols;
    mtx->cap = nrows != 0 ? nrows : 1;
    mtx->stride = ((ncols-1)/8+1)*8;
#ifdef _ISOC11_SOURCE
    mtx->values = aligned_alloc(8 * sizeof (ptiValue), mtx->cap * mtx->stride * sizeof (ptiValue));
#elif _POSIX_C_SOURCE >= 200112L
    {
        int result = posix_memalign((void **) &mtx->values, 8 * sizeof (ptiValue), mtx->cap * mtx->stride * sizeof (ptiValue));
        if(result != 0) {
            mtx->values = NULL;
        }
    }
#else
    mtx->values = malloc(mtx->cap * mtx->stride * sizeof (ptiValue));
#endif
    pti_CheckOSError(!mtx->values, "Mtx New");
    return 0;
}

/**
 * Build a matrix with random number
 *
 * @param mtx   a pointer to an uninitialized matrix
 * @param nrows fill the specified number of rows
 * @param ncols fill the specified number of columns
 *
 * The matrix is filled with uniform distributed pseudorandom number in [0, 1]
 * The random number will have a precision of 31 bits out of 51 bits
 */
int ptiRandomizeMatrix(ptiMatrix *mtx) {
  srand(time(NULL));
  for(ptiIndex i=0; i<mtx->nrows; ++i)
    for(ptiIndex j=0; j<mtx->ncols; ++j) {
      mtx->values[i * mtx->stride + j] = i + j + 1; //ptiRandomValue();
    }
  return 0;
}


/**
 * Fill an identity dense matrix
 *
 * @param mtx   a pointer to an uninitialized matrix
 * @param nrows fill the specified number of rows
 * @param ncols fill the specified number of columns
 *
 */
int ptiIdentityMatrix(ptiMatrix *mtx) {
  ptiIndex const nrows = mtx->nrows;
  ptiIndex const ncols = mtx->ncols;
  assert(nrows == ncols);
  for(ptiIndex i=0; i<nrows; ++i)
    for(ptiIndex j=0; j<ncols; ++j)
      mtx->values[i * mtx->stride + j] = 0;
  for(ptiIndex i=0; i<nrows; ++i)
    mtx->values[i * mtx->stride + i] = 1;

  return 0;
}


/**
 * Fill an existed dense matrix with a specified constant
 *
 * @param mtx   a pointer to a valid matrix
 * @param val   a given value constant
 *
 */
int ptiConstantMatrix(ptiMatrix *mtx, ptiValue const val) {
  for(ptiIndex i=0; i<mtx->nrows; ++i)
    for(ptiIndex j=0; j<mtx->ncols; ++j)
      mtx->values[i * mtx->stride + j] = val;
  return 0;
}


/**
 * Shuffle matrix row indices.
 *
 * @param[in] mtx matrix to be shuffled
 * @param[out] map_inds is the renumbering mapping 
 *
 */
void ptiMatrixInverseShuffleIndices(ptiMatrix *mtx, ptiIndex * mode_map_inds) {
    /* Renumber matrix rows */
    ptiIndex new_i;
    ptiValue * tmp_values = malloc(mtx->cap * mtx->stride * sizeof (ptiValue));

    for(ptiIndex i=0; i<mtx->nrows; ++i) {
        new_i = mode_map_inds[i];
        for(ptiIndex j=0; j<mtx->ncols; ++j) {
            tmp_values[i * mtx->stride + j] = mtx->values[new_i * mtx->stride + j];
        }
    }

    free(mtx->values);
    mtx->values = tmp_values;
}


/**
 * Copy a dense matrix to an uninitialized dense matrix
 *
 * @param dest a pointer to an uninitialized dense matrix
 * @param src  a pointer to an existing valid dense matrix
 *
 * The contents of `src` will be copied to `dest`.
 */
int ptiCopyMatrix(ptiMatrix *dest, const ptiMatrix *src) {
    int result = ptiNewMatrix(dest, src->nrows, src->ncols);
    pti_CheckError(result, "Mtx Copy", NULL);
    assert(dest->stride == src->stride);
    memcpy(dest->values, src->values, dest->nrows * dest->stride * sizeof (ptiValue));
    return 0;
}

/**
 * Add a row to the end of dense matrix
 *
 * @param mtx    a pointer to a valid matrix
 * @param values an array of data to be added
 */
int ptiAppendMatrix(ptiMatrix *mtx, const ptiValue values[]) {
    if(mtx->cap <= mtx->nrows) {
#ifndef MEMCHECK_MODE
        ptiIndex newcap = mtx->cap + mtx->cap/2;
#else
        ptiIndex newcap = mtx->nrows+1;
#endif
        ptiValue *newdata;
#ifdef _ISOC11_SOURCE
        newdata = aligned_alloc(8 * sizeof (ptiValue), newcap * mtx->stride * sizeof (ptiValue));
#elif _POSIX_C_SOURCE >= 200112L
        {
            int result = posix_memalign((void **) &newdata, 8 * sizeof (ptiValue), newcap * mtx->stride * sizeof (ptiValue));
            if(result != 0) {
                newdata = NULL;
            }
        }
#else
        newdata = malloc(newcap * mtx->stride * sizeof (ptiValue));
#endif
        pti_CheckOSError(!newdata, "Mtx Append");
        memcpy(newdata, mtx->values, mtx->nrows * mtx->stride * sizeof (ptiValue));
        free(mtx->values);
        mtx->cap = newcap;
        mtx->values = newdata;
    }
    if(values != NULL) {
        memcpy(&mtx->values[mtx->nrows * mtx->stride], values, mtx->ncols * sizeof (ptiValue));
    }
    ++ mtx->nrows;
    return 0;
}

/**
 * Modify the number of rows in a dense matrix
 *
 * @param mtx     a pointer to a valid matrix
 * @param new_nrows the new number of rows `mtx` will have
 */
int ptiResizeMatrix(ptiMatrix *mtx, ptiIndex const new_nrows) {
    ptiValue *newdata;
#ifdef _ISOC11_SOURCE
    newdata = aligned_alloc(8 * sizeof (ptiValue), new_nrows * mtx->stride * sizeof (ptiValue));
#elif _POSIX_C_SOURCE >= 200112L
    {
        int result = posix_memalign((void **) &newdata, 8 * sizeof (ptiValue), new_nrows * mtx->stride * sizeof (ptiValue));
        if(result != 0) {
            newdata = NULL;
        }
    }
#else
    newdata = malloc(new_nrows * mtx->stride * sizeof (ptiValue));
#endif
    pti_CheckOSError(!newdata, "Mtx Resize");
    memcpy(newdata, mtx->values, mtx->nrows * mtx->stride * sizeof (ptiValue));
    free(mtx->values);
    mtx->nrows = new_nrows;
    mtx->cap = new_nrows;
    mtx->values = newdata;
    return 0;
}

/**
 * Release the memory buffer a dense matrix is holding
 *
 * @param mtx a pointer to a valid matrix
 *
 * By using `ptiFreeMatrix`, a valid matrix would become uninitialized and
 * should not be used anymore prior to another initialization
 */
void ptiFreeMatrix(ptiMatrix *mtx) {
    free(mtx->values);
    mtx->nrows = 0;
    mtx->ncols = 0;
    mtx->cap = 0;
    mtx->stride = 0;
}


/**** ptiMatrix Operations ****/

int ptiMatrixDotMul(ptiMatrix const * A, ptiMatrix const * B, ptiMatrix const * C)
{
    ptiIndex nrows = A->nrows;
    ptiIndex ncols = A->ncols;
    ptiIndex stride = A->stride;
    assert(nrows == B->nrows && nrows == C->nrows);
    assert(ncols == B->ncols && ncols == C->ncols);
    assert(stride == B->stride && stride == C->stride);

    for(ptiIndex i=0; i < nrows; ++i) {
        for(ptiIndex j=0; j < ncols; ++j) {
            C->values[i*stride+j] = A->values[i*stride+j] * B->values[i*stride+j];
        }
    }

    return 0;
}


int ptiMatrixDotMulSeq(ptiIndex const mode, ptiIndex const nmodes, ptiMatrix ** mats)
{
    ptiIndex const nrows = mats[0]->nrows;
    ptiIndex const ncols = mats[0]->ncols;
    ptiIndex const stride = mats[0]->stride;

    for(ptiIndex m=1; m<nmodes+1; ++m) {
        assert(mats[m]->ncols == ncols);
        assert(mats[m]->nrows == nrows);
        assert(mats[m]->stride == stride);
    }

    ptiValue * ovals = mats[nmodes]->values;
#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for
#endif
    for(ptiIndex i=0; i < nrows; ++i) {
        for(ptiIndex j=0; j < ncols; ++j) {
            ovals[i * stride + j] = 1;
        }
    }

    for(ptiIndex m=1; m < nmodes; ++m) {
        ptiIndex const pm = (mode + m) % nmodes;
        ptiValue const * vals = mats[pm]->values;
#ifdef HIPARTI_USE_OPENMP
        #pragma omp parallel for
#endif
        for(ptiIndex i=0; i < nrows; ++i) {
            for(ptiIndex j=0; j < ncols; ++j) {
                ovals[i * stride + j] *= vals[i * stride + j];
            }
        }
    }
    
    return 0;
}


int ptiMatrixDotMulSeqCol(ptiIndex const mode, ptiIndex const nmodes, ptiMatrix ** mats)
{
    ptiIndex const nrows = mats[0]->nrows;
    ptiIndex const ncols = mats[0]->ncols;
    ptiIndex const stride = mats[0]->stride;
    // printf("stride: %lu\n", stride);
    for(ptiIndex m=1; m<nmodes+1; ++m) {
        assert(mats[m]->ncols == ncols);
        assert(mats[m]->nrows == nrows);
        assert(mats[m]->stride == stride);
    }

    ptiValue * ovals = mats[nmodes]->values;
#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for
#endif
    for(ptiIndex j=0; j < ncols; ++j) {
        for(ptiIndex i=0; i < nrows; ++i) {
            ovals[j * stride + i] = 1;
        }
    }


    for(ptiIndex m=1; m < nmodes; ++m) {
        ptiIndex const pm = (mode + m) % nmodes;
        ptiValue const * vals = mats[pm]->values;
#ifdef HIPARTI_USE_OPENMP
        #pragma omp parallel for
#endif
        for(ptiIndex j=0; j < ncols; ++j) {
            for(ptiIndex i=0; i < nrows; ++i) {
                ovals[j * stride + i] *= vals[j * stride + i];
            }
        }
    }
    
    return 0;
}


/* mats (aTa) only stores upper triangle elements. */
int ptiMatrixDotMulSeqTriangle(ptiIndex const mode, ptiIndex const nmodes, ptiMatrix ** mats)
{
    ptiIndex const nrows = mats[0]->nrows;
    ptiIndex const ncols = mats[0]->ncols;
    ptiIndex const stride = mats[0]->stride;
    for(ptiIndex m=1; m<nmodes+1; ++m) {
        assert(mats[m]->ncols == ncols);
        assert(mats[m]->nrows == nrows);
    }

    ptiValue * ovals = mats[nmodes]->values;
#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(ptiIndex i=0; i < nrows; ++i) {
        for(ptiIndex j=0; j < ncols; ++j) {
            ovals[j * stride + i] = 1.0;
        }
    }


    for(ptiIndex m=1; m < nmodes; ++m) {
        ptiIndex const pm = (mode + m) % nmodes;
        ptiValue const * vals = mats[pm]->values;
#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
        for(ptiIndex i=0; i < nrows; ++i) {
            for(ptiIndex j=i; j < ncols; ++j) {
                ovals[i * stride + j] *= vals[i * stride + j];
            }
        }
    }

    /* Copy upper triangle to lower part */
#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(ptiIndex i=0; i < nrows; ++i) {
        for(ptiIndex j=0; j < i; ++j) {
            ovals[i * stride + j] = ovals[j * stride + i];
        }
    }
    
    return 0;
}


// Row-major
int ptiMatrix2Norm(ptiMatrix * const A, ptiValue * const lambda)
{
    ptiIndex const nrows = A->nrows;
    ptiIndex const ncols = A->ncols;
    ptiIndex const stride = A->stride;
    ptiValue * const vals = A->values;
    ptiValue * buffer_lambda;

#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for
#endif
    for(ptiIndex j=0; j < ncols; ++j) {
        lambda[j] = 0.0;
    }

#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel
    {
        int const nthreads = omp_get_num_threads();
        #pragma omp master
        {
            buffer_lambda = (ptiValue *)malloc(nthreads * ncols * sizeof(ptiValue));
            for(ptiNnzIndex j=0; j < nthreads * ncols; ++j)
                buffer_lambda[j] = 0.0;
        }
    }

    #pragma omp parallel
    {
        int const tid = omp_get_thread_num();
        int const nthreads = omp_get_num_threads();
        ptiValue * loc_lambda = buffer_lambda + tid * ncols;

        #pragma omp for
        for(ptiIndex i=0; i < nrows; ++i) {
            for(ptiIndex j=0; j < ncols; ++j) {
                loc_lambda[j] += vals[i*stride + j] * vals[i*stride + j];
            }
        }

        #pragma omp for
        for(ptiIndex j=0; j < ncols; ++j) {
            for(int i=0; i < nthreads; ++i) {
                lambda[j] += buffer_lambda[i*ncols + j];
            }
        }
    }   /* end parallel pragma */

#else

    for(ptiIndex i=0; i < nrows; ++i) {
        for(ptiIndex j=0; j < ncols; ++j) {
            lambda[j] += vals[i*stride + j] * vals[i*stride + j];
        }
    }

#endif

#ifdef HIPARTI_USE_OPENMP
        #pragma omp parallel for
#endif
        for(ptiIndex j=0; j < ncols; ++j) {
            lambda[j] = sqrt(lambda[j]);
        }

#ifdef HIPARTI_USE_OPENMP
        #pragma omp parallel for
#endif
        for(ptiIndex i=0; i < nrows; ++i) {
            for(ptiIndex j=0; j < ncols; ++j) {
                vals[i*stride + j] /= lambda[j];
            }
        }

    
#ifdef HIPARTI_USE_OPENMP
    free(buffer_lambda);
#endif

    return 0;
}

// Row-major
int ptiMatrixMaxNorm(ptiMatrix * const A, ptiValue * const lambda)
{
    ptiIndex const nrows = A->nrows;
    ptiIndex const ncols = A->ncols;
    ptiIndex const stride = A->stride;
    ptiValue * const vals = A->values;
    ptiValue * buffer_lambda;

#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for
#endif
    for(ptiIndex j=0; j < ncols; ++j) {
        lambda[j] = 0.0;
    }

#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel
    {
        int const nthreads = omp_get_num_threads();
        #pragma omp master
        {
            buffer_lambda = (ptiValue *)malloc(nthreads * ncols * sizeof(ptiValue));
            for(ptiNnzIndex j=0; j < nthreads * ncols; ++j)
                buffer_lambda[j] = 0.0;
        }
    }

    #pragma omp parallel
    {
        int const tid = omp_get_thread_num();
        int const nthreads = omp_get_num_threads();
        ptiValue * loc_lambda = buffer_lambda + tid * ncols;

        #pragma omp for
        for(ptiIndex i=0; i < nrows; ++i) {
            for(ptiIndex j=0; j < ncols; ++j) {
                if(vals[i*stride + j] > loc_lambda[j])
                    loc_lambda[j] = vals[i*stride + j];
            }
        }

        #pragma omp for
        for(ptiIndex j=0; j < ncols; ++j) {
            for(int i=0; i < nthreads; ++i) {
                if(buffer_lambda[i*ncols + j] > lambda[j])
                    lambda[j] = buffer_lambda[i*ncols + j];
            }
        }
    }   /* end parallel pragma */

#else
    for(ptiIndex i=0; i < nrows; ++i) {
        for(ptiIndex j=0; j < ncols; ++j) {
            if(vals[i*stride + j] > lambda[j])
                lambda[j] = vals[i*stride + j];
        }
    }
#endif

#ifdef HIPARTI_USE_OPENMP
        #pragma omp parallel for
#endif
        for(ptiIndex j=0; j < ncols; ++j) {
            if(lambda[j] < 1)
                lambda[j] = 1;
        }

#ifdef HIPARTI_USE_OPENMP
        #pragma omp parallel for
#endif
        for(ptiIndex i=0; i < nrows; ++i) {
            for(ptiIndex j=0; j < ncols; ++j) {
                vals[i*stride + j] /= lambda[j];
            }
        }

#ifdef HIPARTI_USE_OPENMP
    free(buffer_lambda);
#endif

    return 0;
}


void GetFinalLambda(
  ptiIndex const rank,
  ptiIndex const nmodes,
  ptiMatrix ** mats,
  ptiValue * const lambda)
{
  ptiValue * tmp_lambda =  (ptiValue *) malloc(rank * sizeof(*tmp_lambda));

  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiMatrix2Norm(mats[m], tmp_lambda);
    for(ptiIndex r=0; r < rank; ++r) {
      lambda[r] *= tmp_lambda[r];
    }
  }

  free(tmp_lambda);
}
