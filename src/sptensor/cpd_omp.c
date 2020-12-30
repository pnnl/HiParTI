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
#include <math.h>
#ifdef HIPARTI_USE_MAGMA
  #include "magma_v2.h"
  #include "magma_lapack.h"
#else
  #include "clapack.h"
#endif
#include "sptensor.h"


double OmpCpdAlsStep(
  ptiSparseTensor const * const ptien,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  const int tk,
  const int use_reduce,
  ptiMatrix ** mats,  // Row-major
  ptiMatrix ** copy_mats,
  ptiValue * const lambda)
{
  ptiIndex const nmodes = ptien->nmodes;
  ptiIndex const stride = mats[0]->stride;
  double fit = 0;
#ifdef HIPARTI_USE_OPENMP  
  omp_set_num_threads(tk);
#endif

  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiAssert(ptien->ndims[m] == mats[m]->nrows);
    ptiAssert(mats[m]->ncols == rank);
    // assert(mats[m]->stride == rank);  // for correct column-major magma functions
  }

  ptiValue alpha = 1.0, beta = 0.0;
  char notrans = 'N';
  char trans = 'T';
  char uplo = 'L';
  int blas_rank = (int) rank;
  int blas_stride = (int) stride;

  ptiMatrix * tmp_mat = mats[nmodes];
  ptiMatrix ** ata = (ptiMatrix **)malloc((nmodes+1) * sizeof(*ata)); // symmetric matrices, but in column-major
  for(ptiIndex m=0; m < nmodes+1; ++m) {
    ata[m] = (ptiMatrix *)malloc(sizeof(ptiMatrix));
    ptiAssert(ptiNewMatrix(ata[m], rank, rank) == 0);
    ptiAssert(mats[m]->stride == ata[m]->stride);
  }

  /* Compute all "ata"s */
  for(ptiIndex m=0; m < nmodes; ++m) {
    /* ata[m] = mats[m]^T * mats[m]), actually do A * A' due to row-major mats, and output an upper triangular matrix. */
    int blas_nrows = (int)(mats[m]->nrows);
    ssyrk_(&uplo, &notrans, &blas_rank, &blas_nrows, &alpha,
      mats[m]->values, &blas_stride, &beta, ata[m]->values, &blas_stride);
  }
  // printf("Initial mats:\n");
  // for(size_t m=0; m < nmodes+1; ++m)
  //   ptiDumpMatrix(mats[m], stdout);
  // printf("Initial ata:\n");
  // for(size_t m=0; m < nmodes+1; ++m)
  //   ptiDumpMatrix(ata[m], stdout);


  double oldfit = 0;
  ptiIndex * mats_order = (ptiIndex*)malloc(nmodes * sizeof(*mats_order));


  for(ptiIndex it=0; it < niters; ++it) {
    // printf("  its = %3lu\n", it+1);
    ptiTimer timer;
    ptiNewTimer(&timer, 0);
    ptiStartTimer(timer);

    for(ptiIndex m=0; m < nmodes; ++m) {
      // printf("\nmode %lu \n", m);
      tmp_mat->nrows = mats[m]->nrows;

      /* Factor Matrices order */
      mats_order[0] = m;
      for(ptiIndex i=1; i<nmodes; ++i)
          mats_order[i] = (m+i) % nmodes;

      // mats[nmodes]: row-major
      if(use_reduce == 1) {
        ptiAssert (ptiOmpMTTKRP_Reduce(ptien, mats, copy_mats, mats_order, m, tk) == 0);
      } else {
        ptiAssert (ptiOmpMTTKRP(ptien, mats, mats_order, m, tk) == 0);
      }
      // printf("ptiMTTKRP mats[nmodes]:\n");
      // ptiDumpMatrix(mats[nmodes], stdout);

      // Row-major
#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for num_threads(tk)
#endif
      for(ptiIndex i=0; i<mats[m]->nrows * stride; ++i)
        mats[m]->values[i] = tmp_mat->values[i];        
      // for(ptiIndex i=0; i<mats[m]->nrows; ++i) {
      //   for(ptiIndex j=0; j<mats[m]->ncols; ++j) {
      //     mats[m]->values[i * mats[m]->stride + j] = tmp_mat->values[i * mats[m]->stride + j];
      //   } 
      // }

      /* Solve ? * ata[nmodes] = mats[nmodes] (tmp_mat) */
      ptiAssert ( ptiMatrixSolveNormals(m, nmodes, ata, mats[m]) == 0 );
      // printf("Inverse mats[m]:\n");
      // ptiDumpMatrix(mats[m], stdout);

      /* Normalized mats[m], store the norms in lambda. Use different norms to avoid precision explosion. */
      if (it == 0 ) {
        ptiMatrix2Norm(mats[m], lambda);
      } else {
        ptiMatrixMaxNorm(mats[m], lambda);
      }
      // printf("Normalize mats[m]:\n");
      // ptiDumpMatrix(mats[m], stdout);
      // printf("lambda:\n");
      // for(size_t i=0; i<rank; ++i)
      //   printf("%lf  ", lambda[i]);
      // printf("\n\n");

      /* ata[m] = mats[m]^T * mats[m]) */
      int blas_nrows = (int)(mats[m]->nrows);
      ssyrk_(&uplo, &notrans, &blas_rank, &blas_nrows, &alpha,
        mats[m]->values, &blas_stride, &beta, ata[m]->values, &blas_stride);
      // printf("Update ata[m]:\n");
      // ptiDumpMatrix(ata[m], stdout);

    } // Loop nmodes

    // PrintDenseValueVector(lambda, rank, "lambda", "debug.txt");
    fit = KruskalTensorFit(ptien, lambda, mats, ata);

    ptiStopTimer(timer);
    double its_time = ptiElapsedTime(timer);
    ptiFreeTimer(timer);

    printf("  its = %3"HIPARTI_PRI_INDEX " ( %.3lf s ) fit = %0.5f  delta = %+0.4e\n",
        it+1, its_time, fit, fit - oldfit);
    if(it > 0 && fabs(fit - oldfit) < tol) {
      break;
    }
    oldfit = fit;

  } // Loop niters

  GetFinalLambda(rank, nmodes, mats, lambda);

  for(ptiIndex m=0; m < nmodes+1; ++m) {
    ptiFreeMatrix(ata[m]);
  }
  free(ata);
  free(mats_order);

  return fit;
}


int ptiOmpCpdAls(
  ptiSparseTensor const * const ptien,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  const int tk,
  const int use_reduce,
  ptiKruskalTensor * ktensor)
{
  ptiIndex nmodes = ptien->nmodes;
#ifdef HIPARTI_USE_MAGMA
  magma_init();
#endif

  /* Initialize factor matrices */
  ptiIndex max_dim = ptiMaxIndexArray(ptien->ndims, nmodes);
  ptiMatrix ** mats = (ptiMatrix **)malloc((nmodes+1) * sizeof(*mats));
  for(ptiIndex m=0; m < nmodes+1; ++m) {
    mats[m] = (ptiMatrix *)malloc(sizeof(ptiMatrix));
  }
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiAssert(ptiNewMatrix(mats[m], ptien->ndims[m], rank) == 0);
    // ptiAssert(ptiConstantMatrix(mats[m], 1) == 0);
    ptiAssert(ptiRandomizeMatrix(mats[m]) == 0);
  }
  ptiAssert(ptiNewMatrix(mats[nmodes], max_dim, rank) == 0);
  ptiAssert(ptiConstantMatrix(mats[nmodes], 0) == 0);

  ptiMatrix ** copy_mats;
  if(use_reduce == 1) {
    copy_mats = (ptiMatrix **)malloc(tk * sizeof(*copy_mats));
    for(int t=0; t<tk; ++t) {
      copy_mats[t] = (ptiMatrix *)malloc(sizeof(ptiMatrix));
      ptiAssert(ptiNewMatrix(copy_mats[t], max_dim, rank) == 0);
      ptiAssert(ptiConstantMatrix(copy_mats[t], 0) == 0);
    }
  }

  ptiTimer timer;
  ptiNewTimer(&timer, 0);
  ptiStartTimer(timer);

  ktensor->fit = OmpCpdAlsStep(ptien, rank, niters, tol, tk, use_reduce, mats, copy_mats, ktensor->lambda);

  ptiStopTimer(timer);
  ptiPrintElapsedTime(timer, "CPU  SpTns CPD-ALS");
  ptiFreeTimer(timer);

  ktensor->factors = mats;

#ifdef HIPARTI_USE_MAGMA
  magma_finalize();
#endif
  ptiFreeMatrix(mats[nmodes]);
  if(use_reduce == 1) {
    for(int t=0; t<tk; ++t) {
      ptiFreeMatrix(copy_mats[t]);
      free(copy_mats[t]);
    }
    free(copy_mats);
  }

  return 0;
}
