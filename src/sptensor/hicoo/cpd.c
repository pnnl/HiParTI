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
#include "hicoo.h"


double CpdAlsStepHiCOO(
  ptiSparseTensorHiCOO const * const hitsr,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  ptiRankMatrix ** mats,
  ptiValue * const lambda)
{
  ptiIndex const nmodes = hitsr->nmodes;
  ptiIndex const stride = mats[0]->stride;
  double fit = 0;

  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiAssert(hitsr->ndims[m] == mats[m]->nrows);
    ptiAssert(mats[m]->ncols == rank);
  }

  ptiValue alpha = 1.0, beta = 0.0;
  char notrans = 'N';
  char trans = 'T';
  char uplo = 'L';
  int blas_rank = (int) rank;
  int blas_stride = (int) stride;

  ptiRankMatrix * tmp_mat = mats[nmodes];
  ptiRankMatrix ** ata = (ptiRankMatrix **)malloc((nmodes+1) * sizeof(*ata));
  for(ptiIndex m=0; m < nmodes+1; ++m) {
    ata[m] = (ptiRankMatrix *)malloc(sizeof(ptiRankMatrix));
    ptiAssert(ptiNewRankMatrix(ata[m], rank, rank) == 0);
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
  //   ptiDumpRankMatrix(mats[m], stdout);
  // printf("Initial ata:\n");
  // for(ptiIndex m=0; m < nmodes+1; ++m)
  //   ptiDumpRankMatrix(ata[m], stdout);

  double oldfit = 0;
  ptiIndex * mats_order = (ptiIndex*)malloc(nmodes * sizeof(*mats_order));

  for(ptiIndex it=0; it < niters; ++it) {
    // printf("  its = %3lu\n", it+1);
    ptiTimer timer;
    ptiNewTimer(&timer, 0);
    ptiStartTimer(timer);

    for(ptiIndex m=0; m < nmodes; ++m) {
      printf("\nmode %u \n", m);
      tmp_mat->nrows = mats[m]->nrows;

      /* Factor Matrices order */
      mats_order[0] = m;
      for(ptiIndex i=1; i<nmodes; ++i)
          mats_order[i] = (m+i) % nmodes;     

      ptiAssert (ptiMTTKRPHiCOO_MatrixTiling(hitsr, mats, mats_order, m) == 0);
      // printf("ptiMTTKRPHiCOO_MatrixTiling mats[nmodes]:\n");
      // ptiDumpRankMatrix(mats[nmodes], stdout);

      memcpy(mats[m]->values, tmp_mat->values, mats[m]->nrows * stride * sizeof(ptiValue));
      /* Solve ? * ata[nmodes] = mats[nmodes] (tmp_mat) */
      ptiAssert ( ptiRankMatrixSolveNormals(m, nmodes, ata, mats[m]) == 0 );
      // printf("Inverse mats[m]:\n");
      // ptiDumpRankMatrix(mats[m], stdout);

      /* Normalized mats[m], store the norms in lambda. Use different norms to avoid precision explosion. */
      if (it == 0 ) {
        ptiRankMatrix2Norm(mats[m], lambda);
      } else {
        ptiRankMatrixMaxNorm(mats[m], lambda);
      }
      // printf("Normalize mats[m]:\n");
      // ptiDumpRankMatrix(mats[m], stdout);
      // printf("lambda:\n");
      // for(size_t i=0; i<rank; ++i)
      //   printf("%lf  ", lambda[i]);
      // printf("\n\n");

      /* ata[m] = mats[m]^T * mats[m]) */
      int blas_nrows = (int)(mats[m]->nrows);
      ssyrk_(&uplo, &notrans, &blas_rank, &blas_nrows, &alpha,
        mats[m]->values, &blas_stride, &beta, ata[m]->values, &blas_stride);
      // printf("Update ata[m]:\n");
      // ptiDumpRankMatrix(ata[m], stdout);

    } // Loop nmodes

    // PrintDenseValueVector(lambda, rank, "lambda", "debug.txt");
    fit = KruskalTensorFitHiCOO(hitsr, lambda, mats, ata);

    ptiStopTimer(timer);
    double its_time = ptiElapsedTime(timer);
    ptiFreeTimer(timer);

    printf("  its = %3u ( %.3lf s ) fit = %0.5f  delta = %+0.4e\n",
        it+1, its_time, fit, fit - oldfit);
    if(it > 0 && fabs(fit - oldfit) < tol) {
      break;
    }
    oldfit = fit;
    
  } // Loop niters

  GetRankFinalLambda(rank, nmodes, mats, lambda);

  for(ptiIndex m=0; m < nmodes+1; ++m) {
    ptiFreeRankMatrix(ata[m]);
  }
  free(ata);
  free(mats_order);


  return fit;
}


int ptiCpdAlsHiCOO(
  ptiSparseTensorHiCOO const * const hitsr,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  ptiRankKruskalTensor * ktensor)
{
  ptiIndex nmodes = hitsr->nmodes;
#ifdef HIPARTI_USE_MAGMA
  magma_init();
#endif

  /* Initialize factor matrices */
  ptiIndex max_dim = 0;
  for(ptiIndex m=0; m < nmodes; ++m) {
    max_dim = (hitsr->ndims[m] > max_dim) ? hitsr->ndims[m] : max_dim;
  }
  ptiRankMatrix ** mats = (ptiRankMatrix **)malloc((nmodes+1) * sizeof(*mats));
  for(ptiIndex m=0; m < nmodes+1; ++m) {
    mats[m] = (ptiRankMatrix *)malloc(sizeof(ptiRankMatrix));
  }
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiAssert(ptiNewRankMatrix(mats[m], hitsr->ndims[m], rank) == 0);
    // ptiAssert(ptiConstantRankMatrix(mats[m], 1) == 0);
    ptiAssert(ptiRandomizeRankMatrix(mats[m], hitsr->ndims[m], rank) == 0);
  }
  ptiAssert(ptiNewRankMatrix(mats[nmodes], max_dim, rank) == 0);
  ptiAssert(ptiConstantRankMatrix(mats[nmodes], 0) == 0);
  // printf("max_dim: %u\n", max_dim);

  ptiTimer timer;
  ptiNewTimer(&timer, 0);
  ptiStartTimer(timer);

  ktensor->fit = CpdAlsStepHiCOO(hitsr, rank, niters, tol, mats, ktensor->lambda);

  ptiStopTimer(timer);
  ptiPrintElapsedTime(timer, "CPU  HiCOO SpTns CPD-ALS");
  ptiFreeTimer(timer);

  ktensor->factors = mats;

#ifdef HIPARTI_USE_MAGMA
  magma_finalize();
#endif
  ptiFreeRankMatrix(mats[nmodes]);

  return 0;
}
