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

#ifdef HIPARTI_USE_OPENMP

double OmpCpdAlsStepHiCOO(
  ptiSparseTensorHiCOO const * const hitsr,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  const int tk,
  const int tb,
  const int * par_iters,
  ptiRankMatrix ** mats,
  ptiRankMatrix *** copy_mats,
  ptiValue * const lambda,
  int balanced)
{
  ptiIndex const nmodes = hitsr->nmodes;
  ptiIndex const stride = mats[0]->stride;
  double fit = 0;

  omp_set_num_threads(tk);
#ifdef HIPARTI_USE_MAGMA
  magma_set_omp_numthreads(tk);
  magma_set_lapack_numthreads(tk);
  // printf("magma nthreads: %d\n", magma_get_parallel_numthreads());
  // printf("magma nthreads: %d\n", magma_get_omp_numthreads());
  // printf("magma lapack nthreads: %d\n", magma_get_lapack_numthreads());
#endif

  // ptiAssert(stride == rank);  // for correct column-major magma functions
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiAssert(hitsr->ndims[m] == mats[m]->nrows);
    ptiAssert(mats[m]->ncols == rank);
  }

  ptiValue alpha = 1.0, beta = 0.0;
  char notrans = 'N';
  // char trans = 'T';
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

  ptiTimer tmp_timer;
  ptiNewTimer(&tmp_timer, 0);
  double mttkrp_time, solver_time, norm_time, ata_time, fit_time;
  // double sum_time = 0.0;

  for(ptiIndex it=0; it < niters; ++it) {
    // printf("  its = %3lu\n", it+1);
    // sum_time = 0.0;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);
    ptiStartTimer(timer);

    for(ptiIndex m=0; m < nmodes; ++m) {
      // printf("\nmode %u \n", m);
      tmp_mat->nrows = mats[m]->nrows;

      /* Factor Matrices order */
      mats_order[0] = m;
      for(ptiIndex i=1; i<nmodes; ++i)
          mats_order[i] = (m+i) % nmodes;     

      // ptiAssert (ptiOmpMTTKRPHiCOO_MatrixTiling(hitsr, mats, mats_order, m) == 0);
      ptiStartTimer(tmp_timer);
      if(par_iters[m] == 1) {
        ptiAssert (ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(hitsr, mats, copy_mats[m], mats_order, m, tk, tb, balanced) == 0);
      } else {
        ptiAssert (ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled(hitsr, mats, mats_order, m, tk, tb, balanced) == 0);
      }
      ptiStopTimer(tmp_timer);
      mttkrp_time = ptiPrintElapsedTime(tmp_timer, "MTTKRP");
      // printf("ptiMTTKRPHiCOO_MatrixTiling mats[nmodes]:\n");
      // ptiDumpRankMatrix(mats[nmodes], stdout);

      ptiStartTimer(tmp_timer);
#ifdef HIPARTI_USE_OPENMP
      #pragma omp parallel for num_threads(tk)
#endif
      for(ptiIndex i=0; i<mats[m]->nrows * stride; ++i)
        mats[m]->values[i] = tmp_mat->values[i];

      /* Solve ? * ata[nmodes] = mats[nmodes] (tmp_mat) */
      /* result is row-major, solve AT XT = BT */
      ptiAssert ( ptiRankMatrixSolveNormals(m, nmodes, ata, mats[m]) == 0 );
      ptiStopTimer(tmp_timer);
      // solver_time = ptiPrintElapsedTime(tmp_timer, "memcpy and ptiRankMatrixSolveNormals");
      // printf("Inverse mats[m]:\n");
      // ptiDumpRankMatrix(mats[m], stdout);

      /* Normalized mats[m], store the norms in lambda. Use different norms to avoid precision explosion. */
      ptiStartTimer(tmp_timer);
      if (it == 0 ) {
        ptiRankMatrix2Norm(mats[m], lambda);
      } else {
        ptiRankMatrixMaxNorm(mats[m], lambda);
      }
      ptiStopTimer(tmp_timer);
      // norm_time = ptiPrintElapsedTime(tmp_timer, "matrix norm");
      // printf("Normalize mats[m]:\n");
      // ptiDumpRankMatrix(mats[m], stdout);
      // printf("lambda:\n");
      // for(size_t i=0; i<rank; ++i)
      //   printf("%lf  ", lambda[i]);
      // printf("\n\n");

      /* ata[m] = mats[m]^T * mats[m]) */
      ptiStartTimer(tmp_timer);
      int blas_nrows = (int)(mats[m]->nrows);
      ssyrk_(&uplo, &notrans, &blas_rank, &blas_nrows, &alpha,
        mats[m]->values, &blas_stride, &beta, ata[m]->values, &blas_stride);
      ptiStopTimer(tmp_timer);
      // ata_time = ptiPrintElapsedTime(tmp_timer, "update ata");
      // printf("Update ata[m]:\n");
      // ptiDumpRankMatrix(ata[m], stdout);

      // sum_time += mttkrp_time + norm_time + ata_time;

    } // Loop nmodes

    // PrintDenseValueVector(lambda, rank, "lambda", "debug.txt");
    ptiStartTimer(tmp_timer);
    fit = KruskalTensorFitHiCOO(hitsr, lambda, mats, ata);
    ptiStopTimer(tmp_timer);
    // fit_time = ptiPrintElapsedTime(tmp_timer, "KruskalTensorFitHiCOO");

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


int ptiOmpCpdAlsHiCOO(
  ptiSparseTensorHiCOO const * const hitsr,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  const int tk,
  const int tb,
  int balanced,
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
    // assert(ptiConstantRankMatrix(mats[m], 1) == 0);
    ptiAssert(ptiRandomizeRankMatrix(mats[m], hitsr->ndims[m], rank) == 0);
  }
  ptiAssert(ptiNewRankMatrix(mats[nmodes], max_dim, rank) == 0);
  ptiAssert(ptiConstantRankMatrix(mats[nmodes], 0) == 0);

  /* determine niters or num_kernel_dim to be parallelized */
  int * par_iters = (int *)malloc(nmodes * sizeof(*par_iters));
  ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
  for(ptiIndex m=0; m < nmodes; ++m) {
    par_iters[m] = 0;
    ptiIndex num_kernel_dim = (hitsr->ndims[m] + sk - 1) / sk;
    // printf("num_kernel_dim: %u, hitsr->nkiters[m] / num_kernel_dim: %u\n", num_kernel_dim, hitsr->nkiters[m]/num_kernel_dim);
    if(num_kernel_dim <= PAR_MIN_DEGREE * NUM_CORES && hitsr->nkiters[m] / num_kernel_dim >= PAR_DEGREE_REDUCE) {
        par_iters[m] = 1;
    }
  }
  printf("par_iters:\n");
  for(ptiIndex m=0; m < nmodes; ++m) {
    printf("%d, ", par_iters[m]);
  }
  printf("\n");

  ptiRankMatrix *** copy_mats = (ptiRankMatrix ***)malloc(nmodes * sizeof(*copy_mats));
  for(ptiIndex m=0; m < nmodes; ++m) {
    if (par_iters[m] == 1) {
      copy_mats[m] = (ptiRankMatrix **)malloc(tk * sizeof(ptiRankMatrix*));
      for(int t=0; t<tk; ++t) {
        copy_mats[m][t] = (ptiRankMatrix *)malloc(sizeof(ptiRankMatrix));
        ptiAssert(ptiNewRankMatrix(copy_mats[m][t], hitsr->ndims[m], rank) == 0);
        ptiAssert(ptiConstantRankMatrix(copy_mats[m][t], 0) == 0);
      }
    }
  }

  ptiTimer timer;
  ptiNewTimer(&timer, 0);
  ptiStartTimer(timer);

  ktensor->fit = OmpCpdAlsStepHiCOO(hitsr, rank, niters, tol, tk, tb, par_iters,  mats, copy_mats, ktensor->lambda, balanced);

  ptiStopTimer(timer);
  ptiPrintElapsedTime(timer, "CPU  HiCOO SpTns CPD-ALS");
  ptiFreeTimer(timer);

  ktensor->factors = mats;

#ifdef HIPARTI_USE_MAGMA
  magma_finalize();
#endif
  ptiFreeRankMatrix(mats[nmodes]);
  for(ptiIndex m=0; m < nmodes; ++m) {
    if(par_iters[m] == 1) {
      for(int t=0; t<tk; ++t) {
        ptiFreeRankMatrix(copy_mats[m][t]);
        free(copy_mats[m][t]);
      }
      free(copy_mats[m]);
    }
  }
  free(copy_mats);

  return 0;
}

#endif
