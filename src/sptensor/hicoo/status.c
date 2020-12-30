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
#include "hicoo.h"
#include <assert.h>

void ptiSparseTensorStatusHiCOO(ptiSparseTensorHiCOO *hitsr, FILE *fp)
{
  ptiIndex nmodes = hitsr->nmodes;
  fprintf(fp, "HiCOO Sparse Tensor information ---------\n");
  fprintf(fp, "DIMS=%"HIPARTI_PRI_INDEX, hitsr->ndims[0]);
  for(ptiIndex m=1; m < nmodes; ++m) {
    fprintf(fp, "x%"HIPARTI_PRI_INDEX, hitsr->ndims[m]);
  }
  fprintf(fp, " NNZ=%"HIPARTI_PRI_NNZ_INDEX, hitsr->nnz);
  fprintf(fp, "\n");
  fprintf(fp, "sb=%"HIPARTI_PRI_INDEX, (ptiIndex)pow(2, hitsr->sb_bits));
  fprintf(fp, " sk=%"HIPARTI_PRI_INDEX, (ptiIndex)pow(2, hitsr->sk_bits));
  fprintf(fp, " sc=%"HIPARTI_PRI_INDEX, (ptiIndex)pow(2, hitsr->sc_bits));
  fprintf(fp, "\n");
  fprintf(fp, "nb=%"HIPARTI_PRI_NNZ_INDEX, hitsr->bptr.len - 1);
  fprintf(fp, " nk=%"HIPARTI_PRI_NNZ_INDEX, hitsr->kptr.len - 1);
  fprintf(fp, " nc=%"HIPARTI_PRI_NNZ_INDEX, hitsr->cptr.len - 1);
  fprintf(fp, "\n");

  ptiNnzIndex bytes = hitsr->nnz * ( sizeof(ptiValue) + nmodes * sizeof(ptiElementIndex) );
  bytes += hitsr->binds[0].len * nmodes * sizeof(ptiBlockIndex);
  bytes += hitsr->bptr.len * sizeof(ptiNnzIndex);
  bytes += hitsr->kptr.len * sizeof(ptiNnzIndex);
  bytes += hitsr->cptr.len * sizeof(ptiNnzIndex);
  /* add kschr */
  ptiIndex sk = (ptiIndex)pow(2, hitsr->sk_bits);
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
    for(ptiIndex i=0; i < kernel_ndim; ++i) {
      bytes += hitsr->kschr[m][i].len * sizeof(ptiIndex);
    }
    bytes += kernel_ndim * sizeof(ptiIndexVector *);
  }
  bytes += nmodes * sizeof(ptiIndexVector **);
  /* add nkiters  */
  bytes += nmodes * sizeof(ptiIndex);

  /* add kschr_balanced */
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
    for(ptiIndex i=0; i < kernel_ndim; ++i) {
      bytes += hitsr->kschr_balanced[m][i].len * sizeof(ptiIndex);
    }
    bytes += kernel_ndim * sizeof(ptiIndexVector *);
  }
  bytes += nmodes * sizeof(ptiIndexVector **);

  /* add kschr_balanced_pos */
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
    for(ptiIndex i=0; i < kernel_ndim; ++i) {
      bytes += hitsr->kschr_balanced_pos[m][i].len * sizeof(ptiIndex);
    }
    bytes += kernel_ndim * sizeof(ptiIndexVector *);
  }
  bytes += nmodes * sizeof(ptiIndexVector **);

  /* add kschr_rest */
  for(ptiIndex m=0; m < nmodes; ++m) {
    bytes += hitsr->kschr_rest[m].len * sizeof(ptiIndexVector *);
  }
  bytes += nmodes * sizeof(ptiIndexVector *);

  /* add knnzs */
  bytes += hitsr->knnzs.len * sizeof(ptiNnzIndex);

  char * bytestr = ptiBytesString(bytes);
  fprintf(fp, "HiCOO-STORAGE=%s\n", bytestr);
  free(bytestr);

  fprintf(fp, "SCHEDULE INFO [KERNEL]: \n");
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
    fprintf(fp, "SCHEDULE MODE %"HIPARTI_PRI_INDEX" : %"HIPARTI_PRI_INDEX" x %"HIPARTI_PRI_INDEX"\n", m, kernel_ndim, hitsr->nkiters[m]);
  }

  fprintf(fp, "BALANCED SCHEDULE INFO [KERNEL]: \n");
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
    ptiIndex npars = hitsr->nkpars[m];
    fprintf(fp, "SCHEDULE MODE %"HIPARTI_PRI_INDEX" : %"HIPARTI_PRI_INDEX" x %"HIPARTI_PRI_INDEX"\n", m, kernel_ndim, npars);
  }
  
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiNnzIndex sum_balanced_nnzk = 0;
    ptiIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
    for(ptiIndex i=0; i < kernel_ndim; ++i) {
      for(ptiIndex j=0; j < hitsr->kschr_balanced[m][i].len; ++j) {
        ptiIndex kernel_num = hitsr->kschr_balanced[m][i].data[j];
        sum_balanced_nnzk += hitsr->knnzs.data[kernel_num];
      }
    }
    fprintf(fp, "MODE %"HIPARTI_PRI_INDEX" : Balanced nnzs: %.2lf, rest nnzs: %.2lf\n", m, (double)sum_balanced_nnzk / hitsr->nnz, 1.0 - (double)sum_balanced_nnzk / hitsr->nnz);
  }

  // fprintf(fp, "SCHEDULE DETAILS (kschr): \n");
  // for(ptiIndex m=0; m < nmodes; ++m) {
  //   printf("Mode %u\n", m);
  //   ptiIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
  //   for(ptiIndex i=0; i < kernel_ndim; ++i) {
  //     ptiDumpIndexVector(&hitsr->kschr[m][i], fp);
  //   }
  //   fprintf(fp, "\n");
  // }
  // fprintf(fp, "\n");

  // fprintf(fp, "SCHEDULE DETAILS (kschr_balanced): \n");
  // for(ptiIndex m=0; m < nmodes; ++m) {
  //   printf("Mode %u\n", m);
  //   ptiIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
  //   for(ptiIndex i=0; i < kernel_ndim; ++i) {
  //     ptiDumpIndexVector(&hitsr->kschr_balanced[m][i], fp);
  //   }
  //   fprintf(fp, "\n");
  // }
  // fprintf(fp, "\n");

  // fprintf(fp, "SCHEDULE DETAILS (kschr_balanced_pos): \n");
  // for(ptiIndex m=0; m < nmodes; ++m) {
  //   printf("Mode %u\n", m);
  //   ptiIndex kernel_ndim = (hitsr->ndims[m] + sk - 1)/sk;
  //   for(ptiIndex i=0; i < kernel_ndim; ++i) {
  //     ptiDumpIndexVector(&hitsr->kschr_balanced_pos[m][i], fp);
  //   }
  //   fprintf(fp, "\n");
  // }
  // fprintf(fp, "\n");

  // fprintf(fp, "kschr_rest: \n");
  // for(ptiIndex m=0; m < nmodes; ++m) {
  //   printf("Mode %u\n", m);
  //   ptiDumpIndexVector(&hitsr->kschr_rest[m], fp);
  // }

  ptiNnzIndex max_nnzk = 0;
  ptiNnzIndex min_nnzk = HIPARTI_NNZ_INDEX_MAX;
  ptiNnzIndex sum_nnzk = 0;
  ptiNnzIndex avg_nnzk = hitsr->nnz / (hitsr->kptr.len - 1);
  double std_nnzk = 0.0;
  for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
    ptiNnzIndex nnzk = hitsr->knnzs.data[k];
    // ptiNnzIndex nnzk = 0;
    // for(ptiIndex b=hitsr->kptr.data[k]; b<hitsr->kptr.data[k+1]; ++b) {
    //   ptiNnzIndex nnzb = hitsr->bptr.data[b+1] - hitsr->bptr.data[b];
    //   nnzk += nnzb;
    // }
    sum_nnzk += nnzk;
    std_nnzk += (nnzk - avg_nnzk) * (nnzk - avg_nnzk);
    if(min_nnzk > nnzk) min_nnzk = nnzk;
    if(max_nnzk < nnzk) max_nnzk = nnzk;
  }
  std_nnzk = sqrt(std_nnzk / (hitsr->kptr.len - 1));
  assert(sum_nnzk == hitsr->nnz);
  // fprintf(fp, "kernel pointers:\n");
  // ptiDumpNnzIndexVector(&hitsr->kptr, fp);
  // fprintf(fp, "kernel nnzs:\n");
  // ptiDumpNnzIndexVector(&hitsr->knnzs, fp);
  fprintf(fp, "Nnzk: Max=%" HIPARTI_PRI_NNZ_INDEX ", Min=%" HIPARTI_PRI_NNZ_INDEX ", Avg=%" HIPARTI_PRI_NNZ_INDEX ", Std: %.1lf\n", max_nnzk, min_nnzk, avg_nnzk, std_nnzk);

  ptiIndex sb = (ptiIndex)pow(2, hitsr->sb_bits);
  ptiNnzIndex max_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
  ptiNnzIndex min_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
  ptiNnzIndex sum_nnzb = 0;
  double geo_mean_nnzb = 1;
  ptiNnzIndex nb = hitsr->bptr.len - 1;
  ptiNnzIndex * nnzb_array = (ptiNnzIndex *)malloc(nb * sizeof(* nnzb_array));
  for(ptiNnzIndex i=0; i < hitsr->bptr.len - 1; ++i) {
    ptiNnzIndex nnzb = hitsr->bptr.data[i+1] - hitsr->bptr.data[i];
    // fprintf(fp, "%lu, ", nnzb);
    if(max_nnzb < nnzb) {
      max_nnzb = nnzb;
    }
    if(min_nnzb > nnzb) {
      min_nnzb = nnzb;
    }
    sum_nnzb += nnzb;
    geo_mean_nnzb *= pow( (double)nnzb / sb, 1.0/nb );
    nnzb_array[i] = nnzb;
  }
  assert(sum_nnzb == hitsr->nnz);
  ptiNnzIndex avg_nnzb = (ptiNnzIndex)sum_nnzb / (hitsr->bptr.len - 1);

  /* Compute median */
  ptiQuickSortNnzIndexArray(nnzb_array, 0, nb);
  ptiNnzIndex median_loc = (nb + 1) / 2 - 1;
  assert (median_loc >= 0);
  ptiNnzIndex median_nnzb = nnzb_array[median_loc];
  free(nnzb_array);
  
  fprintf(fp, "block nnzs:\n");
  fprintf(fp, "Nnzb: Max=%" HIPARTI_PRI_NNZ_INDEX ", Min=%" HIPARTI_PRI_NNZ_INDEX ", Avg=%" HIPARTI_PRI_NNZ_INDEX "\n", max_nnzb, min_nnzb, avg_nnzb);
  fprintf(fp, "cb: Max=%.3lf, Min=%.3lf, Avg=%.3lf\n", (double)max_nnzb / sb, (double)min_nnzb / sb, (double)avg_nnzb / sb);
  fprintf(fp, "median cb: %.3lf, geometric mean cb: %.3lf\n", (double)median_nnzb / sb, geo_mean_nnzb);
  fprintf(fp, "alpha_b: %lf\n", (double)(hitsr->bptr.len - 1) / hitsr->nnz);

  fprintf(fp, "\nParameter configuration --------\n");
  fprintf(fp, "Suggest B (sb) <= %.2lf / R. For cache efficiency\n", (double)L1_SIZE / hitsr->nmodes / sizeof(ptiValue));
  fprintf(fp, "Suggest alpha_b in (0,1], small is better. For tensor storage\n");
  fprintf(fp, "Suggest cb > 1, large is better. For MTTKRP performance\n");
  fprintf(fp, "Suggest num_tasks should in [%d, %d] PAR_DEGREE: [%d, %d]. For parallel efficiency\n", PAR_MIN_DEGREE * NUM_CORES, PAR_MAX_DEGREE * NUM_CORES, PAR_MIN_DEGREE, PAR_MAX_DEGREE);
  fprintf(fp, "\n\n");

}
