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
#include <assert.h>

void ptiSparseMatrixStatusHiCOO(ptiSparseMatrixHiCOO *himtx, FILE *fp)
{
  fprintf(fp, "HiCOO Sparse Tensor information ---------\n");
  fprintf(fp, " DIMS=%"HIPARTI_PRI_INDEX "x%"HIPARTI_PRI_INDEX "\n", himtx->nrows, himtx->ncols);
  fprintf(fp, " NNZ=%"HIPARTI_PRI_NNZ_INDEX "\n", himtx->nnz);
  fprintf(fp, " sb=%"HIPARTI_PRI_INDEX "\n", (ptiIndex)pow(2, himtx->sb_bits));
  fprintf(fp, " nb=%"HIPARTI_PRI_NNZ_INDEX "\n", himtx->bptr.len - 1);
  fprintf(fp, " nk=%"HIPARTI_PRI_NNZ_INDEX "\n", himtx->kptr.len - 1);

  ptiNnzIndex bytes = himtx->nnz * ( sizeof(ptiValue) + 2 * sizeof(ptiElementIndex) );
  bytes += himtx->bindI.len * 2 * sizeof(ptiBlockIndex);
  bytes += himtx->bptr.len * sizeof(ptiNnzIndex);
  bytes += himtx->kptr.len * sizeof(ptiNnzIndex);
  /* add kschr */
  ptiIndex sk = (ptiIndex)pow(2, himtx->sk_bits);
  ptiIndex kernel_ndim = (himtx->nrows + sk - 1)/sk;
  for(ptiIndex i=0; i < kernel_ndim; ++i) {
    bytes += himtx->kschr[i].len * sizeof(ptiIndex);
  }
  bytes += kernel_ndim * sizeof(ptiIndexVector *);
  
  char * bytestr = ptiBytesString(bytes);
  fprintf(fp, " HiCOO-STORAGE=%s\n", bytestr);
  free(bytestr);


  fprintf(fp, "SCHEDULE INFO [KERNEL]: \n");
  fprintf(fp, "SCHEDULE : %"HIPARTI_PRI_INDEX" x %"HIPARTI_PRI_INDEX"\n", kernel_ndim, himtx->nkiters);


  ptiNnzIndex max_nnzk = 0;
  ptiNnzIndex min_nnzk = HIPARTI_NNZ_INDEX_MAX;
  ptiNnzIndex sum_nnzk = 0;
  ptiNnzIndex avg_nnzk = himtx->nnz / (himtx->kptr.len - 1);
  double std_nnzk = 0.0;
  for(ptiIndex k=0; k<himtx->kptr.len - 1; ++k) {
    ptiNnzIndex nnzk = 0;
    for(ptiIndex b=himtx->kptr.data[k]; b<himtx->kptr.data[k+1]; ++b) {
      ptiNnzIndex nnzb = himtx->bptr.data[b+1] - himtx->bptr.data[b];
      nnzk += nnzb;
    }
    sum_nnzk += nnzk;
    std_nnzk += (nnzk - avg_nnzk) * (nnzk - avg_nnzk);
    if(min_nnzk > nnzk) min_nnzk = nnzk;
    if(max_nnzk < nnzk) max_nnzk = nnzk;
  }
  std_nnzk = sqrt(std_nnzk / (himtx->kptr.len - 1));
  assert(sum_nnzk == himtx->nnz);
  fprintf(fp, "Nnzk: Max=%" HIPARTI_PRI_NNZ_INDEX ", Min=%" HIPARTI_PRI_NNZ_INDEX ", Avg=%" HIPARTI_PRI_NNZ_INDEX ", Std: %.1lf\n", max_nnzk, min_nnzk, avg_nnzk, std_nnzk);

  ptiIndex sb = (ptiIndex)pow(2, himtx->sb_bits);
  ptiNnzIndex max_nnzb = himtx->bptr.data[1] - himtx->bptr.data[0];
  ptiNnzIndex min_nnzb = himtx->bptr.data[1] - himtx->bptr.data[0];
  ptiNnzIndex sum_nnzb = 0;
  double geo_mean_nnzb = 1;
  ptiNnzIndex nb = himtx->bptr.len - 1;
  ptiNnzIndex * nnzb_array = (ptiNnzIndex *)malloc(nb * sizeof(* nnzb_array));
  for(ptiNnzIndex i=0; i < himtx->bptr.len - 1; ++i) {
    ptiNnzIndex nnzb = himtx->bptr.data[i+1] - himtx->bptr.data[i];
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
  assert(sum_nnzb == himtx->nnz);
  ptiNnzIndex avg_nnzb = (ptiNnzIndex)sum_nnzb / (himtx->bptr.len - 1);

  /* Compute median */
  ptiQuickSortNnzIndexArray(nnzb_array, 0, nb);
  long median_loc = (nb + 1) / 2 - 1;
  assert (median_loc >= 0);
  ptiNnzIndex median_nnzb = nnzb_array[median_loc];
  free(nnzb_array);
  
  fprintf(fp, " block nnz info:\n");
  fprintf(fp, " Nnzb: Max=%" HIPARTI_PRI_NNZ_INDEX ", Min=%" HIPARTI_PRI_NNZ_INDEX ", Avg=%" HIPARTI_PRI_NNZ_INDEX "\n", max_nnzb, min_nnzb, avg_nnzb);
  fprintf(fp, " cb: Max=%.3lf, Min=%.3lf, Avg=%.3lf\n", (double)max_nnzb / sb, (double)min_nnzb / sb, (double)avg_nnzb / sb);
  fprintf(fp, " median cb: %.3lf, geometric mean cb: %.3lf\n", (double)median_nnzb / sb, geo_mean_nnzb);
  fprintf(fp, " alpha_b: %lf\n", (double)(himtx->bptr.len - 1) / himtx->nnz);

}
