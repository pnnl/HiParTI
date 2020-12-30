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

#ifndef HIPARTI_SPTENSOR_H
#define HIPARTI_SPTENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <HiParTI.h>
 

double pti_SparseTensorNorm(const ptiSparseTensor *X);

int pti_SparseTensorCompareIndices(ptiSparseTensor * const tsr1, ptiNnzIndex loc1,  ptiSparseTensor * const tsr2, ptiNnzIndex loc2);
int pti_SparseTensorCompareIndicesMorton2D(
    ptiSparseTensor * const tsr1,
    uint64_t loc1, 
    ptiSparseTensor * const tsr2,
    uint64_t loc2,
    ptiIndex * mode_order,
    ptiElementIndex sb_bits);
int pti_SparseTensorCompareIndicesExceptSingleMode(ptiSparseTensor * const tsr1, ptiNnzIndex loc1, ptiSparseTensor * const tsr2, ptiNnzIndex loc2, ptiIndex * const mode_order);
int pti_SparseTensorCompareIndicesExceptSingleModeCantor(ptiSparseTensor * const tsr1, ptiNnzIndex loc1, ptiSparseTensor * const tsr2, ptiNnzIndex loc2, ptiIndex * const mode_order);
int pti_SparseTensorCompareIndicesRowBlock(
    ptiSparseTensor * const tsr1,
    ptiNnzIndex loc1,
    ptiSparseTensor * const tsr2,
    ptiNnzIndex loc2,
    ptiElementIndex sk_bits);
int pti_SparseTensorCompareIndicesExceptSingleModeRowBlock(
    ptiSparseTensor * const tsr1,
    ptiNnzIndex loc1,
    ptiSparseTensor * const tsr2,
    ptiNnzIndex loc2,
    ptiIndex * const mode_order,
    ptiElementIndex sk_bits);
int pti_SparseTensorCompareIndicesRange(ptiSparseTensor * const tsr, ptiNnzIndex loc, ptiIndex * const inds1, ptiIndex * const inds2);
int pti_SparseTensorCompareIndicesCustomize(ptiSparseTensor * const tsr1, ptiNnzIndex loc1, ptiIndex * const mode_order_1, ptiSparseTensor * const tsr2, ptiNnzIndex loc2, ptiIndex * const mode_order_2, ptiIndex num_ncmodes);
void pti_SwapValues(ptiSparseTensor *tsr, ptiNnzIndex ind1, ptiNnzIndex ind2);

void pti_SparseTensorCollectZeros(ptiSparseTensor *tsr);

int pti_DistSparseTensor(ptiSparseTensor * tsr,
    int const nthreads,
    ptiNnzIndex * const dist_nnzs,
    ptiIndex * dist_nrows);

int pti_DistSparseTensorFixed(ptiSparseTensor * tsr,
    int const nthreads,
    ptiNnzIndex * const dist_nnzs,
    ptiNnzIndex * dist_nrows);

int pti_GetSubSparseTensor(ptiSparseTensor *dest, const ptiSparseTensor *tsr, const ptiIndex limit_low[], const ptiIndex limit_high[]);

struct pti_TagSplitHandle {
    size_t nsplits;
    ptiSparseTensor *tsr;
    size_t *max_size_by_mode;
    size_t *inds_low;
    size_t *inds_high;
    size_t level;
    int *resume_branch;
    size_t *cut_idx;
    size_t *cut_low;
};
typedef struct pti_TagSplitHandle *pti_SplitHandle;
int pti_StartSplitSparseTensor(pti_SplitHandle *handle, const ptiSparseTensor *tsr, const ptiIndex max_size_by_mode[]);
int pti_SplitSparseTensor(ptiSparseTensor *dest, ptiIndex *inds_low, ptiIndex *inds_high, pti_SplitHandle handle);
void pti_FinishSplitSparseTensor(pti_SplitHandle handle);

typedef struct pti_TagSplitResult {
    ptiSparseTensor tensor;
    ptiIndex *inds_low;
    ptiIndex *inds_high;
    struct pti_TagSplitResult *next;    // Not use now, for one gpu implementation. Now all splits inside one queue has a real subtsr contigously, the length is marked by real_queue_size.
} pti_SplitResult;
/* FIXME: index_limit_by_mode is not used yet */
int pti_SparseTensorGetAllSplits(pti_SplitResult **splits, size_t *nsplits, const ptiSparseTensor *tsr, const size_t nnz_limit_by_mode[], const size_t index_limit_by_mode[], int emit_map);
// void pti_SparseTensorFreeAllSplits(pti_SplitResult *splits);
void pti_SparseTensorFreeAllSplits(pti_SplitResult *splits, size_t const nsplits);
int pti_SparseTensorDumpAllSplits(const pti_SplitResult * splits, ptiIndex const nsplits, FILE *fp);

// abundant
int pti_SparseTensorBalancedSplit(
    pti_SplitResult **splits,
    size_t *nsplits,
    ptiSparseTensor *tsr,
    const size_t nnz_limit,
    const size_t index_limit_by_mode[]);

/* Coarse-grain split */
int pti_ComputeCoarseSplitParameters(
    size_t * split_idx_len,
    size_t const nsplits,
    ptiSparseTensor * const tsr,
    size_t * const slice_nnzs,
    size_t const idx_begin,
    size_t const mode,
    size_t const stride,
    size_t const memwords);

int pti_ComputeCoarseSplitParametersOne(
    size_t * split_idx_len, // size: nsplits
    size_t const nsplits,
    ptiSparseTensor * const tsr,
    size_t * const slice_nnzs,
    size_t const idx_begin,
    size_t const mode,
    size_t const stride,
    size_t const memwords,
    size_t const max_nthreadsx);

int pti_CoarseSplitSparseTensorBatch(
    pti_SplitResult * splits,
    size_t * nnz_split_next,
    size_t * real_nsplits,
    size_t const nsplits,
    size_t * const split_idx_len,
    const size_t mode,
    ptiSparseTensor * tsr,
    size_t const nnz_split_begin);

// abundant
int pti_CoarseSplitSparseTensorAll(
    pti_SplitResult ** splits,
    size_t * nsplits,
    const size_t split_idx_len,
    const size_t mode,
    ptiSparseTensor * tsr);

// abundant
int ptiCoarseSplitSparseTensor(ptiSparseTensor *tsr, const int num, ptiSparseTensor *cstsr);

int pti_CoarseSplitSparseTensorStep(
    pti_SplitResult * splits,
    size_t * nnz_ptr_next,
    const size_t split_idx_len,
    const size_t mode,
    const ptiSparseTensor * tsr,
    const size_t nnz_ptr_begin);


/* Fine-grain split */
int pti_ComputeFineSplitParametersOne(
    size_t * split_nnz_len, // Scalar
    ptiSparseTensor * const tsr,
    size_t const max_nthreadsx);

int pti_ComputeFineSplitParameters(
    size_t * split_nnz_len, // Scalar
    ptiSparseTensor * const tsr,
    size_t const stride,
    size_t const memwords);

int pti_FineSplitSparseTensorBatch(
    pti_SplitResult * splits,
    size_t * nnz_split_next,
    size_t * real_nsplits,
    const size_t nsplits,
    const size_t split_nnz_len,
    ptiSparseTensor * tsr,
    size_t const nnz_split_begin);

int pti_FineSplitSparseTensorStep(
    pti_SplitResult * split,
    size_t * nnz_ptr_next,
    const size_t split_nnz_len,
    ptiSparseTensor * tsr,
    const size_t nnz_ptr_begin);


/* Medium-grain split */
int pti_ComputeMediumSplitParameters(
    size_t * split_idx_len, // size: nmodes
    ptiSparseTensor * const tsr,
    size_t const stride,
    size_t const memwords);

int pti_MediumSplitSparseTensorBatch(
    pti_SplitResult * splits,
    size_t * nnz_split_next,
    size_t * real_nsplits,
    size_t const nsplits,
    size_t * const split_idx_lens,
    ptiSparseTensor * tsr,
    size_t const nnz_split_begin,
    size_t * est_inds_low,
    size_t * est_inds_high);

int pti_MediumSplitSparseTensorStep(    // In-place
    pti_SplitResult * split,
    size_t * nnz_ptr_next,
    size_t * subnnz,
    size_t * const split_idx_lens,
    ptiSparseTensor * tsr,
    const size_t nnz_ptr_begin,
    size_t * const est_inds_low,
    size_t * const est_inds_high);



#ifdef __cplusplus
}
#endif

#endif
