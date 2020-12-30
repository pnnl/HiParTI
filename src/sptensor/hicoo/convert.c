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
#include "../sptensor.h"
#include "hicoo.h"

/**
 * Compare two coordinates, in the order of mode-0,...,N-1. One from the sparse tensor, the other is specified.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z > item; otherwise, 0.
 */
static int ptiLargerThanCoordinates(
    ptiSparseTensor *tsr,
    const ptiNnzIndex z,
    const ptiIndex * item)
{
    ptiIndex nmodes = tsr->nmodes;
    ptiIndex i1, i2;

    for(ptiIndex m=0; m<nmodes; ++m) {
        i1 = tsr->inds[m].data[z];
        i2 = item[m];
        if(i1 > i2) {
            return 1;
            break;
        }
    }
    return 0;
}


/**
 * Compare two coordinates, in the order of mode-0,...,N-1. One from the sparse tensor, the other is specified.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z < item; otherwise, 0.
 */
static int ptiSmallerThanCoordinates(
    ptiSparseTensor *tsr,
    const ptiNnzIndex z,
    const ptiIndex * item)
{
    ptiIndex nmodes = tsr->nmodes;
    ptiIndex i1, i2;

    for(ptiIndex m=0; m<nmodes; ++m) {
        i1 = tsr->inds[m].data[z];
        i2 = item[m];
        if(i1 < i2) {
            return 1;
            break;
        }
    }
    return 0;
}


/**
 * Compare two coordinates, in the order of mode-0,...,N-1. One from the sparse tensor, the other is specified.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z = item; otherwise, 0.
 */
static int ptiEqualWithCoordinates(
    ptiSparseTensor *tsr,
    const ptiNnzIndex z,
    const ptiIndex * item)
{
    ptiIndex nmodes = tsr->nmodes;
    ptiIndex i1, i2;

    for(ptiIndex m=0; m<nmodes; ++m) {
        i1 = tsr->inds[m].data[z];
        i2 = item[m];
        if(i1 != i2) {
            return 0;
            break;
        }
    }
    return 1;
}


/**
 * Compare two specified coordinates.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z == item; otherwise, 0.
 */
static int ptiEqualWithTwoCoordinates(
    const ptiIndex * item1,
    const ptiIndex * item2,
    const ptiIndex nmodes)
{
    ptiIndex i1, i2;
    for(ptiIndex m=0; m<nmodes; ++m) {
        i1 = item1[m];
        i2 = item2[m];
        if(i1 != i2) {
            return 0;
            break;
        }
    }
    return 1;
}

/**
 * Check if a nonzero item is in the range of two given coordinates, in the order of mode-0,...,N-1. 
 * @param tsr    a pointer to a sparse tensor
 * @return      1, yes; 0, no.
 */
static int ptiCoordinatesInRange(
    ptiSparseTensor *tsr,
    const ptiNnzIndex z,
    const ptiIndex * range_begin,
    const ptiIndex * range_end)
{
    if ( (ptiLargerThanCoordinates(tsr, z, range_begin) == 1 ||
        ptiEqualWithCoordinates(tsr, z, range_begin) == 1) &&
        ptiSmallerThanCoordinates(tsr, z, range_end) == 1) {
        return 1;
    }
    return 0;
}

/**
 * Compute the beginning of the next block
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of the next block
 */
static int ptiNextBlockBegin(
    ptiIndex * out_item,
    ptiSparseTensor *tsr,
    const ptiIndex * in_item,
    const ptiElementIndex sb)
{
    ptiIndex nmodes = tsr->nmodes;

    for(int32_t m=nmodes-1; m>=0; --m) {
        if(in_item[m] < tsr->ndims[m]-1) {
            out_item[m] = in_item[m]+sb-1 < tsr->ndims[m] ? in_item[m]+sb-1 : tsr->ndims[m] - 1;
            break;
        }
    }

    return 0;
}


/**
 * Compute the end of this block
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the end indices of this block
 */
static int ptiBlockEnd(
    ptiIndex * out_item,
    ptiSparseTensor *tsr,
    const ptiIndex * in_item,
    const ptiElementIndex sb)
{
    ptiIndex nmodes = tsr->nmodes;

    for(ptiIndex m=0; m<nmodes; ++m) {
        ptiAssert(in_item[m] < tsr->ndims[m]);
        out_item[m] = in_item[m]+sb < tsr->ndims[m] ? in_item[m]+sb : tsr->ndims[m];    // exclusive
    }

    return 0;
}


/**
 * Locate the beginning of the block/kernel containing the coordinates
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of this block
 */
static int ptiLocateBeginCoord(
    ptiIndex * out_item,
    ptiSparseTensor *tsr,
    const ptiIndex * in_item,
    const ptiElementIndex bits)
{
    ptiIndex nmodes = tsr->nmodes;
    
    for(ptiIndex m=0; m<nmodes; ++m) {
        out_item[m] = in_item[m] >> bits;
    }

    return 0;
}


/**
 * Compute the strides for kernels, mode order N-1, ..., 0 (row-like major)
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of this block
 */
static int ptiKernelStrides(
    ptiIndex * strides,
    ptiSparseTensor *tsr,
    const ptiIndex sk)
{
    ptiIndex nmodes = tsr->nmodes;
    ptiIndex kernel_size = 0;
    
    // TODO: efficiently use bitwise operation
    strides[nmodes-1] = 1;
    for(ptiIndex m=nmodes-2; m>=1; --m) {
        kernel_size = (ptiIndex)(tsr->ndims[m+1] + sk - 1) / sk;
        strides[m] = strides[m+1] * kernel_size;
    }
    kernel_size = (ptiIndex)(tsr->ndims[1] + sk - 1) / sk;
    strides[0] = strides[1] * kernel_size;

    return 0;
}





/**
 * Compute the end of this kernel
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the end indices of this block
 */
static int ptiKernelEnd(
    ptiIndex * out_item,
    ptiSparseTensor *tsr,
    const ptiIndex * in_item,
    const ptiIndex sk)
{
    ptiIndex nmodes = tsr->nmodes;

    for(ptiIndex m=0; m<nmodes; ++m) {
        ptiAssert(in_item[m] < tsr->ndims[m]);
        out_item[m] = in_item[m]+sk < tsr->ndims[m] ? in_item[m]+sk : tsr->ndims[m];    // exclusive
    }

    return 0;
}



/**
 * Record mode pointers for kernel rows, from a sorted tensor.
 * @param mptr  a vector of pointers as a dense array
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int ptiGetRowBlockPointers(
    ptiNnzIndexVector *mptr,
    ptiSparseTensor *tsr,
    const ptiIndex sk)
{
    ptiNnzIndex nnz = tsr->nnz;
    ptiIndex i = tsr->inds[0].data[0];
    ptiNnzIndex k = 0;  // count blocks
    ptiNnzIndex knnz = 0;   // #Nonzeros per block
    mptr->data[0] = 0;
    while(1) {
        /* check if mode-0 index in block-b */
        if(i >= sk * k && i < sk * (k+1)) {
            ++ knnz;
            break;
        } else {
            ++ k;
            mptr->data[k] = knnz + mptr->data[k-1];
            knnz = 0;
        }
    }

    
    for(ptiNnzIndex z=1; z<nnz; ++z) {
        i = tsr->inds[0].data[z];
        /* Compare with the next block row index */
        while(1) {
            if(i >= sk * k && i < sk * (k+1)) {
                ++ knnz;
                break;
            } else {
                ++ k;
                mptr->data[k] = knnz + mptr->data[k-1];
                knnz = 0;
            }
        }
    }
    ptiAssert(k < (tsr->ndims[0] + sk -1 ) / sk);
    ptiAssert(mptr->data[mptr->len-1] + knnz == nnz);

    return 0;
}


/**
 * Record mode pointers for kernel rows, from a sorted tensor.
 * @param kptr  a vector of kernel pointers
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int ptiSetKernelPointers(
    ptiNnzIndexVector *kptr,
    ptiNnzIndexVector *knnzs,
    ptiSparseTensor *tsr,
    const ptiElementIndex sk_bits)
{
    ptiIndex nmodes = tsr->nmodes;
    ptiNnzIndex nnz = tsr->nnz;
    ptiNnzIndex k = 0;  // count kernels
    ptiNnzIndex knnz = 0;   // #Nonzeros per kernel
    int result = 0;
    result = ptiAppendNnzIndexVector(kptr, 0);
    pti_CheckError(result, "HiSpTns Convert", NULL);

    ptiIndex * coord = (ptiIndex *)malloc(nmodes * sizeof(*coord));
    ptiIndex * kernel_coord = (ptiIndex *)malloc(nmodes * sizeof(*kernel_coord));
    ptiIndex * kernel_coord_prior = (ptiIndex *)malloc(nmodes * sizeof(*kernel_coord_prior));

    /* Process first nnz to get the first kernel_coord_prior */
    for(ptiIndex m=0; m<nmodes; ++m)
        coord[m] = tsr->inds[m].data[0];    // first nonzero indices
    result = ptiLocateBeginCoord(kernel_coord_prior, tsr, coord, sk_bits);
    pti_CheckError(result, "HiSpTns Convert", NULL);

    for(ptiNnzIndex z=0; z<nnz; ++z) {
        for(ptiIndex m=0; m<nmodes; ++m)
            coord[m] = tsr->inds[m].data[z];
        result = ptiLocateBeginCoord(kernel_coord, tsr, coord, sk_bits);
        pti_CheckError(result, "HiSpTns Convert", NULL);

        if(ptiEqualWithTwoCoordinates(kernel_coord, kernel_coord_prior, nmodes) == 1) {
            ++ knnz;
        } else {
            ++ k;
            result = ptiAppendNnzIndexVector(kptr, knnz + kptr->data[k-1]);
            pti_CheckError(result, "HiSpTns Convert", NULL);
            result = ptiAppendNnzIndexVector(knnzs, knnz);
            pti_CheckError(result, "HiSpTns Convert", NULL);
            for(ptiIndex m=0; m<nmodes; ++m)
                kernel_coord_prior[m] = kernel_coord[m];
            knnz = 1;
        }
    }
    ptiAssert(k < kptr->len);
    ptiAssert(kptr->data[kptr->len-1] + knnz == nnz);

    /* Set the last element for kptr */
    ptiAppendNnzIndexVector(kptr, nnz);
    ptiAppendNnzIndexVector(knnzs, knnz);

    free(coord);
    free(kernel_coord);
    free(kernel_coord_prior);

    return 0;
}


/**
 * Set scheduler for kernels.
 * @param kschr  nmodes kernel schedulers.
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
static int ptiSetKernelScheduler(
    ptiIndexVector **kschr,
    ptiIndex *nkiters,
    ptiNnzIndexVector * const kptr,
    ptiSparseTensor *tsr,
    const ptiElementIndex sk_bits)
{
    ptiIndex nmodes = tsr->nmodes;
    ptiIndex * ndims = tsr->ndims;
    int result = 0;

    ptiIndex * coord = (ptiIndex *)malloc(nmodes * sizeof(*coord));
    ptiIndex * kernel_coord = (ptiIndex *)malloc(nmodes * sizeof(*kernel_coord));

    for(ptiNnzIndex k=0; k<kptr->len - 1; ++k) {
        ptiNnzIndex z = kptr->data[k];
        for(ptiIndex m=0; m<nmodes; ++m)
            coord[m] = tsr->inds[m].data[z];
        result = ptiLocateBeginCoord(kernel_coord, tsr, coord, sk_bits);
        pti_CheckError(result, "HiSpTns Convert", NULL);

        for(ptiIndex m=0; m<nmodes; ++m) {
            result = ptiAppendIndexVector(&(kschr[m][kernel_coord[m]]), k);
            pti_CheckError(result, "HiSpTns Convert", NULL);
        }
    }

    free(coord);
    free(kernel_coord);

    ptiIndex sk = (ptiIndex)pow(2, sk_bits);
    ptiIndex tmp;
    for(ptiIndex m=0; m<nmodes; ++m) {
        tmp = 0;
        ptiIndex kernel_ndim = (ndims[m] + sk - 1) / sk;
        for(ptiIndex i=0; i<kernel_ndim; ++i) {
            if(tmp < kschr[m][i].len)
                tmp = kschr[m][i].len;
        }
        nkiters[m] = tmp;
    }

    return 0;
}



/**
 * Pre-process COO sparse tensor by permuting, sorting, and record pointers to blocked rows. Kernels in Row-major order, blocks and elements are in Z-Morton order.
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int ptiPreprocessSparseTensor(
    ptiNnzIndexVector * kptr,
    ptiIndexVector **kschr,
    ptiIndex *nkiters,
    ptiIndexVector **kschr_balanced,
    ptiIndexVector **kschr_balanced_pos,
    ptiIndex *nkpars,
    ptiIndexVector * kschr_rest,
    ptiNnzIndexVector * knnzs,
    ptiSparseTensor *tsr,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits,
    int const tk)
{
    ptiNnzIndex nnz = tsr->nnz;
    int result;

    // TODO: possible permute modes to improve parallelism

    /* Sort tsr in a Row-major Block order to get all kernels. Not use Morton-order for kernels: 1. better support for higher-order tensors by limiting kernel size, because Morton key bit <= 128; */
    ptiTimer rowblock_sort_timer;
    ptiNewTimer(&rowblock_sort_timer, 0);
    ptiStartTimer(rowblock_sort_timer);

    ptiSparseTensorSortIndexRowBlock(tsr, 1, 0, nnz, sk_bits, tk);  // Parallelized inside

    ptiStopTimer(rowblock_sort_timer);
    ptiPrintElapsedTime(rowblock_sort_timer, "\t\trowblock sorting");
    ptiFreeTimer(rowblock_sort_timer);
#if PARTI_DEBUG == 3
    printf("Sorted by ptiSparseTensorSortIndexRowBlock.\n");
    ptiAssert(ptiDumpSparseTensor(tsr, 0, stdout) == 0);
#endif

    ptiTimer set_kernel_timer;
    ptiNewTimer(&set_kernel_timer, 0);
    ptiStartTimer(set_kernel_timer);

    result = ptiSetKernelPointers(kptr, knnzs, tsr, sk_bits);
    pti_CheckError(result, "HiSpTns Preprocess", NULL);
    result = ptiSetKernelScheduler(kschr, nkiters, kptr, tsr, sk_bits);
    pti_CheckError(result, "HiSpTns Preprocess", NULL);
    // printf("OK\n"); fflush(stdout);

    /* Set balanced data structures: kschr_balanced, kschr_rest */
    ptiNnzIndex avg_nnzk = tsr->nnz / (kptr->len - 1);
    ptiNnzIndex max_nnzk = 0;
    for(ptiIndex k=0; k<kptr->len - 1; ++k) {
        ptiNnzIndex nnzk = knnzs->data[k];
        if(max_nnzk < nnzk) max_nnzk = nnzk;
    }
    // ptiNnzIndex par_nnzk_th = 20 * avg_nnzk; // threshold for nnzk per thread
    ptiNnzIndex par_nnzk_th = 5 * max_nnzk; // threshold for nnzk per thread
    printf("par_nnzk_th: %lu\n", par_nnzk_th);
    ptiIndex sk = (ptiIndex)pow(2, sk_bits);
    // printf("OK-2\n"); fflush(stdout);

    for(ptiIndex m=0; m < tsr->nmodes; ++m) {   // Loop kschr for each mode

        ptiIndexVector * restrict kschr_mode = kschr[m];
        ptiIndexVector * restrict kschr_balanced_mode = kschr_balanced[m];
        ptiIndexVector * restrict kschr_balanced_pos_mode = kschr_balanced_pos[m];
        ptiIndex kernel_ndim = (tsr->ndims[m] + sk - 1)/sk;
        for(ptiIndex i=0; i < kernel_ndim; ++i) {
            ptiAppendIndexVector(&(kschr_balanced_pos_mode[i]), 0);
        }
        ptiIndex j_rest = nkiters[m];
        ptiIndex npars = 0;
        int tag_rest = 0;
        ptiIndex count_nk = 0;
        ptiIndex empty_schr_rows_th = 1.0 * kernel_ndim > 1 ? 1.0 * kernel_ndim : 1;
        printf("[mode %u] empty_schr_rows_th: %u\n", m, empty_schr_rows_th);

        while(tag_rest == 0 && count_nk < kptr->len - 1) {  // Loop for partitions. tag_rest = 1, maybe there is no rest.
            /* Check two ranges: npars and j or tmp_j !!! */
            ptiIndex max_nnzk_per_col = 0, par_nnzk = 0;
            ptiIndex count_empty_schr_rows = 0;
            for(ptiIndex i=0; i < kernel_ndim; ++i) {   // Find the max nnzk
                if(count_empty_schr_rows > empty_schr_rows_th) {
                    tag_rest = 1;
                    break;
                }
                if(npars >= kschr_balanced_pos_mode[i].len) {
                    ++ count_empty_schr_rows;
                    continue;
                } else {
                    ptiIndex j = kschr_balanced_pos_mode[i].data[npars];
                    if(j >= kschr_mode[i].len) {
                        ++ count_empty_schr_rows;
                        continue;
                    }
                    ptiIndex kernel_num = kschr_mode[i].data[j];
                    ptiNnzIndex kernel_nnz = knnzs->data[kernel_num];
                    if (max_nnzk_per_col < kernel_nnz) {
                        max_nnzk_per_col = kernel_nnz;
                    }
                }
            }   // End of i

            if(tag_rest == 1) {   // an empty superblock met, to kschr_rest
                for(ptiIndex i=0; i < kernel_ndim; ++i) {
                    if(npars >= kschr_balanced_pos_mode[i].len) continue;
                    ptiIndex j2 = kschr_balanced_pos_mode[i].data[npars];
                    for(; j2 < kschr_mode[i].len; ++j2) {
                        ptiAppendIndexVector(&kschr_rest[m], kschr_mode[i].data[j2]);
                        ++ count_nk;
                    }
                }
            } else {    // all non-empty superblocks for this column, to kschr_balanced, kschr_balanced_pos
                /* set par_nnzk */
                if(max_nnzk_per_col > par_nnzk_th) {
                    par_nnzk = max_nnzk_per_col;    // split according to the superblock with the max nnzk
                } else {
                    par_nnzk = par_nnzk_th;
                }

                /* Real partition */
                for(ptiIndex i=0; i < kernel_ndim; ++i) {
                    if(npars >= kschr_balanced_pos_mode[i].len) continue;
                    ptiIndex tmp_j = kschr_balanced_pos_mode[i].data[npars];
                    if(tmp_j >= kschr_mode[i].len) continue;
                    ptiIndex kernel_num = kschr_mode[i].data[tmp_j];
                    ptiNnzIndex sum_nnzk = knnzs->data[kernel_num];
                    while(sum_nnzk <= par_nnzk) {
                        ptiAppendIndexVector(&(kschr_balanced_mode[i]), kernel_num);
                        ++ count_nk;
                        ++ tmp_j;
                        if(tmp_j < kschr_mode[i].len) {
                            kernel_num = kschr_mode[i].data[tmp_j]; // j + 1
                            sum_nnzk += knnzs->data[kernel_num];
                        } else {
                            break;
                        }
                    }   // End of while
                    ptiAppendIndexVector(&(kschr_balanced_pos_mode[i]), tmp_j);
                }
                ++ npars;
            }
            // printf("count_nk: %u\n", count_nk); fflush(stdout);
        }   // End of while
        nkpars[m] = npars;  // kschr_balanced_pos.len is npars + 1.
    }   // End loop of modes


    ptiStopTimer(set_kernel_timer);
    ptiPrintElapsedTime(set_kernel_timer, "\t\tSet Kernel Ptrs");
    ptiFreeTimer(set_kernel_timer);

    ptiTimer morton_sort_timer;
    ptiNewTimer(&morton_sort_timer, 0);
    ptiStartTimer(morton_sort_timer);

    /* Sort blocks in each kernel in Morton-order */
    ptiNnzIndex k_begin, k_end;
    /* Loop for all kernels, 0-kptr.len for OMP code */
    #pragma omp parallel for num_threads(tk) 
    for(ptiNnzIndex k=0; k<kptr->len - 1; ++k) {
        k_begin = kptr->data[k];
        k_end = kptr->data[k+1];   // exclusive
        /* Sort blocks in each kernel in Morton-order */
        ptiSparseTensorSortIndexMorton(tsr, 1, k_begin, k_end, sb_bits, tk);
        // ptiSparseTensorSortIndexRowBlock(tsr, 1, k_begin, k_end, sb_bits, tk);
#if PARTI_DEBUG == 3
    printf("Kernel %"HIPARTI_PRI_NNZ_INDEX ": Sorted by ptiSparseTensorSortIndexMorton.\n", k);
    ptiAssert(ptiDumpSparseTensor(tsr, 0, stdout) == 0);
#endif
    }

    ptiStopTimer(morton_sort_timer);
    ptiPrintElapsedTime(morton_sort_timer, "\t\tMorton sorting");
    // ptiPrintElapsedTime(morton_sort_timer, "\t\t2nd Rowblock sorting");
    ptiFreeTimer(morton_sort_timer);

    return 0;
}


/**
 * Pre-process COO sparse tensor by permuting, sorting, and record pointers to blocked rows for TTM. Kernels, blocks are both in row-major order, elements in a block is in an arbitrary order.
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int ptiPreprocessSparseTensor_RowBlock(
    ptiNnzIndexVector * kptr,
    ptiIndexVector **kschr,
    ptiIndex *nkiters,
    ptiIndex *nfibs,
    ptiNnzIndexVector * knnzs,
    ptiSparseTensor *tsr,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits,
    int const tk)
{
    ptiNnzIndex nnz = tsr->nnz;
    int result;

    /* Sort tsr in a Row-major Block order to get all kernels. */
    ptiSparseTensorSortIndexRowBlock(tsr, 1, 0, nnz, sk_bits, tk);
    result = ptiSetKernelPointers(kptr, knnzs, tsr, sk_bits);
    pti_CheckError(result, "HiSpTns Preprocess", NULL);
    // result = ptiSetKernelScheduler(kschr, nkiters, kptr, tsr, sk_bits);
    // pti_CheckError(result, "HiSpTns Preprocess", NULL);

    /* Sort blocks in each kernel in Row-major block order. */
    ptiNnzIndex k_begin, k_end;
    /* Loop for all kernels, 0-kptr.len for OMP code */
    for(ptiNnzIndex k=0; k<kptr->len - 1; ++k) {
        k_begin = kptr->data[k];
        k_end = kptr->data[k+1];   // exclusive
        ptiSparseTensorSortIndexRowBlock(tsr, 1, k_begin, k_end, sb_bits, tk);
    }

    return 0;
}


int ptiSparseTensorToHiCOO(
    ptiSparseTensorHiCOO *hitsr,
    ptiNnzIndex *max_nnzb,
    ptiSparseTensor *tsr,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits,
    const ptiElementIndex sc_bits,
    int const tk)
{
    ptiAssert(sk_bits >= sb_bits);
    ptiAssert(sc_bits >= sb_bits);

    ptiIndex i;
    int result;
    ptiIndex nmodes = tsr->nmodes;
    ptiNnzIndex nnz = tsr->nnz;

    ptiElementIndex sb = pow(2, sb_bits);
    ptiIndex sc = pow(2, sc_bits);

    /* Set HiCOO parameters. ndims for type conversion, size_t -> ptiIndex */
    ptiIndex * ndims = malloc(nmodes * sizeof *ndims);
    pti_CheckOSError(!ndims, "HiSpTns Convert");
    for(i = 0; i < nmodes; ++i) {
        ndims[i] = (ptiIndex)tsr->ndims[i];
    }

    result = ptiNewSparseTensorHiCOO(hitsr, (ptiIndex)tsr->nmodes, ndims, (ptiNnzIndex)tsr->nnz, sb_bits, sk_bits, sc_bits);
    pti_CheckError(result, "HiSpTns Convert", NULL);

    /* Pre-process tensor to get hitsr->kptr, values are nonzero locations. */
    ptiTimer sort_timer;
    ptiNewTimer(&sort_timer, 0);
    ptiStartTimer(sort_timer);

    ptiPreprocessSparseTensor(&hitsr->kptr, hitsr->kschr, hitsr->nkiters, hitsr->kschr_balanced, hitsr->kschr_balanced_pos, hitsr->nkpars, hitsr->kschr_rest, &hitsr->knnzs, tsr, sb_bits, sk_bits, tk);

    ptiStopTimer(sort_timer);
    ptiPrintElapsedTime(sort_timer, "\tHiCOO sorting (rowblock + morton)");
    ptiFreeTimer(sort_timer);
#if PARTI_DEBUG >= 2
    printf("Kernels: Row-major, blocks: Morton-order sorted:\n");
    ptiAssert(ptiDumpSparseTensor(tsr, 0, stdout) == 0);
    printf("hitsr->kptr:\n");
    ptiDumpNnzIndexVector(&hitsr->kptr, stdout);
#endif

    ptiTimer gen_timer;
    ptiNewTimer(&gen_timer, 0);
    ptiStartTimer(gen_timer);

    /* Temporary storage */
    ptiIndex * block_begin = (ptiIndex *)malloc(nmodes * sizeof(*block_begin));
    ptiIndex * block_end = (ptiIndex *)malloc(nmodes * sizeof(*block_end));
    ptiIndex * block_begin_prior = (ptiIndex *)malloc(nmodes * sizeof(*block_begin_prior));
    ptiIndex * block_coord = (ptiIndex *)malloc(nmodes * sizeof(*block_coord));

    ptiNnzIndex k_begin, k_end; // #Nonzeros locations
    ptiNnzIndex nk = 0; // #Kernels
    ptiNnzIndex nc = 0; // #Chunks
    ptiNnzIndex nb = 1; // #Blocks  // counting from the first nnz
    ptiNnzIndex nb_tmp = 0;
    ptiNnzIndex ne = 0; // #Nonzeros per block
    ptiIndex eindex = 0;
    ptiBlockIndex chunk_size = 0;

    /* different appending methods:
     * elements: append every nonzero entry
     * blocks: append when seeing a new block.
     * chunks: appending when seeting a new chunk. Notice the boundary of kernels and the last chunk of the whole tensor may be larger than the sc.
     * kernels: append when seeing a new kernel. Not appending a vector, just write data into an allocated array.
     */
    /* Process first nnz */
    for(ptiIndex m=0; m<nmodes; ++m)
        block_coord[m] = tsr->inds[m].data[0];    // first nonzero indices
    result = ptiLocateBeginCoord(block_begin_prior, tsr, block_coord, sb_bits);
    pti_CheckError(result, "HiSpTns Convert", NULL);
    for(ptiIndex m=0; m<nmodes; ++m)
        ptiAppendBlockIndexVector(&hitsr->binds[m], (ptiBlockIndex)block_begin_prior[m]);
    ptiAppendNnzIndexVector(&hitsr->bptr, 0);


    /* Loop for all kernels, 0 - hitsr->kptr.len - 1 for OMP code */
    for(ptiNnzIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        k_begin = hitsr->kptr.data[k];
        k_end = hitsr->kptr.data[k+1]; // exclusive
        nb_tmp = k == 0 ? 0: nb;
        /* Modify kptr pointing to block locations */
        hitsr->kptr.data[k] = nb_tmp;
        ++ nk;

        /* Only append a chunk for the new kernel, the last chunk in the old kernel may be larger than sc */
        ptiAppendNnzIndexVector(&hitsr->cptr, nb_tmp);
        // printf("cptr 1:\n");
        // ptiDumpNnzIndexVector(&hitsr->cptr, stdout);
        ++ nc;
        chunk_size = 0;

        /* Loop nonzeros in each kernel */
        for(ptiNnzIndex z = k_begin; z < k_end; ++z) {
            #if PARTI_DEBUG == 5
                printf("z: %"HIPARTI_PRI_NNZ_INDEX "\n", z);
            #endif

            for(ptiIndex m=0; m<nmodes; ++m)
                block_coord[m] = tsr->inds[m].data[z];    // first nonzero indices
            #if PARTI_DEBUG == 5
                printf("block_coord:\n");
                ptiAssert(ptiDumpIndexArray(block_coord, nmodes, stdout) == 0);
            #endif

            result = ptiLocateBeginCoord(block_begin, tsr, block_coord, sb_bits);
            // pti_CheckError(result, "HiSpTns Convert", NULL);
            #if PARTI_DEBUG == 5
                printf("block_begin_prior:\n");
                ptiAssert(ptiDumpIndexArray(block_begin_prior, nmodes, stdout) == 0);
                printf("block_begin:\n");
                ptiAssert(ptiDumpIndexArray(block_begin, nmodes, stdout) == 0);
            #endif

            result = ptiBlockEnd(block_end, tsr, block_begin, sb);  // exclusive
            // pti_CheckError(result, "HiSpTns Convert", NULL);

            /* Append einds and values */
            for(ptiIndex m=0; m<nmodes; ++m) {
                eindex = tsr->inds[m].data[z] < (block_begin[m] << sb_bits) ? tsr->inds[m].data[z] : tsr->inds[m].data[z] - (block_begin[m] << sb_bits);
                ptiAssert(eindex < sb);
                ptiAppendElementIndexVector(&hitsr->einds[m], (ptiElementIndex)eindex);
            }
            ptiAppendValueVector(&hitsr->values, tsr->values.data[z]);


            /* z in the same block with last z */
            if (ptiEqualWithTwoCoordinates(block_begin, block_begin_prior, nmodes) == 1)
            {
                /* ne: #Elements in current block */
                ++ ne;
            } else { /* New block */
                /* ne: #Elements in the last block */
                /* Append block bptr and bidx */
                ptiAppendNnzIndexVector(&hitsr->bptr, (ptiBlockIndex)z);
                for(ptiIndex m=0; m<nmodes; ++m)
                    ptiAppendBlockIndexVector(&hitsr->binds[m], (ptiBlockIndex)block_begin[m]);
                for(ptiIndex m=0; m<nmodes; ++m)
                    block_begin_prior[m] = block_begin[m];

                /* ne: old block's number of nonzeros */
                // if(chunk_size + ne > sc || ne >= sc) { 
                // if(chunk_size + ne >= sc && chunk_size > 0) {    // calculate the prior block
                //     /* Append a chunk ending by the old block */
                //     ptiAppendNnzIndexVector(&hitsr->cptr, nb-1);
                //     // printf("cptr 2:\n");
                //     // ptiDumpNnzIndexVector(&hitsr->cptr, stdout);
                //     ++ nc;
                //     chunk_size = ne;
                // } else {
                //     chunk_size += ne;
                // }

                if(chunk_size + ne >= sc) {    // calculate the prior block
                    /* Append a chunk ending by the old block */
                    ptiAppendNnzIndexVector(&hitsr->cptr, nb);
                    // printf("cptr 2:\n");
                    // ptiDumpNnzIndexVector(&hitsr->cptr, stdout);
                    // printf("nb: %u, chunk_size: %u, ne: %u\n", nb, chunk_size, ne);
                    ++ nc;
                    chunk_size = 0;
                } else {
                    chunk_size += ne;
                }

                ++ nb;
                ne = 1;              
            } // End new block
            #if PARTI_DEBUG == 5
                printf("nk: %u, nc: %u, nb: %u, ne: %u, chunk_size: %lu\n\n", nk, nc, nb, ne, chunk_size);
            #endif

        }   // End z loop
        
    }   // End k loop
    ptiAssert(nb <= nnz);
    ptiAssert(nb == hitsr->binds[0].len);
    // ptiAssert(nc <= nb);
    ptiAssert(nk == hitsr->kptr.len - 1);

    /* Last element for kptr, cptr, bptr */
    hitsr->kptr.data[hitsr->kptr.len - 1] = hitsr->bptr.len;
    ptiAppendNnzIndexVector(&hitsr->cptr, hitsr->bptr.len);
    ptiAppendNnzIndexVector(&hitsr->bptr, nnz);


    *max_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
    ptiNnzIndex sum_nnzb = 0;
    for(ptiIndex i=0; i < hitsr->bptr.len - 1; ++i) {
        ptiNnzIndex nnzb = hitsr->bptr.data[i+1] - hitsr->bptr.data[i];
        sum_nnzb += nnzb;
        if(*max_nnzb < nnzb) {
          *max_nnzb = nnzb;
        }
    }
    ptiAssert(sum_nnzb == hitsr->nnz);

    ptiStopTimer(gen_timer);
    ptiPrintElapsedTime(gen_timer, "\tGenerate HiCOO");
    ptiFreeTimer(gen_timer);


    free(block_begin);
    free(block_end);
    free(block_begin_prior);
    free(block_coord);

	return 0;
}
