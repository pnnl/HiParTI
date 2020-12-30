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

#include <ParTI.h>
#include "../sptensor.h"
#include "hicoo.h"


/**
 * Compare two specified coordinates.
 * @param tsr    a pointer to a sparse tensor
 * @return      1, z == item; otherwise, 0.
 */
static int sptEqualWithTwoCoordinates(
    const sptIndex * item1,
    const sptIndex * item2,
    const sptIndex nmodes)
{
    sptIndex i1, i2;
    for(sptIndex m=0; m<nmodes; ++m) {
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
 * Compute the end of this block
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the end indices of this block
 */
static int sptBlockEnd(
    sptIndex * out_item,
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptElementIndex sb)
{
    sptIndex nmodes = tsr->nmodes;

    for(sptIndex m=0; m<nmodes; ++m) {
        sptAssert(in_item[m] < tsr->ndims[m]);
        out_item[m] = in_item[m]+sb < tsr->ndims[m] ? in_item[m]+sb : tsr->ndims[m];    // exclusive
    }

    return 0;
}


/**
 * Locate the beginning of the block/kernel containing the coordinates
 * @param tsr    a pointer to a sparse tensor
 * @return out_item     the beginning indices of this block
 */
static int sptLocateBeginCoord(
    sptIndex * out_item,
    sptSparseTensor *tsr,
    const sptIndex * in_item,
    const sptElementIndex bits)
{
    sptIndex nmodes = tsr->nmodes;
    
    for(sptIndex m=0; m<nmodes; ++m) {
        out_item[m] = in_item[m] >> bits;
    }

    return 0;
}



/**
 * Record mode pointers for kernel rows, from a sorted tensor.
 * @param mptr  a vector of pointers as a dense array
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int sptGetRowBlockPointers(
    sptNnzIndexVector *mptr,
    sptSparseTensor *tsr, 
    const sptIndex sk)
{
    sptNnzIndex nnz = tsr->nnz;
    sptIndex i = tsr->inds[0].data[0];
    sptNnzIndex k = 0;  // count blocks
    sptNnzIndex knnz = 0;   // #Nonzeros per block
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

    
    for(sptNnzIndex z=1; z<nnz; ++z) {
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
    sptAssert(k < (tsr->ndims[0] + sk -1 ) / sk);
    sptAssert(mptr->data[mptr->len-1] + knnz == nnz);

    return 0;
}


/**
 * Record mode pointers for kernel rows, from a sorted tensor.
 * @param kptr  a vector of kernel pointers
 * @param tsr    a pointer to a sparse tensor
 * @return      mode pointers
 */
int sptSetKernelPointers(
    sptNnzIndexVector *kptr,
    sptNnzIndexVector *knnzs,
    sptSparseTensor *tsr, 
    const sptElementIndex sk_bits)
{
    sptIndex nmodes = tsr->nmodes;
    sptNnzIndex nnz = tsr->nnz;
    sptNnzIndex k = 0;  // count kernels
    sptNnzIndex knnz = 0;   // #Nonzeros per kernel
    int result = 0;
    result = sptAppendNnzIndexVector(kptr, 0);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    sptIndex * coord = (sptIndex *)malloc(nmodes * sizeof(*coord));
    sptIndex * kernel_coord = (sptIndex *)malloc(nmodes * sizeof(*kernel_coord));
    sptIndex * kernel_coord_prior = (sptIndex *)malloc(nmodes * sizeof(*kernel_coord_prior));

    /* Process first nnz to get the first kernel_coord_prior */
    for(sptIndex m=0; m<nmodes; ++m) 
        coord[m] = tsr->inds[m].data[0];    // first nonzero indices
    result = sptLocateBeginCoord(kernel_coord_prior, tsr, coord, sk_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    for(sptNnzIndex z=0; z<nnz; ++z) {
        for(sptIndex m=0; m<nmodes; ++m) 
            coord[m] = tsr->inds[m].data[z];
        result = sptLocateBeginCoord(kernel_coord, tsr, coord, sk_bits);
        spt_CheckError(result, "HiSpTns Convert", NULL);

        if(sptEqualWithTwoCoordinates(kernel_coord, kernel_coord_prior, nmodes) == 1) {
            ++ knnz;
        } else {
            ++ k;
            result = sptAppendNnzIndexVector(kptr, knnz + kptr->data[k-1]);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            result = sptAppendNnzIndexVector(knnzs, knnz);
            spt_CheckError(result, "HiSpTns Convert", NULL);
            for(sptIndex m=0; m<nmodes; ++m) 
                kernel_coord_prior[m] = kernel_coord[m];
            knnz = 1;
        }
    }
    sptAssert(k < kptr->len);
    sptAssert(kptr->data[kptr->len-1] + knnz == nnz);

    /* Set the last element for kptr */
    sptAppendNnzIndexVector(kptr, nnz); 
    sptAppendNnzIndexVector(knnzs, knnz);

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
int sptSetKernelScheduler(
    sptIndexVector **kschr,
    sptIndex *nkiters,
    sptNnzIndexVector * const kptr,
    sptSparseTensor *tsr, 
    const sptElementIndex sk_bits)
{
    sptIndex nmodes = tsr->nmodes;
    sptIndex * ndims = tsr->ndims;
    int result = 0;

    sptIndex * coord = (sptIndex *)malloc(nmodes * sizeof(*coord));
    sptIndex * kernel_coord = (sptIndex *)malloc(nmodes * sizeof(*kernel_coord));

    for(sptNnzIndex k=0; k<kptr->len - 1; ++k) {
        sptNnzIndex z = kptr->data[k];
        for(sptIndex m=0; m<nmodes; ++m) 
            coord[m] = tsr->inds[m].data[z];
        result = sptLocateBeginCoord(kernel_coord, tsr, coord, sk_bits);
        spt_CheckError(result, "HiSpTns Convert", NULL);

        for(sptIndex m=0; m<nmodes; ++m) {
            result = sptAppendIndexVector(&(kschr[m][kernel_coord[m]]), k);
            spt_CheckError(result, "HiSpTns Convert", NULL);
        }
    }

    free(coord);
    free(kernel_coord);

    sptIndex sk = (sptIndex)pow(2, sk_bits);
    sptIndex tmp;
    for(sptIndex m=0; m<nmodes; ++m) {
        tmp = 0;
        sptIndex kernel_ndim = (ndims[m] + sk - 1) / sk;
        for(sptIndex i=0; i<kernel_ndim; ++i) {
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
int sptPreprocessSparseTensor(
    sptNnzIndexVector * kptr,
    sptIndexVector **kschr,
    sptIndex *nkiters,
    sptIndexVector **kschr_balanced,
    sptIndexVector **kschr_balanced_pos,
    sptIndex *nkpars,
    sptIndexVector * kschr_rest,
    sptNnzIndexVector * knnzs,
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    int const tk)
{
    sptNnzIndex nnz = tsr->nnz;
    int result;

    // TODO: possible permute modes to improve parallelism

    /* Sort tsr in a Row-major Block order to get all kernels. Not use Morton-order for kernels: 1. better support for higher-order tensors by limiting kernel size, because Morton key bit <= 128; */
    sptTimer rowblock_sort_timer;
    sptNewTimer(&rowblock_sort_timer, 0);
    sptStartTimer(rowblock_sort_timer);

    sptSparseTensorSortIndexRowBlock(tsr, 1, 0, nnz, sk_bits, tk);  // Parallelized inside

    sptStopTimer(rowblock_sort_timer);
    sptPrintElapsedTime(rowblock_sort_timer, "\t\trowblock sorting");
    sptFreeTimer(rowblock_sort_timer);
#if PARTI_DEBUG == 3
    printf("Sorted by sptSparseTensorSortIndexRowBlock.\n");
    sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);
#endif

    sptTimer set_kernel_timer;
    sptNewTimer(&set_kernel_timer, 0);
    sptStartTimer(set_kernel_timer);

    result = sptSetKernelPointers(kptr, knnzs, tsr, sk_bits);
    spt_CheckError(result, "HiSpTns Preprocess", NULL);
    result = sptSetKernelScheduler(kschr, nkiters, kptr, tsr, sk_bits);
    spt_CheckError(result, "HiSpTns Preprocess", NULL);
    // printf("OK\n"); fflush(stdout);

    /* Set balanced data structures: kschr_balanced, kschr_rest */
    // sptNnzIndex avg_nnzk = tsr->nnz / (kptr->len - 1);
    sptNnzIndex max_nnzk = 0;
    for(sptIndex k=0; k<kptr->len - 1; ++k) {
        sptNnzIndex nnzk = knnzs->data[k];
        if(max_nnzk < nnzk) max_nnzk = nnzk;
    }
    // sptNnzIndex par_nnzk_th = 20 * avg_nnzk; // threshold for nnzk per thread
    sptNnzIndex par_nnzk_th = 5 * max_nnzk; // threshold for nnzk per thread
    printf("par_nnzk_th: %lu\n", par_nnzk_th);
    sptIndex sk = (sptIndex)pow(2, sk_bits);
    // printf("OK-2\n"); fflush(stdout);

    for(sptIndex m=0; m < tsr->nmodes; ++m) {   // Loop kschr for each mode

        sptIndexVector * restrict kschr_mode = kschr[m];
        sptIndexVector * restrict kschr_balanced_mode = kschr_balanced[m];
        sptIndexVector * restrict kschr_balanced_pos_mode = kschr_balanced_pos[m];
        sptIndex kernel_ndim = (tsr->ndims[m] + sk - 1)/sk;
        for(sptIndex i=0; i < kernel_ndim; ++i) {
            sptAppendIndexVector(&(kschr_balanced_pos_mode[i]), 0);
        }
        // sptIndex j_rest = nkiters[m];
        sptIndex npars = 0;
        int tag_rest = 0;
        sptIndex count_nk = 0;
        sptIndex empty_schr_rows_th = 1.0 * kernel_ndim > 1 ? 1.0 * kernel_ndim : 1;
        printf("[mode %u] empty_schr_rows_th: %u\n", m, empty_schr_rows_th);

        while(tag_rest == 0 && count_nk < kptr->len - 1) {  // Loop for partitions. tag_rest = 1, maybe there is no rest.
            /* Check two ranges: npars and j or tmp_j !!! */
            sptIndex max_nnzk_per_col = 0, par_nnzk = 0;
            sptIndex count_empty_schr_rows = 0;
            for(sptIndex i=0; i < kernel_ndim; ++i) {   // Find the max nnzk
                if(count_empty_schr_rows > empty_schr_rows_th) {
                    tag_rest = 1;
                    break;
                }
                if(npars >= kschr_balanced_pos_mode[i].len) {
                    ++ count_empty_schr_rows;
                    continue;
                } else {
                    sptIndex j = kschr_balanced_pos_mode[i].data[npars];
                    if(j >= kschr_mode[i].len) {
                        ++ count_empty_schr_rows;
                        continue;
                    }
                    sptIndex kernel_num = kschr_mode[i].data[j];
                    sptNnzIndex kernel_nnz = knnzs->data[kernel_num];
                    if (max_nnzk_per_col < kernel_nnz) {
                        max_nnzk_per_col = kernel_nnz;
                    }
                }
            }   // End of i

            if(tag_rest == 1) {   // an empty superblock met, to kschr_rest
                for(sptIndex i=0; i < kernel_ndim; ++i) {
                    if(npars >= kschr_balanced_pos_mode[i].len) continue;
                    sptIndex j2 = kschr_balanced_pos_mode[i].data[npars];
                    for(; j2 < kschr_mode[i].len; ++j2) {
                        sptAppendIndexVector(&kschr_rest[m], kschr_mode[i].data[j2]);
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
                for(sptIndex i=0; i < kernel_ndim; ++i) {
                    if(npars >= kschr_balanced_pos_mode[i].len) continue;
                    sptIndex tmp_j = kschr_balanced_pos_mode[i].data[npars];
                    if(tmp_j >= kschr_mode[i].len) continue;
                    sptIndex kernel_num = kschr_mode[i].data[tmp_j];
                    sptNnzIndex sum_nnzk = knnzs->data[kernel_num];
                    while(sum_nnzk <= par_nnzk) {
                        sptAppendIndexVector(&(kschr_balanced_mode[i]), kernel_num);
                        ++ count_nk;
                        ++ tmp_j;
                        if(tmp_j < kschr_mode[i].len) {
                            kernel_num = kschr_mode[i].data[tmp_j]; // j + 1
                            sum_nnzk += knnzs->data[kernel_num];
                        } else {
                            break;
                        }
                    }   // End of while
                    sptAppendIndexVector(&(kschr_balanced_pos_mode[i]), tmp_j);
                }
                ++ npars;
            }
            // printf("count_nk: %u\n", count_nk); fflush(stdout);
        }   // End of while
        nkpars[m] = npars;  // kschr_balanced_pos.len is npars + 1.
    }   // End loop of modes


    sptStopTimer(set_kernel_timer);
    sptPrintElapsedTime(set_kernel_timer, "\t\tSet Kernel Ptrs");
    sptFreeTimer(set_kernel_timer);

    sptTimer morton_sort_timer;
    sptNewTimer(&morton_sort_timer, 0);
    sptStartTimer(morton_sort_timer);

    /* Sort blocks in each kernel in Morton-order */
    sptNnzIndex k_begin, k_end;
    /* Loop for all kernels, 0-kptr.len for OMP code */
    #pragma omp parallel for num_threads(tk) 
    for(sptNnzIndex k=0; k<kptr->len - 1; ++k) {
        k_begin = kptr->data[k];
        k_end = kptr->data[k+1];   // exclusive
        /* Sort blocks in each kernel in Morton-order */
        sptSparseTensorSortIndexMorton(tsr, 1, k_begin, k_end, sb_bits, tk);
        // sptSparseTensorSortIndexRowBlock(tsr, 1, k_begin, k_end, sb_bits, tk);
#if PARTI_DEBUG == 3
    printf("Kernel %"PARTI_PRI_NNZ_INDEX ": Sorted by sptSparseTensorSortIndexMorton.\n", k);
    sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);
#endif
    }

    sptStopTimer(morton_sort_timer);
    sptPrintElapsedTime(morton_sort_timer, "\t\tMorton sorting");
    // sptPrintElapsedTime(morton_sort_timer, "\t\t2nd Rowblock sorting");
    sptFreeTimer(morton_sort_timer);

    return 0;
}


int sptSparseTensorToHiCOO(
    sptSparseTensorHiCOO *hitsr,
    sptNnzIndex *max_nnzb,
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits,
    int const tk)
{
    sptAssert(sk_bits >= sb_bits);
    sptAssert(sc_bits >= sb_bits);

    sptIndex i;
    int result;
    sptIndex nmodes = tsr->nmodes;
    sptNnzIndex nnz = tsr->nnz;

    sptElementIndex sb = pow(2, sb_bits);
    sptIndex sc = pow(2, sc_bits);

    /* Set HiCOO parameters. ndims for type conversion, size_t -> sptIndex */
    sptIndex * ndims = malloc(nmodes * sizeof *ndims);
    spt_CheckOSError(!ndims, "HiSpTns Convert");
    for(i = 0; i < nmodes; ++i) {
        ndims[i] = (sptIndex)tsr->ndims[i];
    }

    result = sptNewSparseTensorHiCOO(hitsr, (sptIndex)tsr->nmodes, ndims, (sptNnzIndex)tsr->nnz, sb_bits, sk_bits, sc_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);

    /* Pre-process tensor to get hitsr->kptr, values are nonzero locations. */
    sptTimer sort_timer;
    sptNewTimer(&sort_timer, 0);
    sptStartTimer(sort_timer);

    sptPreprocessSparseTensor(&hitsr->kptr, hitsr->kschr, hitsr->nkiters, hitsr->kschr_balanced, hitsr->kschr_balanced_pos, hitsr->nkpars, hitsr->kschr_rest, &hitsr->knnzs, tsr, sb_bits, sk_bits, tk);

    sptStopTimer(sort_timer);
    sptPrintElapsedTime(sort_timer, "\tHiCOO sorting (rowblock + morton)");
    sptFreeTimer(sort_timer);
#if PARTI_DEBUG >= 2
    printf("Kernels: Row-major, blocks: Morton-order sorted:\n");
    sptAssert(sptDumpSparseTensor(tsr, 0, stdout) == 0);
    printf("hitsr->kptr:\n");
    sptDumpNnzIndexVector(&hitsr->kptr, stdout);
#endif

    sptTimer gen_timer;
    sptNewTimer(&gen_timer, 0);
    sptStartTimer(gen_timer);

    /* Temporary storage */
    sptIndex * block_begin = (sptIndex *)malloc(nmodes * sizeof(*block_begin));
    sptIndex * block_end = (sptIndex *)malloc(nmodes * sizeof(*block_end));
    sptIndex * block_begin_prior = (sptIndex *)malloc(nmodes * sizeof(*block_begin_prior));
    sptIndex * block_coord = (sptIndex *)malloc(nmodes * sizeof(*block_coord));

    sptNnzIndex k_begin, k_end; // #Nonzeros locations
    sptNnzIndex nk = 0; // #Kernels  
    sptNnzIndex nc = 0; // #Chunks  
    sptNnzIndex nb = 1; // #Blocks  // counting from the first nnz
    sptNnzIndex nb_tmp = 0;
    sptNnzIndex ne = 0; // #Nonzeros per block
    sptIndex eindex = 0;
    sptBlockIndex chunk_size = 0;

    /* different appending methods:
     * elements: append every nonzero entry
     * blocks: append when seeing a new block.
     * chunks: appending when seeting a new chunk. Notice the boundary of kernels and the last chunk of the whole tensor may be larger than the sc.
     * kernels: append when seeing a new kernel. Not appending a vector, just write data into an allocated array.
     */
    /* Process first nnz */
    for(sptIndex m=0; m<nmodes; ++m) 
        block_coord[m] = tsr->inds[m].data[0];    // first nonzero indices
    result = sptLocateBeginCoord(block_begin_prior, tsr, block_coord, sb_bits);
    spt_CheckError(result, "HiSpTns Convert", NULL);
    for(sptIndex m=0; m<nmodes; ++m)
        sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin_prior[m]);
    sptAppendNnzIndexVector(&hitsr->bptr, 0);


    /* Loop for all kernels, 0 - hitsr->kptr.len - 1 for OMP code */
    for(sptNnzIndex k=0; k<hitsr->kptr.len - 1; ++k) {
        k_begin = hitsr->kptr.data[k];
        k_end = hitsr->kptr.data[k+1]; // exclusive
        nb_tmp = k == 0 ? 0: nb;
        /* Modify kptr pointing to block locations */
        hitsr->kptr.data[k] = nb_tmp;
        ++ nk;

        /* Only append a chunk for the new kernel, the last chunk in the old kernel may be larger than sc */
        sptAppendNnzIndexVector(&hitsr->cptr, nb_tmp);
        // printf("cptr 1:\n");
        // sptDumpNnzIndexVector(&hitsr->cptr, stdout);
        ++ nc;
        chunk_size = 0;

        /* Loop nonzeros in each kernel */
        for(sptNnzIndex z = k_begin; z < k_end; ++z) {
            #if PARTI_DEBUG == 5
                printf("z: %"PARTI_PRI_NNZ_INDEX "\n", z);
            #endif

            for(sptIndex m=0; m<nmodes; ++m) 
                block_coord[m] = tsr->inds[m].data[z];    // first nonzero indices
            #if PARTI_DEBUG == 5
                printf("block_coord:\n");
                sptAssert(sptDumpIndexArray(block_coord, nmodes, stdout) == 0);
            #endif

            result = sptLocateBeginCoord(block_begin, tsr, block_coord, sb_bits);
            // spt_CheckError(result, "HiSpTns Convert", NULL);
            #if PARTI_DEBUG == 5
                printf("block_begin_prior:\n");
                sptAssert(sptDumpIndexArray(block_begin_prior, nmodes, stdout) == 0);
                printf("block_begin:\n");
                sptAssert(sptDumpIndexArray(block_begin, nmodes, stdout) == 0);
            #endif

            result = sptBlockEnd(block_end, tsr, block_begin, sb);  // exclusive
            // spt_CheckError(result, "HiSpTns Convert", NULL);

            /* Append einds and values */
            for(sptIndex m=0; m<nmodes; ++m) {
                eindex = tsr->inds[m].data[z] < (block_begin[m] << sb_bits) ? tsr->inds[m].data[z] : tsr->inds[m].data[z] - (block_begin[m] << sb_bits);
                sptAssert(eindex < sb);
                sptAppendElementIndexVector(&hitsr->einds[m], (sptElementIndex)eindex);
            }
            sptAppendValueVector(&hitsr->values, tsr->values.data[z]);


            /* z in the same block with last z */
            if (sptEqualWithTwoCoordinates(block_begin, block_begin_prior, nmodes) == 1)
            {
                /* ne: #Elements in current block */
                ++ ne;
            } else { /* New block */
                /* ne: #Elements in the last block */
                /* Append block bptr and bidx */
                sptAppendNnzIndexVector(&hitsr->bptr, (sptBlockIndex)z);
                for(sptIndex m=0; m<nmodes; ++m)
                    sptAppendBlockIndexVector(&hitsr->binds[m], (sptBlockIndex)block_begin[m]);
                for(sptIndex m=0; m<nmodes; ++m)
                    block_begin_prior[m] = block_begin[m];

                /* ne: old block's number of nonzeros */
                // if(chunk_size + ne > sc || ne >= sc) { 
                // if(chunk_size + ne >= sc && chunk_size > 0) {    // calculate the prior block
                //     /* Append a chunk ending by the old block */
                //     sptAppendNnzIndexVector(&hitsr->cptr, nb-1);
                //     // printf("cptr 2:\n");
                //     // sptDumpNnzIndexVector(&hitsr->cptr, stdout);
                //     ++ nc;
                //     chunk_size = ne;
                // } else {
                //     chunk_size += ne;
                // }

                if(chunk_size + ne >= sc) {    // calculate the prior block
                    /* Append a chunk ending by the old block */
                    sptAppendNnzIndexVector(&hitsr->cptr, nb);
                    // printf("cptr 2:\n");
                    // sptDumpNnzIndexVector(&hitsr->cptr, stdout);
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
    sptAssert(nb <= nnz);
    sptAssert(nb == hitsr->binds[0].len); 
    // sptAssert(nc <= nb);
    sptAssert(nk == hitsr->kptr.len - 1);

    /* Last element for kptr, cptr, bptr */
    hitsr->kptr.data[hitsr->kptr.len - 1] = hitsr->bptr.len;
    sptAppendNnzIndexVector(&hitsr->cptr, hitsr->bptr.len);
    sptAppendNnzIndexVector(&hitsr->bptr, nnz);


    *max_nnzb = hitsr->bptr.data[1] - hitsr->bptr.data[0];
    sptNnzIndex sum_nnzb = 0;
    for(sptIndex i=0; i < hitsr->bptr.len - 1; ++i) {
        sptNnzIndex nnzb = hitsr->bptr.data[i+1] - hitsr->bptr.data[i];
        sum_nnzb += nnzb;
        if(*max_nnzb < nnzb) {
          *max_nnzb = nnzb;
        }
    }
    sptAssert(sum_nnzb == hitsr->nnz);

    sptStopTimer(gen_timer);
    sptPrintElapsedTime(gen_timer, "\tGenerate HiCOO");
    sptFreeTimer(gen_timer);


    free(block_begin);
    free(block_end);
    free(block_begin_prior);
    free(block_coord);

	return 0;
}
