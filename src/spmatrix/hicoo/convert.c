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


static int ptiBlockEnd(
    ptiIndex * out_item,
    ptiIndex nmodes,
    ptiIndex nrows,
    ptiIndex ncols,
    const ptiIndex * in_item,
    const ptiElementIndex sb)
{
    ptiAssert(in_item[0] < nrows);
    out_item[0] = in_item[0]+sb < nrows ? in_item[0]+sb : nrows;    // exclusive
    ptiAssert(in_item[1] < ncols);
    out_item[1] = in_item[1]+sb < ncols ? in_item[1]+sb : ncols;    // exclusive

    return 0;
}


static int ptiLocateBeginCoord(
    ptiIndex * out_item,
    ptiIndex nmodes,
    const ptiIndex * in_item,
    const ptiElementIndex bits)
{
    for(ptiIndex m=0; m<nmodes; ++m) {
        out_item[m] = in_item[m] >> bits;
    }

    return 0;
}


/**
 * Record mode pointers for kernel rows, from a sorted matrix.
 * @param kptr  a vector of kernel pointers
 * @param mtx    a pointer to a sparse matrix
 * @return      mode pointers
 */
int ptiSetKernelPointersMat(
    ptiNnzIndexVector *kptr,
    ptiSparseMatrix *mtx,
    const ptiElementIndex sk_bits)
{
    ptiIndex nmodes = 2;
    ptiNnzIndex nnz = mtx->nnz;
    ptiNnzIndex k = 0;  // count kernels
    ptiNnzIndex knnz = 0;   // #Nonzeros per kernel
    int result = 0;
    result = ptiAppendNnzIndexVector(kptr, 0);
    pti_CheckError(result, "HiSpMtx Convert", NULL);

    ptiIndex * coord = (ptiIndex *)malloc(nmodes * sizeof(*coord));
    ptiIndex * kernel_coord = (ptiIndex *)malloc(nmodes * sizeof(*kernel_coord));
    ptiIndex * kernel_coord_prior = (ptiIndex *)malloc(nmodes * sizeof(*kernel_coord_prior));

    /* Process first nnz to get the first kernel_coord_prior */
    coord[0] = mtx->rowind.data[0];    // first nonzero indices
    coord[1] = mtx->colind.data[0];    // first nonzero indices
    result = ptiLocateBeginCoord(kernel_coord_prior, nmodes, coord, sk_bits);
    pti_CheckError(result, "HiSpMtx Convert", NULL);

    for(ptiNnzIndex z=0; z<nnz; ++z) {
        coord[0] = mtx->rowind.data[z];
        coord[1] = mtx->colind.data[z];
        result = ptiLocateBeginCoord(kernel_coord, nmodes, coord, sk_bits);
        pti_CheckError(result, "HiSpMtx Convert", NULL);

        if(ptiEqualWithTwoCoordinates(kernel_coord, kernel_coord_prior, nmodes) == 1) {
            ++ knnz;
        } else {
            ++ k;
            result = ptiAppendNnzIndexVector(kptr, knnz + kptr->data[k-1]);
            pti_CheckError(result, "HiSpMtx Convert", NULL);
            for(ptiIndex m=0; m<nmodes; ++m)
                kernel_coord_prior[m] = kernel_coord[m];
            knnz = 1;
        }
    }   // End loop z
    ptiAssert(k < kptr->len);
    ptiAssert(kptr->data[kptr->len-1] + knnz == nnz);

    /* Set the last element for kptr */
    ptiAppendNnzIndexVector(kptr, nnz);

    free(coord);
    free(kernel_coord);
    free(kernel_coord_prior);

    return 0;
}


/**
 * Set scheduler for kernels.
 * @param kschr  nmodes kernel schedulers.
 * @param tsr    a pointer to a sparse matrix
 * @return      mode pointers
 */
static int ptiSetKernelScheduler(
    ptiIndexVector *kschr,
    ptiIndex *nkiters,
    ptiNnzIndexVector * const kptr,
    ptiSparseMatrix *mtx,
    const ptiElementIndex sk_bits)
{
    ptiIndex nmodes = 2;
    int result = 0;

    ptiIndex * coord = (ptiIndex *)malloc(nmodes * sizeof(*coord));
    ptiIndex * kernel_coord = (ptiIndex *)malloc(nmodes * sizeof(*kernel_coord));

    for(ptiNnzIndex k=0; k<kptr->len - 1; ++k) {    // superblocks
        ptiNnzIndex z = kptr->data[k];
        coord[0] = mtx->rowind.data[z];
        coord[1] = mtx->colind.data[z];
        result = ptiLocateBeginCoord(kernel_coord, nmodes, coord, sk_bits);
        pti_CheckError(result, "HiSpMtx Convert", NULL);

        result = ptiAppendIndexVector(&(kschr[kernel_coord[0]]), k);    // only according to column index for this scheduler
        pti_CheckError(result, "HiSpMtx Convert", NULL);
    }   // End loop k

    free(coord);
    free(kernel_coord);

    ptiIndex sk = (ptiIndex)pow(2, sk_bits);
    ptiIndex tmp = 0;
    ptiIndex kernel_ndim = (mtx->ncols + sk - 1) / sk;
    for(ptiIndex i=0; i<kernel_ndim; ++i) { // find the maximum length of each row of kschr
        if(tmp < kschr[i].len)
            tmp = kschr[i].len;
    }
    *nkiters = tmp;

    return 0;
}

/**
 * Pre-process COO sparse matrix by permuting, sorting, and record pointers to blocked rows. Kernels in Row-major order, blocks and elements are in Z-Morton order.
 * @param mtx    a pointer to a sparse matrix
 * @return      mode pointers
 */
int ptiPreprocessSparseMatrix(
    ptiNnzIndexVector * kptr,
    ptiIndexVector *kschr,
    ptiIndex *nkiters,
    ptiSparseMatrix *mtx,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits)
{
    ptiNnzIndex nnz = mtx->nnz;
    int result;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    /* Sort tsr in a Row-major Block order to get all kernels. Not use Morton-order for kernels: 1. better support for higher-order tensors by limiting kernel size, because Morton key bit <= 128; */
    ptiStartTimer(timer);
    ptiSparseMatrixSortIndexRowBlock(mtx, 1, 0, nnz, sk_bits);  // Parallelized inside
    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "\t\trowblock sorting");
#if PARTI_DEBUG == 3
    printf("Sorted by ptiSparseTensorSortIndexRowBlock.\n");
    ptiAssert(ptiDumpSparseMatrix(mtx, 0, stdout) == 0);
#endif

    ptiStartTimer(timer);

    result = ptiSetKernelPointersMat(kptr, mtx, sk_bits);
    pti_CheckError(result, "HiSpMtx Preprocess", NULL);
    result = ptiSetKernelScheduler(kschr, nkiters, kptr, mtx, sk_bits);
    pti_CheckError(result, "HiSpTns Preprocess", NULL);

    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "\t\tSet Kernel Ptrs");

    ptiStartTimer(timer);
    /* Sort blocks in each kernel in Morton-order */
    /* Loop for all kernels, 0-kptr.len for OMP code */
    #pragma omp parallel for
    for(ptiNnzIndex k=0; k<kptr->len - 1; ++k) {
        ptiNnzIndex k_begin = kptr->data[k];
        ptiNnzIndex k_end = kptr->data[k+1];   // exclusive
        /* Sort blocks in each kernel in Morton-order */
        ptiSparseMatrixSortIndexMorton(mtx, 1, k_begin, k_end, sb_bits);
#if PARTI_DEBUG == 3
    printf("Kernel %"HIPARTI_PRI_NNZ_INDEX ": Sorted by ptiSparseTensorSortIndexMorton.\n", k);
    ptiAssert(ptiDumpSparseTensor(tsr, 0, stdout) == 0);
#endif
    }
    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "\t\tMorton sorting");
    
    ptiFreeTimer(timer);

    return 0;
}


int ptiSparseMatrixPartition(
    ptiSparseMatrixHiCOO *himtx,
    ptiNnzIndex *max_nnzb,
    ptiSparseMatrix *mtx,
    const ptiElementIndex sb_bits)
{
    int result;
    ptiNnzIndex nnz = mtx->nnz;
    ptiIndex nrows = mtx->nrows;
    ptiIndex ncols = mtx->ncols;
    ptiIndex nmodes = 2;
    ptiElementIndex sb = pow(2, sb_bits);

    /* Temporary storage */
    ptiIndex * block_begin = (ptiIndex *)malloc(nmodes * sizeof(*block_begin));
    ptiIndex * block_end = (ptiIndex *)malloc(nmodes * sizeof(*block_end));
    ptiIndex * block_begin_prior = (ptiIndex *)malloc(nmodes * sizeof(*block_begin_prior));
    ptiIndex * block_coord = (ptiIndex *)malloc(nmodes * sizeof(*block_coord));

    ptiNnzIndex k_begin, k_end; // #Nonzeros locations
    ptiNnzIndex nk = 0; // #Kernels
    ptiNnzIndex nb = 1; // #Blocks  // counting from the first nnz
    ptiNnzIndex nb_tmp = 0;
    ptiNnzIndex ne = 0; // #Nonzeros per block
    ptiIndex eindex = 0;

    /* different appending methods:
     * elements: append every nonzero entry
     * blocks: append when seeing a new block.
     * chunks: appending when seeting a new chunk. Notice the boundary of kernels and the last chunk of the whole tensor may be larger than the sc.
     * kernels: append when seeing a new kernel. Not appending a vector, just write data into an allocated array.
     */
    /* Process first nnz */
    block_coord[0] = mtx->rowind.data[0];    // first nonzero indices
    block_coord[1] = mtx->colind.data[0];    // first nonzero indices
    result = ptiLocateBeginCoord(block_begin_prior, nmodes, block_coord, sb_bits);
    pti_CheckError(result, "HiSpMtx Convert", NULL);
    ptiAppendBlockIndexVector(&himtx->bindI, (ptiBlockIndex)block_begin_prior[0]);
    ptiAppendBlockIndexVector(&himtx->bindJ, (ptiBlockIndex)block_begin_prior[1]);
    ptiAppendNnzIndexVector(&himtx->bptr, 0);

    /* Loop for all kernels, 0 - himtx->kptr.len - 1 */
    for(ptiNnzIndex k=0; k<himtx->kptr.len - 1; ++k) {
        k_begin = himtx->kptr.data[k];
        k_end = himtx->kptr.data[k+1]; // exclusive
        nb_tmp = k == 0 ? 0: nb;
        /* Modify kptr pointing to block locations */
        himtx->kptr.data[k] = nb_tmp;
        ++ nk;

        /* Loop nonzeros in each kernel */
        for(ptiNnzIndex z = k_begin; z < k_end; ++z) {
            // printf("z: %"HIPARTI_PRI_NNZ_INDEX "\n", z);

            block_coord[0] = mtx->rowind.data[z];    // first nonzero indices
            block_coord[1] = mtx->colind.data[z];    // first nonzero indices
            // printf("block_coord:\n");
            // ptiAssert(ptiDumpIndexArray(block_coord, nmodes, stdout) == 0);

            result = ptiLocateBeginCoord(block_begin, nmodes, block_coord, sb_bits);
            pti_CheckError(result, "HiSpMtx Convert", NULL);
            // printf("block_begin_prior:\n");
            // ptiAssert(ptiDumpIndexArray(block_begin_prior, nmodes, stdout) == 0);
            // printf("block_begin:\n");
            // ptiAssert(ptiDumpIndexArray(block_begin, nmodes, stdout) == 0);

            result = ptiBlockEnd(block_end, nmodes, nrows, ncols, block_begin, sb);  // exclusive
            pti_CheckError(result, "HiSpMtx Convert", NULL);

            /* Append einds and values */
            eindex = mtx->rowind.data[z] < (block_begin[0] << sb_bits) ? mtx->rowind.data[z] : mtx->rowind.data[z] - (block_begin[0] << sb_bits);
            ptiAssert(eindex < sb);
            ptiAppendElementIndexVector(&himtx->eindI, (ptiElementIndex)eindex);
            eindex = mtx->colind.data[z] < (block_begin[1] << sb_bits) ? mtx->colind.data[z] : mtx->colind.data[z] - (block_begin[1] << sb_bits);
            ptiAssert(eindex < sb);
            ptiAppendElementIndexVector(&himtx->eindJ, (ptiElementIndex)eindex);
            ptiAppendValueVector(&himtx->values, mtx->values.data[z]);


            /* z in the same block with last z */
            if (ptiEqualWithTwoCoordinates(block_begin, block_begin_prior, nmodes) == 1)
            {
                /* ne: #Elements in current block */
                ++ ne;
            } else { /* New block */
                /* ne: #Elements in the last block */
                /* Append block bptr and bidx */
                ptiAppendNnzIndexVector(&himtx->bptr, (ptiBlockIndex)z);
                ptiAppendBlockIndexVector(&himtx->bindI, (ptiBlockIndex)block_begin[0]);
                ptiAppendBlockIndexVector(&himtx->bindJ, (ptiBlockIndex)block_begin[1]);
                for(ptiIndex m=0; m<nmodes; ++m)
                    block_begin_prior[m] = block_begin[m];

                ++ nb;
                ne = 1;              
            } // End new block
            // printf("nb: %"HIPARTI_PRI_NNZ_INDEX ", ne: %"HIPARTI_PRI_NNZ_INDEX "\n\n", nb, ne);

        }   // End z loop
    }   // End k loop
    
    ptiAssert(nb <= nnz);
    ptiAssert(nb == himtx->bindI.len);
    ptiAssert(nk == himtx->kptr.len - 1);

    /* Last element for kptr, bptr */
    himtx->kptr.data[himtx->kptr.len - 1] = himtx->bptr.len;    // change kptr pointing to blocks
    ptiAppendNnzIndexVector(&himtx->bptr, nnz);

    *max_nnzb = himtx->bptr.data[1] - himtx->bptr.data[0];
    ptiNnzIndex sum_nnzb = 0;
    for(ptiIndex i=0; i < himtx->bptr.len - 1; ++i) {
        ptiNnzIndex nnzb = himtx->bptr.data[i+1] - himtx->bptr.data[i];
        sum_nnzb += nnzb;
        if(*max_nnzb < nnzb) {
          *max_nnzb = nnzb;
        }
    }
    ptiAssert(sum_nnzb == himtx->nnz);

    free(block_begin);
    free(block_end);
    free(block_begin_prior);
    free(block_coord);
    return 0;
}

int ptiSparseMatrixToHiCOO(
    ptiSparseMatrixHiCOO *himtx,
    ptiNnzIndex *max_nnzb,
    ptiSparseMatrix *mtx,
    const ptiElementIndex sb_bits,
    const ptiElementIndex sk_bits)
{
    ptiAssert(sk_bits >= sb_bits);

    int result;
    ptiNnzIndex nnz = mtx->nnz;
    ptiIndex nrows = mtx->nrows;
    ptiIndex ncols = mtx->ncols;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    result = ptiNewSparseMatrixHiCOO(himtx, nrows, ncols, nnz, sb_bits, sk_bits);
    pti_CheckError(result, "HiSpMtx Convert", NULL);

    /* Sort blocks in each kernel in Morton-order */
    ptiStartTimer(timer);
    ptiPreprocessSparseMatrix(&himtx->kptr, himtx->kschr, &himtx->nkiters, mtx, sb_bits, sk_bits);
    // ptiSparseMatrixSortIndexMorton(mtx, 1, 0, nnz, sb_bits);
    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "\tHiCOO sorting (Morton)");


    ptiStartTimer(timer);
    ptiSparseMatrixPartition(himtx, max_nnzb, mtx, sb_bits);
    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "Generate HiCOO");

    ptiFreeTimer(timer);
    return 0;
}

