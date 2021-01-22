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

#include <assert.h>
#include <math.h>
#include <time.h>
#include <HiParTI.h>

static const uint32_t MASKS[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
static const uint32_t SHIFTS[] = {1, 2, 4, 8};

void pti_SwapValuesMat(ptiSparseMatrix *mtx, ptiNnzIndex ind1, ptiNnzIndex ind2) {

    ptiIndex eleind1;
    eleind1 = mtx->rowind.data[ind1];
    mtx->rowind.data[ind1] = mtx->rowind.data[ind2];
    mtx->rowind.data[ind2] = eleind1;
    eleind1 = mtx->colind.data[ind1];
    mtx->colind.data[ind1] = mtx->colind.data[ind2];
    mtx->colind.data[ind2] = eleind1;

    ptiValue val1 = mtx->values.data[ind1];
    mtx->values.data[ind1] = mtx->values.data[ind2];
    mtx->values.data[ind2] = val1;
}

/* Compare functions */
int pti_SparseMatrixCompareIndicesMorton2D(
    ptiSparseMatrix * const mtx1,
    uint64_t loc1, 
    ptiSparseMatrix * const mtx2,
    uint64_t loc2,
    ptiElementIndex sb_bits)
{
    uint64_t mkey1 = 0, mkey2 = 0;
    
    /* Only support 3-D tensors, with 32-bit indices. */
    uint32_t x1 = mtx1->rowind.data[loc1];
    uint32_t y1 = mtx1->colind.data[loc1];
    uint32_t x2 = mtx2->rowind.data[loc2];
    uint32_t y2 = mtx2->colind.data[loc2];

    /* Compare block indices */
    ptiIndex blk_x1 = x1 >> sb_bits;
    ptiIndex blk_y1 = y1 >> sb_bits;
    ptiIndex blk_x2 = x2 >> sb_bits;
    ptiIndex blk_y2 = y2 >> sb_bits;

    if(blk_x1 < blk_x2) {
        return -1;
    } else if(blk_x1 > blk_x2) {
        return 1;
    } else if(blk_y1 < blk_y2) {  // if blk_x1 == blk_x2
        return -1;
    } else if(blk_y1 > blk_y2) {  // if blk_x1 == blk_x2
        return 1;
    }

    /* blk_x1 == blk_x2, blk_y1 == blk_y2, sort inside a block in Z-Morton order */
    uint64_t x = x1 - (blk_x1 << sb_bits);
    uint64_t y = y1 - (blk_y1 << sb_bits);
    x = (x | (x << SHIFTS[3])) & MASKS[3];
    x = (x | (x << SHIFTS[2])) & MASKS[2];
    x = (x | (x << SHIFTS[1])) & MASKS[1];
    x = (x | (x << SHIFTS[0])) & MASKS[0];
    y = (y | (y << SHIFTS[3])) & MASKS[3];
    y = (y | (y << SHIFTS[2])) & MASKS[2];
    y = (y | (y << SHIFTS[1])) & MASKS[1];
    y = (y | (y << SHIFTS[0])) & MASKS[0];
    mkey1 = y | (x << 1);

    x = x2 - (blk_x2 << sb_bits);
    y = y2 - (blk_y2 << sb_bits);
    x = (x | (x << SHIFTS[3])) & MASKS[3];
    x = (x | (x << SHIFTS[2])) & MASKS[2];
    x = (x | (x << SHIFTS[1])) & MASKS[1];
    x = (x | (x << SHIFTS[0])) & MASKS[0];
    y = (y | (y << SHIFTS[3])) & MASKS[3];
    y = (y | (y << SHIFTS[2])) & MASKS[2];
    y = (y | (y << SHIFTS[1])) & MASKS[1];
    y = (y | (y << SHIFTS[0])) & MASKS[0];
    mkey2 = y | (x << 1);

    if(mkey1 < mkey2) {
        return -1;
    } else if(mkey1 > mkey2) {
        return 1;
    } else {
        return 0;
    }
    
}


int pti_SparseMatrixCompareIndicesSingleMode(ptiSparseMatrix * const mtx1, ptiNnzIndex loc1, ptiSparseMatrix * const mtx2, ptiNnzIndex loc2, ptiIndex const mode)
{
    ptiIndex eleind1, eleind2;
    if (mode == 0) {
        eleind1 = mtx1->rowind.data[loc1];
        eleind2 = mtx2->rowind.data[loc2];
    } else if (mode == 1) {
        eleind1 = mtx1->colind.data[loc1];
        eleind2 = mtx2->colind.data[loc2];
    }
    // printf("eleind1: %u (loc: %lu), eleind2: %u (loc: %lu)\n", eleind1, loc1, eleind2, loc2); fflush(stdout);
    if(eleind1 < eleind2) {
        return -1;
    } else if(eleind1 > eleind2) {
        return 1;
    }

    return 0;
}


int pti_SparseMatrixCompareIndicesRowBlock(
    ptiSparseMatrix * const mtx1,
    ptiNnzIndex loc1,
    ptiSparseMatrix * const mtx2,
    ptiNnzIndex loc2,
    ptiElementIndex sk_bits)
{
    ptiIndex eleind1 = mtx1->rowind.data[loc1];
    ptiIndex eleind2 = mtx2->rowind.data[loc2];
    ptiIndex blkind1 = eleind1 >> sk_bits;
    ptiIndex blkind2 = eleind2 >> sk_bits;
    // printf("blkind1: %lu, blkind2: %lu\n", blkind1, blkind2);

    if(blkind1 < blkind2) {
        return -1;
    } else if(blkind1 > blkind2) {
        return 1;
    } 

    eleind1 = mtx1->colind.data[loc1];
    eleind2 = mtx2->colind.data[loc2];
    blkind1 = eleind1 >> sk_bits;
    blkind2 = eleind2 >> sk_bits;

    if(blkind1 < blkind2) {
        return -1;
    } else if(blkind1 > blkind2) {
        return 1;
    } 

    return 0;
}


/* Quick sort functions */
static void pti_QuickSortIndexMorton2D(ptiSparseMatrix *mtx, ptiNnzIndex l, ptiNnzIndex r, ptiElementIndex sb_bits)
{

    uint64_t i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(pti_SparseMatrixCompareIndicesMorton2D(mtx, i, mtx, p, sb_bits) < 0) {
            // printf("(%lu, %lu) result: %d\n", i, p, pti_SparseMatrixCompareIndicesMorton2D(mtx, i, mtx, p));
            ++i;
        }
        while(pti_SparseMatrixCompareIndicesMorton2D(mtx, p, mtx, j, sb_bits) < 0) {
            // printf("(%lu, %lu) result: %d\n", p, j,pti_SparseMatrixCompareIndicesMorton2D(mtx, p, mtx, j));
            --j;
        }
        if(i >= j) {
            break;
        }
        pti_SwapValuesMat(mtx, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    #pragma omp task firstprivate(l,i) shared(mtx)
    {
        pti_QuickSortIndexMorton2D(mtx, l, i, sb_bits);
    }
    pti_QuickSortIndexMorton2D(mtx, i, r, sb_bits);
    #pragma omp taskwait 
    
}


static void pti_QuickSortIndexSingleMode(ptiSparseMatrix *mtx, ptiNnzIndex l, ptiNnzIndex r, ptiIndex mode)
{
    // printf("l: %lu, r: %lu.\n", l, r); fflush(stdout);
    ptiNnzIndex i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        // printf("i: %lu, j: %lu.\n", i, j); fflush(stdout);
        while(pti_SparseMatrixCompareIndicesSingleMode(mtx, i, mtx, p, mode) < 0) {
            ++i;
        }
        while(pti_SparseMatrixCompareIndicesSingleMode(mtx, p, mtx, j, mode) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        // printf("new i: %lu (%u, %u), j: %lu (%u, %u).\n", i, mtx->rowind.data[i], mtx->colind.data[i], j, mtx->rowind.data[j], mtx->colind.data[j]); fflush(stdout);
        pti_SwapValuesMat(mtx, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
        // printf("p: %lu.\n", p); fflush(stdout);
    }
    #pragma omp task firstprivate(l,i) shared(mtx, mode)
    {
        pti_QuickSortIndexSingleMode(mtx, l, i, mode);
    }
    pti_QuickSortIndexSingleMode(mtx, i, r, mode);
    #pragma omp taskwait
}


static void pti_QuickSortIndexRowBlock(ptiSparseMatrix *mtx, ptiNnzIndex l, ptiNnzIndex r,  ptiElementIndex sk_bits)
{
    ptiNnzIndex i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(pti_SparseMatrixCompareIndicesRowBlock(mtx, i, mtx, p, sk_bits) < 0) {
            ++i;
        }
        while(pti_SparseMatrixCompareIndicesRowBlock(mtx, p, mtx, j, sk_bits) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        pti_SwapValuesMat(mtx, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    #pragma omp task firstprivate(l,i) shared(mtx, sk_bits)
    {
        pti_QuickSortIndexRowBlock(mtx, l, i, sk_bits);
    }
    pti_QuickSortIndexRowBlock(mtx, i, r, sk_bits);
    #pragma omp taskwait
}


/****************************
 * Sorting functions
 ****************************/
void ptiSparseMatrixSortIndexMorton(
    ptiSparseMatrix *mtx,
    int force,
    ptiNnzIndex begin,
    ptiNnzIndex end,
    ptiElementIndex sb_bits)
{
    if(force) {
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                pti_QuickSortIndexMorton2D(mtx, begin, end, sb_bits);
            }
        }
    }
}

void ptiSparseMatrixSortIndexSingleMode(ptiSparseMatrix *mtx, int force, ptiIndex mode, int tk)
{
    if(force) {
        #pragma omp parallel num_threads(tk) 
        {
            #pragma omp single nowait 
            {
                pti_QuickSortIndexSingleMode(mtx, 0, mtx->nnz, mode);
            }
        }
    }
}

/**
 * Reorder the elements in a COO sparse matrix lexicographically, sorting by row major order.
 * @param mtx  the sparse matrix to operate on
 */
void ptiSparseMatrixSortIndexRowBlock(
    ptiSparseMatrix *mtx,
    int force,
    ptiNnzIndex begin,
    ptiNnzIndex end,
    ptiElementIndex sk_bits)
{
    if(force) {
        #pragma omp parallel 
        {
            #pragma omp single nowait
            {
                pti_QuickSortIndexRowBlock(mtx, begin, end, sk_bits);
            }
        }
    }
}

/**
 * Randomly shuffle all indices.
 *
 * @param[in] mtx matrix to be shuffled
 * @param[out] map_inds records the randomly generated mapping
 *
 */
void ptiGetRandomShuffledIndicesMat(ptiSparseMatrix *mtx, ptiIndex ** map_inds)
{
    /* Get randomly renumbering indices */
    for(ptiIndex m = 0; m < 2; ++m) {
        ptiIndex dim_len;
        if (m == 0) dim_len = mtx->nrows;
        else if (m == 1) dim_len = mtx->ncols;
        
        for(long int i = dim_len - 1; i > 0; --i) {
            srand(m+i+1+time(NULL));
            ptiIndex new_loc = (ptiIndex) (rand() % (i+1));
            /* Swap i <-> new_loc */
            ptiIndex tmp = map_inds[m][i];
            map_inds[m][i] = map_inds[m][new_loc];
            map_inds[m][new_loc] = tmp;
        }
    }
}