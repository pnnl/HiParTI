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
#include "sptensor.h"
#include <assert.h>

/* TODO: delete these functions below, this function is different from "ptiSparseTensorSortIndexSingleMode" */
    
static void pti_QuickSortAtMode(ptiSparseTensor *tsr, ptiNnzIndex const l, ptiNnzIndex const r, ptiIndex const mode);
static int pti_SparseTensorCompareAtMode(const ptiSparseTensor *tsr1, ptiNnzIndex const ind1, const ptiSparseTensor *tsr2, ptiNnzIndex const ind2, ptiIndex const mode);

/**
 * Reorder the elements in a sparse tensor lexicographically, but consider mode `mode` the last one
 * @param tsr  the sparse tensor to operate on
 * @param mode the mode to be considered the last
 */
void ptiSparseTensorSortIndexAtMode(ptiSparseTensor *tsr, ptiIndex const mode, int force) {
    ptiIndex m;
    int needsort = 0;

    for(m = 0; m < mode; ++m) {
        if(tsr->sortorder[m] != m) {
            tsr->sortorder[m] = m;
            needsort = 1;
        }
    }
    for(m = mode+1; m < tsr->nmodes; ++m) {
        if(tsr->sortorder[m-1] != m) {
            tsr->sortorder[m-1] = m;
            needsort = 1;
        }
    }
    if(tsr->sortorder[tsr->nmodes-1] != mode) {
        tsr->sortorder[tsr->nmodes-1] = mode;
        needsort = 1;
    }

    if(needsort || force) {
        pti_QuickSortAtMode(tsr, 0, tsr->nnz, mode);
    }
}

static void pti_QuickSortAtMode(ptiSparseTensor *tsr, ptiNnzIndex const l, ptiNnzIndex const r, ptiIndex const mode) {
    ptiNnzIndex i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(pti_SparseTensorCompareAtMode(tsr, i, tsr, p, mode) < 0) {
            ++i;
        }
        while(pti_SparseTensorCompareAtMode(tsr, p, tsr, j, mode) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        pti_SwapValues(tsr, i, j);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    pti_QuickSortAtMode(tsr, l, i, mode);
    pti_QuickSortAtMode(tsr, i, r, mode);
}

static int pti_SparseTensorCompareAtMode(const ptiSparseTensor *tsr1, ptiNnzIndex const ind1, const ptiSparseTensor *tsr2, ptiNnzIndex const ind2, ptiIndex const mode) {
    ptiIndex i;
    ptiIndex eleind1, eleind2;
    assert(tsr1->nmodes == tsr2->nmodes);
    for(i = 0; i < tsr1->nmodes; ++i) {
        if(i != mode) {
            eleind1 = tsr1->inds[i].data[ind1];
            eleind2 = tsr2->inds[i].data[ind2];
            if(eleind1 < eleind2) {
                return -1;
            } else if(eleind1 > eleind2) {
                return 1;
            }
        }
    }
    eleind1 = tsr1->inds[mode].data[ind1];
    eleind2 = tsr2->inds[mode].data[ind2];
    if(eleind1 < eleind2) {
        return -1;
    } else if(eleind1 > eleind2) {
        return 1;
    } else {
        return 0;
    }
}
