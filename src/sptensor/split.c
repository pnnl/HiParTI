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

#include <assert.h>
#include <HiParTI.h>
#include <stdio.h>
#include "sptensor.h"


static int pti_FindSplitStep(const ptiSparseTensor *tsr, ptiNnzIndex cut_point, int direction) {
    if(direction) {
        if(cut_point == 0) {
            ++cut_point;
        }
        while(cut_point < tsr->nnz &&
            tsr->inds[0].data[cut_point-1] == tsr->inds[0].data[cut_point]) {
                ++cut_point;
        }
    } else {
        while(cut_point != 0 &&
            tsr->inds[0].data[cut_point-1] == tsr->inds[0].data[cut_point]) {
                --cut_point;
        }
    }
    return cut_point;
}

static void pti_RotateMode(ptiSparseTensor *tsr) {
    if(tsr->nmodes == 0) {
        return;
    }

    ptiIndexVector inds0 = tsr->inds[0];
    memmove(&tsr->inds[0], &tsr->inds[1], (tsr->nmodes-1) * sizeof (ptiIndexVector));
    tsr->inds[tsr->nmodes-1] = inds0;
}

/**
 * Split a sparse tensor into several sub-tensors.
 *
 * Call pti_StartSplitSparseTensor to start a split operation.
 * Call pti_SplitSparseTensor to get the next result from this split operation.
 * Call pti_FinishSplitSparseTensor to finish this split operation.
 *
 * @param[out] handle            The handle of this split operation
 * @param[in]  tsr               The sparse tensor to split from
 * @param[in]  max_size_by_mode  The number of cuts at each mode, length `tsr->nmodes`
 */
int pti_StartSplitSparseTensor(pti_SplitHandle *handle, const ptiSparseTensor *tsr, const ptiIndex max_size_by_mode[]) {
    int result = 0;

    *handle = malloc(sizeof **handle);
    (*handle)->nsplits = 0;
    (*handle)->tsr = malloc((tsr->nmodes + 1) * sizeof (ptiSparseTensor));
    pti_CheckOSError((*handle)->tsr == NULL, "SpTns Splt");
    (*handle)->max_size_by_mode = malloc(5 * tsr->nmodes * sizeof (ptiIndex));
    pti_CheckOSError((*handle)->max_size_by_mode == NULL, "SpTns Splt");
    (*handle)->inds_low = (*handle)->max_size_by_mode + tsr->nmodes;
    (*handle)->inds_high = (*handle)->max_size_by_mode + 2 * tsr->nmodes;
    (*handle)->level = 0;
    (*handle)->resume_branch = malloc((tsr->nmodes + 1) * sizeof (int));
    pti_CheckOSError((*handle)->resume_branch == NULL, "SpTns Splt");
    (*handle)->cut_idx = (*handle)->max_size_by_mode + 3 * tsr->nmodes;
    (*handle)->cut_low = (*handle)->max_size_by_mode + 4 * tsr->nmodes;

    result = ptiCopySparseTensor(&(*handle)->tsr[0], tsr, 1);
    pti_CheckError(result, "SpTns Splt", NULL);

    if(max_size_by_mode != NULL) {
        memcpy((*handle)->max_size_by_mode, max_size_by_mode, tsr->nmodes * sizeof (ptiIndex));
        ptiIndex m;
        for(m = 0; m + 1 < tsr->nmodes; ++m) {
            (*handle)->max_size_by_mode[m + 1] *= max_size_by_mode[m];
        }
    } else {
        memset((*handle)->max_size_by_mode, 0, tsr->nmodes * sizeof (ptiIndex));
    }

    (*handle)->resume_branch[0] = 0;

    return 0;
}

/**
 * Split a sparse tensor into several sub-tensors.
 *
 * Call pti_StartSplitSparseTensor to start a split operation.
 * Call pti_SplitSparseTensor to get the next result from this split operation.
 * Call pti_FinishSplitSparseTensor to finish this split operation.
 *
 * @param[out] dest       A pointer to an uninitialized ptiSparseTensor, to hold the sub-tensor output
 * @param[out] inds_low   The low index limit of this subtensor at each mode, length `dest->nmodes`
 * @param[out] inds_high  The high index limit of this subtensor at each mode, length `dest->nmodes`
 * @param      handle     The handle to this split operation
 * @return                Zero on success, `PTIERR_NO_MORE` when there is no more splits, or any other values to indicate an error
 */
int pti_SplitSparseTensor(ptiSparseTensor *dest, ptiIndex *inds_low, ptiIndex *inds_high, pti_SplitHandle handle) {
    int result;

    for(;;) {
        ptiSparseTensor *curtsr = &handle->tsr[handle->level];

        if(handle->resume_branch[handle->level] == 1) {
            if(handle->cut_low[handle->level] < curtsr->nnz) {
                ptiNnzIndex cut_high_est = handle->cut_low[handle->level] + handle->max_size_by_mode[handle->level];
                ptiNnzIndex cut_high;
                if(cut_high_est < curtsr->nnz) {
                    /* Find a previous step on the index */
                    cut_high = pti_FindSplitStep(curtsr, cut_high_est, 0);
                    if(cut_high <= handle->cut_low[handle->level]) {
                        /* Find a next step instead */
                        cut_high = pti_FindSplitStep(curtsr, cut_high_est, 1);
                        fprintf(stderr, "[SpTns Splt] cut #%zu size may exceed limit (%" HIPARTI_PRI_NNZ_INDEX " > %zu)\n", handle->nsplits + 1, cut_high - handle->cut_low[handle->level], handle->max_size_by_mode[handle->level]);
                    }
                } else {
                    cut_high = curtsr->nnz;
                }
                assert(cut_high > handle->cut_low[handle->level]);

                /* Extract this cut into a new subtensor */
                ptiSparseTensor *subtsr = &handle->tsr[handle->level + 1];
                result = ptiNewSparseTensor(subtsr, curtsr->nmodes, curtsr->ndims);
                ptiIndex m;
                for(m = 0; m < subtsr->nmodes; ++m) {
                    result = ptiResizeIndexVector(&subtsr->inds[m], cut_high - handle->cut_low[handle->level]);
                    pti_CheckError(result, "SpTns PartSplt", NULL);
                    memcpy(subtsr->inds[m].data, &curtsr->inds[m].data[handle->cut_low[handle->level]], (cut_high - handle->cut_low[handle->level]) * sizeof (ptiIndex));
                }
                result = ptiResizeValueVector(&subtsr->values, cut_high - handle->cut_low[handle->level]);
                pti_CheckError(result, "SpTns PartSplt", NULL);
                memcpy(subtsr->values.data, &curtsr->values.data[handle->cut_low[handle->level]], (cut_high - handle->cut_low[handle->level]) * sizeof (ptiValue));
                subtsr->nnz = cut_high - handle->cut_low[handle->level];

                handle->inds_low[handle->level] = subtsr->inds[0].data[0];
                handle->inds_high[handle->level] = subtsr->inds[0].data[subtsr->nnz - 1] + 1;

                handle->cut_low[handle->level] = cut_high;

                pti_RotateMode(subtsr);
                ptiSparseTensorSortIndex(subtsr, 1, 1);

                ++handle->level;
                handle->resume_branch[handle->level] = 0;
                continue;
            } else {
                ptiFreeSparseTensor(curtsr);
                if(handle->level == 0) {
                    return PTIERR_NO_MORE;
                }
                --handle->level;
                continue;
            }
        } else if(handle->resume_branch[handle->level] == 2) {
            if(handle->level == 0) {
                return PTIERR_NO_MORE;
            }
            --handle->level;
            continue;
        } else {
            assert(handle->resume_branch[handle->level] == 0);
        }

        /* Do nothing with an empty tensor */
        if(curtsr->nnz == 0) {
            ptiFreeSparseTensor(curtsr);

            if(handle->level == 0) {
                return PTIERR_NO_MORE;
            }
            --handle->level;
            continue;
        }

        /* All modes cut, finish the recursion */
        if(handle->level >= curtsr->nmodes) {
            if(dest) {
                *dest = *curtsr;
            }
            if(inds_low) {
                memcpy(inds_low, handle->inds_low, curtsr->nmodes * sizeof (ptiIndex));
            }
            if(inds_high) {
                memcpy(inds_high, handle->inds_high, curtsr->nmodes * sizeof (ptiIndex));
            }
            ++handle->nsplits;

            --handle->level;
            return 0;
        }

        /* No cuts required at this mode */
        if(handle->max_size_by_mode[handle->level] == 0) {
            handle->inds_low[handle->level] = curtsr->inds[0].data[0];
            handle->inds_high[handle->level] = curtsr->inds[0].data[curtsr->nnz - 1] + 1;

            pti_RotateMode(curtsr);
            ptiSparseTensorSortIndex(curtsr, 1, 1);
            handle->tsr[handle->level + 1] = *curtsr;

            handle->resume_branch[handle->level] = 2;
            ++handle->level;
            handle->resume_branch[handle->level] = 0;
            continue;
        }

        handle->cut_idx[handle->level] = 0;
        handle->cut_low[handle->level] = 0;
        handle->resume_branch[handle->level] = 1;
    }

    return 0;
}

/**
 * Split a sparse tensor into several sub-tensors.
 *
 * Call pti_StartSplitSparseTensor to start a split operation.
 * Call pti_SplitSparseTensor to get the next result from this split operation.
 * Call pti_FinishSplitSparseTensor to finish this split operation.
 *
 * @param handle  The handle to this split operation
 */
void pti_FinishSplitSparseTensor(pti_SplitHandle handle) {
    free(handle->resume_branch);
    free(handle->max_size_by_mode);
    free(handle->tsr);
    free(handle);
}
