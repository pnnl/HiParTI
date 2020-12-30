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

#include <stdio.h>
#include <stdlib.h>
#include <HiParTI.h>
#include "../src/sptensor/sptensor.h"

int main(int argc, char const *argv[]) {
    FILE *fX, *fY;
    ptiSparseTensor X, spY;
    ptiSemiSparseTensor Y;
    ptiMatrix U;
    ptiIndex mode = 0;
    ptiIndex R = 16;
    int dev_id = -2;
    int niters = 5;

    if(argc < 5) {
        printf("Usage: %s X mode renumber [dev_id, R, Y]\n\n", argv[0]);
        return 1;
    }

    fX = fopen(argv[1], "r");
    ptiAssert(fX != NULL);
    ptiAssert(ptiLoadSparseTensor(&X, 1, fX) == 0);
    fclose(fX);

    sscanf(argv[2], "%"HIPARTI_SCN_INDEX, &mode);
    int renumber = 0;
    int niters_renum = 5;
    ptiElementIndex sb_bits = 7;
    /* renumber:
     * = 0 : no renumbering.
     * = 1 : renumber with Lexi-order
     * = 2 : renumber with BFS-like
     * = 3 : randomly renumbering, specify niters_renum.
     */
    sscanf(argv[3], "%d", &renumber);
    printf("renumber: %d\n", renumber);
    if (renumber == 1)
        printf("niters_renum: %d\n\n", niters_renum);   

    if(argc > 4) {
        sscanf(argv[4], "%d", &dev_id);
    }
    if(argc > 5) {
        sscanf(argv[5], "%"HIPARTI_SCN_INDEX, &R);
    }

    /* Renumber the input tensor */
    ptiIndex ** map_inds;
    if (renumber > 0) {
        map_inds = (ptiIndex **)malloc(X.nmodes * sizeof *map_inds);
        pti_CheckOSError(!map_inds, "MTTKRP HiCOO");
        for(ptiIndex m = 0; m < X.nmodes; ++m) {
            map_inds[m] = (ptiIndex *)malloc(X.ndims[m] * sizeof (ptiIndex));
            pti_CheckError(!map_inds[m], "MTTKRP HiCOO", NULL);
            for(ptiIndex i = 0; i < X.ndims[m]; ++i)
                map_inds[m][i] = i;
        }

        ptiTimer renumber_timer;
        ptiNewTimer(&renumber_timer, 0);
        ptiStartTimer(renumber_timer);

        if ( renumber == 1 || renumber == 2) { /* Set the Lexi-order or BFS-like renumbering */
            ptiIndexRenumber(&X, map_inds, renumber, niters_renum, sb_bits, 1, 1);
        }
        if ( renumber == 3) { /* Set randomly renumbering */
            printf("[Random Indexing]\n");
            ptiGetRandomShuffledIndices(&X, map_inds);
        }
        // fflush(stdout);

        ptiStopTimer(renumber_timer);
        ptiPrintElapsedTime(renumber_timer, "Renumbering");
        ptiFreeTimer(renumber_timer);

        ptiTimer shuffle_timer;
        ptiNewTimer(&shuffle_timer, 0);
        ptiStartTimer(shuffle_timer);

        ptiSparseTensorShuffleIndices(&X, map_inds);

        ptiStopTimer(shuffle_timer);
        ptiPrintElapsedTime(shuffle_timer, "Shuffling time");
        ptiFreeTimer(shuffle_timer);
        printf("\n");

        // ptiSparseTensorSortIndex(&X, 1, 1);
        // printf("map_inds:\n");
        // for(ptiIndex m = 0; m < X.nmodes; ++m) {
        //     ptiDumpIndexArray(map_inds[m], X.ndims[m], stdout);
        // }
        // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);
    }


    fprintf(stderr, "ptiRandomizeMatrix(&U, %"HIPARTI_PRI_INDEX ", %"HIPARTI_PRI_INDEX ")\n", X.ndims[mode], R);
    ptiAssert(ptiNewMatrix(&U, X.ndims[mode], R) == 0);
    ptiAssert(ptiConstantMatrix(&U, 1) == 0);
    // ptiAssert(ptiRandomizeMatrix(&U) == 0);

    /* For warm-up caches, timing not included */
    if(dev_id == -2) {
        ptiAssert(ptiSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    } else if(dev_id == -1) {
        ptiAssert(ptiOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    }

    for(int it=0; it<niters; ++it) {
        ptiFreeSemiSparseTensor(&Y);
        if(dev_id == -2) {
            ptiAssert(ptiSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        } else if(dev_id == -1) {
            ptiAssert(ptiOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        }
    }


    if(argc > 7) {
        ptiAssert(ptiSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9) == 0);

        fY = fopen(argv[7], "w");
        ptiAssert(fY != NULL);
        ptiAssert(ptiDumpSparseTensor(&spY, 0, fY) == 0);
        fclose(fY);

        ptiFreeSparseTensor(&spY);
    }

    if (renumber > 0) {
        for(ptiIndex m = 0; m < X.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }
    ptiFreeSemiSparseTensor(&Y);
    ptiFreeMatrix(&U);
    ptiFreeSparseTensor(&X);

    return 0;
}
