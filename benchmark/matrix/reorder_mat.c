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

#include <stdio.h>
#include <getopt.h>
#include <HiParTI.h>

int main(int argc, char *argv[]) {
    FILE *fi = NULL, *fo = NULL;
    ptiSparseMatrix mtx;
    int relabel = 1 ;
    int niters_renum = 5;
    /* relabel:
     * = 0 : no relabeling.
     * = 1 : relabel with Lexi-order, specify niters_renum.
     * = 2 : randomly relabeling
     */

    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"relabel", optional_argument, 0, 'r'},
        {0, 0, 0, 0}
    };

    for(;;) {
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:r:", long_options, &option_index);
        if(c == -1) {
            break;
        }

        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            ptiAssert(fi != NULL);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            ptiAssert(fo != NULL);
            break;
        case 'r':
            sscanf(optarg, "%d", &relabel);
            break;
        default:
            abort();
        }
    }
    printf("relabel: %d\n", relabel);
    if (relabel == 1)
        printf("niters_renum: %d\n\n", niters_renum);

    if(optind > argc || argc < 3) {
        printf("Usage: %s\n", argv[0]);
        printf("Options: -i INPUT, --input=INPUT\n");
        printf("         -o OUTPUT, --output=OUTPUT\n");
        printf("         -r RELABEL (1: Lexi-order (default); 2: random ordering)\n");
        printf("\n");
        return -1;
    }

    ptiAssert(ptiLoadSparseMatrix(&mtx, 1, fi) == 0);
    fclose(fi);
    ptiSparseMatrixStatus(&mtx, stdout);

    /* Relabel the input matrix */
    ptiIndex nmodes = 2; // for matrices
    ptiIndex ** map_inds;
    if (relabel > 0) {
        map_inds = (ptiIndex **)malloc(nmodes * sizeof *map_inds);
        map_inds[0] = (ptiIndex *)malloc(mtx.nrows * sizeof (ptiIndex));
        for(ptiIndex i = 0; i < mtx.nrows; ++i)
            map_inds[0][i] = i;
        map_inds[1] = (ptiIndex *)malloc(mtx.ncols * sizeof (ptiIndex));
        for(ptiIndex i = 0; i < mtx.ncols; ++i)
            map_inds[1][i] = i;

        ptiStartTimer(timer);

        if ( relabel == 1) { /* Set the Lexi-order relabeling */
            ptiIndexRelabel(&mtx, map_inds, relabel, niters_renum, 1);
        }
        if ( relabel == 3) { /* Set randomly renumbering */
            printf("[Random Indexing]\n");
            ptiGetRandomShuffledIndicesMat(&mtx, map_inds);
        }

        ptiStopTimer(timer);
        ptiPrintElapsedTime(timer, "Relabeling");

        ptiStartTimer(timer);

        ptiSparseMatrixShuffleIndices(&mtx, map_inds);

        ptiStopTimer(timer);
        ptiPrintElapsedTime(timer, "Shuffling time");
        printf("\n");

        // ptiSparseTensorSortIndex(&X, 1, 1);
        // printf("map_inds:\n");
        // for(ptiIndex m = 0; m < X.nmodes; ++m) {
        //     ptiDumpIndexArray(map_inds[m], X.ndims[m], stdout);
        // }
        // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);
    }

    if (fo != NULL) {
        ptiAssert(ptiDumpSparseMatrix(&mtx, 1, fo) == 0);
        fclose(fo);
    }

    ptiFreeSparseMatrix(&mtx);
    ptiFreeTimer(timer);
    return 0;
}
