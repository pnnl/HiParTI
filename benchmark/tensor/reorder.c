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
#include <getopt.h>
#include <omp.h>
#include <HiParTI.h>
#include "../src/sptensor/sptensor.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -e RENUMBER (1: Lexi-order, default; 3: random reordering)\n");
    printf("         -n NITERS_RENUM (an integer in [3,10]. 5 by default)\n");
    printf("         -p IMPL_NUM (1: default)\n");
    printf("         -b BLOCKSIZE (bits) (block size in bits, 7 by default\n");
    printf("         -t NTHREADS (1: default)\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    char ifname[1000];
    FILE *fo = NULL;
    ptiSparseTensor X;

    int nthreads = 1;
    int impl_num = 1;
    int renumber = 1;
    int niters_renum = 5;
    /* renumber:
     * = 0 : no renumbering.
     * = 1 : renumber with Lexi-order, specify niters_renum.
     * = 2 : renumber with BFS-like
     * = 3 : randomly renumbering
     */
    ptiElementIndex sb_bits = 7;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    if(argc < 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", optional_argument, 0, 'o'},
            {"bs", optional_argument, 0, 'b'},
            {"impl-num", optional_argument, 0, 'p'},
            {"renumber", optional_argument, 0, 'e'},
            {"niters-renum", optional_argument, 0, 'n'},
            {"nthreads", optional_argument, 0, 't'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:o:b:p:e:t:n:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            strcpy(ifname, optarg);
            printf("input file: %s\n", optarg); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "aw");
            ptiAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'b':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 'e':
            sscanf(optarg, "%d", &renumber);
            break;
        case 'n':
            sscanf(optarg, "%d", &niters_renum);
            break;
        case 't':
            sscanf(optarg, "%d", &nthreads);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }

    ptiAssert(renumber > 0);
    printf("renumber: %d\n", renumber);
    if (renumber == 1)
        printf("niters_renum: %d\n\n", niters_renum);

    /* Load a sparse tensor from file as it is */
    ptiAssert(ptiLoadSparseTensor(&X, 1, ifname) == 0);
    ptiSparseTensorStatus(&X, stdout);
    // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);

    /* Renumber the input tensor */
    ptiIndex ** map_inds;
    
    map_inds = (ptiIndex **)malloc(X.nmodes * sizeof *map_inds);
    pti_CheckOSError(!map_inds, "REORDER");
    for(ptiIndex m = 0; m < X.nmodes; ++m) {
        map_inds[m] = (ptiIndex *)malloc(X.ndims[m] * sizeof (ptiIndex));
        pti_CheckError(!map_inds[m], "REORDER", NULL);
        for(ptiIndex i = 0; i < X.ndims[m]; ++i)
            map_inds[m][i] = i;
    }

    ptiStartTimer(timer);

    if ( renumber == 1 || renumber == 2) { /* Set the Lexi-order or BFS-like renumbering */
        ptiIndexRenumber(&X, map_inds, renumber, niters_renum, sb_bits, nthreads, impl_num);
    }
    if ( renumber == 3) { /* Set randomly renumbering */
        printf("[Random Indexing]\n");
        ptiGetRandomShuffledIndices(&X, map_inds);
    }

    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "Renumbering");

    ptiStartTimer(timer);
    ptiSparseTensorShuffleIndices(&X, map_inds);
    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "Shuffling");
    printf("\n");

    /* Print output tensor status */
    ptiSparseTensorStatus(&X, stdout);

    // ptiSparseTensorSortIndex(&X, 1, 1);
    // printf("map_inds:\n");
    // for(ptiIndex m = 0; m < X.nmodes; ++m) {
    //     ptiDumpIndexArray(map_inds[m], X.ndims[m], stdout);
    // }
    // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);

    if(fo != NULL) {
        ptiAssert(ptiDumpSparseTensor(&X, 1, fo) == 0);
        fclose(fo);
    }
    if (renumber > 0) {
        for(ptiIndex m = 0; m < X.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }

    ptiFreeSparseTensor(&X);
    ptiFreeTimer(timer);

    return 0;
}
