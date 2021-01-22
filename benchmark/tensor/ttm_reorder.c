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
#include <HiParTI.h>
#include "../src/sptensor/sptensor.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -m MODE, --mode=MODE (default -1: loop all modes)\n");
    printf("         -e RENUMBER, --renumber=RENUMBER\n");
    printf("         -n NITERS_RENUM\n");
    printf("         -d DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char const *argv[]) {
    char ifname[1000];
    FILE *fo = NULL;
    ptiSparseTensor X, spY;
    ptiSemiSparseTensor Y;
    ptiMatrix U;
    ptiIndex mode = 0;
    ptiIndex R = 16;
    int dev_id = -2;
    int niters = 5;
    int renumber = 0;
    /* renumber:
     * = 0 : no renumbering.
     * = 1 : renumber with Lexi-order
     * = 2 : renumber with BFS-like
     * = 3 : randomly renumbering, specify niters_renum.
     */
    int niters_renum = 5;

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"mode", required_argument, 0, 'm'},
        {"output", optional_argument, 0, 'o'},
        {"renumber", optional_argument, 0, 'e'},
        {"niters-renum", optional_argument, 0, 'n'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {"rank", optional_argument, 0, 'r'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };

    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:e:d:r:n:", long_options, &option_index);
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
        case 'm':
            sscanf(optarg, "%"HIPARTI_SCN_INDEX, &mode);
            break;
        case 'e':
            sscanf(optarg, "%d", &renumber);
            break;
        case 'n':
            sscanf(optarg, "%d", &niters_renum);
            break;
        case 'd':
            sscanf(optarg, "%d", &dev_id);
            break;
        case 'r':
            sscanf(optarg, "%u"HIPARTI_SCN_INDEX, &R);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("mode: %"HIPARTI_PRI_INDEX "\n", mode);
    printf("dev_id: %d\n", dev_id);
    printf("renumber: %d\n", renumber);
    if (renumber == 1)
        printf("niters_renum: %d\n\n", niters_renum);

    /* Load a sparse tensor from file as it is */
    ptiAssert(ptiLoadSparseTensor(&X, 1, ifname) == 0);
    ptiSparseTensorStatus(&X, stdout);
    // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);


    ptiElementIndex sb_bits = 7;
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

    if (renumber > 0) {
        ptiMatrixInverseShuffleIndices(&U, map_inds[mode]);
        // ptiAssert(ptiDumpMatrix(U[nmodes], stdout) == 0);
    }

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


    if(fo != NULL) {
        ptiAssert(ptiSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9) == 0);

        ptiAssert(ptiDumpSparseTensor(&spY, 0, fo) == 0);
        fclose(fo);

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
