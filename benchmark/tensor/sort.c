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
    printf("         -m MODE, --mode=MODE\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits) (required when -s 3) \n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits) (required when -s 3) \n");
    printf("         -s sortcase, --sortcase=SORTCASE (0,1,2,3,4)\n");
    printf("         -t NTHREADS, --nthreads=NTHREADS\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    FILE *fi = NULL, *fo = NULL;
    ptiSparseTensor X;
    /* sortcase:
     * = 0 : the same with the old COO code.
     * = 1 : best case. Sort order: [mode, (ordered by increasing dimension sizes)]
     * = 2 : worse case. Sort order: [(ordered by decreasing dimension sizes)]
     * = 3 : Z-Morton ordering (same with HiCOO format order)
     * = 4 : random shuffling.
     */
    int sortcase = 0;
    ptiIndex mode = 0;
    int nthreads = 1;
    ptiElementIndex sb_bits = 7;
    ptiElementIndex sk_bits = sb_bits;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},//input file
            {"output", optional_argument, 0, 'o'},
            {"mode", optional_argument, 0, 'm'},
            {"bs", optional_argument, 0, 'b'},
            {"ks", optional_argument, 0, 'k'},
            {"sortcase", optional_argument, 0, 's'},
            {"nthreads", optional_argument, 0, 't'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:o:m:b:k:s:t:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            fi = fopen(optarg, "r");
            ptiAssert(fi != NULL);
            printf("input file: %s\n", optarg); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            ptiAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'm':
            sscanf(optarg, "%"HIPARTI_SCN_INDEX, &mode);
            break;
        case 'b':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sk_bits);
            break;
        case 's':
            sscanf(optarg, "%d", &sortcase);
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

    printf("sortcase: %d\n", sortcase);
    printf("mode: %"HIPARTI_PRI_INDEX "\n", mode);

    /* Load a sparse tensor from file as it is */
    ptiAssert(ptiLoadSparseTensor(&X, 1, fi) == 0);
    fclose(fi);
    ptiSparseTensorStatus(&X, stdout);
    // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);

    ptiIndex * mode_order = (ptiIndex*) malloc(X.nmodes * sizeof(*mode_order));

    /* Sort sparse tensor */
    memset(mode_order, 0, X.nmodes * sizeof(*mode_order));
    ptiStartTimer(timer);
    switch (sortcase) {
        case 0:
            ptiSparseTensorSortIndex(&X, 1, nthreads);
            break;
        case 1:
            ptiGetBestModeOrder(mode_order, mode, X.ndims, X.nmodes);
            ptiSparseTensorSortIndexCustomOrder(&X, mode_order, 1, nthreads);
            break;
        case 2:
            ptiGetWorstModeOrder(mode_order, mode, X.ndims, X.nmodes);
            ptiSparseTensorSortIndexCustomOrder(&X, mode_order, 1, nthreads);
            break;
        case 3:
            /* Sort tensor in the same way with the one used in HiCOO. */
            ptiSparseTensorMixedOrder(&X, sb_bits, sk_bits, nthreads);
            break;
        case 4:
            ptiGetRandomShuffleElements(&X);
            break;
        default:
            printf("Wrong sortcase number, reset by -s. \n");
    }
    ptiStopTimer(timer);

    /* Print the status of the sorted tensor */
    if(sortcase != 0) {
        printf("mode_order:\n");
        ptiDumpIndexArray(mode_order, X.nmodes, stdout);
    }
    ptiSparseTensorStatus(&X, stdout);
    // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);
  
    double aver_time = ptiPrintElapsedTime(timer, "CPU SORT");

    if(fo != NULL) {
        ptiAssert(ptiDumpSparseTensor(&X, 1, fo) == 0);
        fclose(fo);
    }

    ptiFreeSparseTensor(&X);
    free(mode_order);
    ptiFreeTimer(timer);

    return 0;
}
