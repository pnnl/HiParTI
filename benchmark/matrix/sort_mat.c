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
#include <stdlib.h>
#include <getopt.h>
#include <HiParTI.h>

int main(int argc, char *argv[]) {
    FILE *fi = NULL, *fo = NULL;
    ptiSparseMatrix mtx;
    int nthreads = 1;   // get from OMP_NUM_THREADS environment
    int sort = 0;
    ptiElementIndex block_bits = 0;
    /* sort:
     * = 0 : No sorting.
     * = 1 : Natural row-sorting.
     * = 2 : Natural column-sorting.
     * = 3 : Natural block-sorting.
     * = 4 : Z-Morton block-sorting.
     */

    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"sort", optional_argument, 0, 's'},
        {"block_bits", optional_argument, 0, 'b'},
        {0, 0, 0, 0}
    };

    for(;;) {
        int option_index = 0;
        int c = 1;
        c = getopt_long(argc, argv, "i:o:s:b:", long_options, &option_index);
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
        case 's':
            sscanf(optarg, "%d", &sort);
            break;
        case 'b':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &block_bits);
            break;
        default:
            abort();
        }
    }
    printf("sort: %d\n", sort);
#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel
    nthreads = omp_get_num_threads();
#endif
    printf("nthreads: %d\n", nthreads);
    if (sort == 3 || sort == 4) {
        printf("block size: %d\n", (int)pow(2,block_bits));
        ptiAssert(block_bits > 0);
    }


    if(optind > argc || argc < 3) {
        printf("Usage: %s\n", argv[0]);
        printf("Options: -i INPUT, --input=INPUT\n");
        printf("         -o OUTPUT, --output=OUTPUT\n");
        printf("         -s SORT\n");
        printf("         -b block_bits\n");
        printf("\n");
        return -1;
    }

    ptiAssert(ptiLoadSparseMatrix(&mtx, 1, fi) == 0);
    fclose(fi);

    /* Sort the input matrix */
    ptiStartTimer(timer);
    switch(sort)
    {
        case 0:
            printf("[No Sorting]\n");
            break;
        case 1:
            printf("[Row-Sorting]\n");
            ptiSparseMatrixSortIndexSingleMode(&mtx, 1, 0, nthreads);
            break;
        case 2:
            printf("[Column-Sorting]\n");
            ptiSparseMatrixSortIndexSingleMode(&mtx, 1, 1, nthreads);
            break;
        case 3:
            printf("[Natural Block-Sorting]\n");
            ptiSparseMatrixSortIndexRowBlock(&mtx, 1, 0, mtx.nnz, block_bits);
            break;
        case 4:
            printf("[Z-Morton Block-Sorting]\n");
            ptiSparseMatrixSortIndexMorton(&mtx, 1, 0, mtx.nnz, block_bits);
            break;
        default:
            printf("[Wrong defined sort types]\n");
    }
    ptiStopTimer(timer);
    ptiPrintElapsedTime(timer, "Sort");

    if (fo != NULL) {
        ptiAssert(ptiDumpSparseMatrix(&mtx, 1, fo) == 0);
        fclose(fo);
    }

    ptiFreeSparseMatrix(&mtx);
    ptiFreeTimer(timer);
    return 0;
}
