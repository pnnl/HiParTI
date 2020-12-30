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

static void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -X INPUT (.tns file)\n");
    printf("         -a INPUT (a scalar)\n");
    printf("         -Z OUTPUT (output file name)\n");
    printf("         Parallel CPU: use 'export OMP_NUM_THREADS = [number]'\n");
    printf("         --help\n");
    printf("\n");
}

/**
 * Benchmark COO tensor multiplication with a scalar. 
 */
int main(int argc, char *argv[]) {
    FILE *fZ = NULL;
    char Xfname[1000];
    ptiValue a;
    ptiSparseTensor X;
    int niters = 5;
    int nthreads;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    if(argc < 3) {
        print_usage(argv);
        exit(1);
    }

    static struct option long_options[] = {
        {"Xinput", required_argument, 0, 'X'},
        {"ainput", required_argument, 0, 'a'},
        {"Zoutput", optional_argument, 0, 'Z'},
        {"dev-id", optional_argument, 0, 'd'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "X:a:Z:d:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'X':
            strcpy(Xfname, optarg);
            printf("X input file: %s\n", Xfname); fflush(stdout);
            break;
        case 'a':
            sscanf(optarg, "%"HIPARTI_SCN_VALUE, &a);
            break;
        case 'Z':
            fZ = fopen(optarg, "w");
            ptiAssert(fZ != NULL);
            printf("Z output file: %s\n", optarg); fflush(stdout);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("Scaling a: %"HIPARTI_PRI_VALUE"\n", a); 

    ptiAssert(ptiLoadSparseTensor(&X, 1, Xfname) == 0);

    /* For warm-up caches, timing not included */
#ifdef HIPARTI_USE_OPENMP
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
#endif        
    ptiAssert(ptiSparseTensorMulScalar(&X, a) == 0);

    ptiStartTimer(timer);
    for(int it=0; it<niters; ++it) {
        ptiAssert(ptiSparseTensorMulScalar(&X, a) == 0);
    }
    ptiStopTimer(timer);
    ptiPrintAverageElapsedTime(timer, niters, "Average CooMulScalar");
    ptiFreeTimer(timer);

    if(fZ != NULL) {
        ptiAssert(ptiDumpSparseTensor(&X, 1, fZ) == 0);
        fclose(fZ);
    }

    ptiFreeSparseTensor(&X);

    return 0;
}
