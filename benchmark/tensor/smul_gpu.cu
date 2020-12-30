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
    printf("         -d CUDA_DEV_ID (>=0:GPU id)\n");
    printf("         --help\n");
    printf("\n");
}

/**
 * Benchmark COO tensor multiplication with a scalar. 
 */
int main(int argc, char *argv[]) {
    FILE *fZ = NULL;
    char Xfname[1000];
    ptiValue a = 1.0;
    ptiSparseTensor X;
    int cuda_dev_id = 0;
    int niters = 5;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    if(argc < 3) {
        print_usage(argv);
        exit(1);
    }

    static struct option long_options[] = {
        {"Xinput", required_argument, 0, 'X'},
        {"ainput", required_argument, 0, 'a'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };
    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "X:a:d:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'X':
            strcpy(Xfname, optarg);
            printf("X input file: %s\n", Xfname); fflush(stdout);
            break;
        case 'a':
            sscanf(optarg, "%" HIPARTI_SCN_VALUE, &a);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            if(cuda_dev_id < -2) {
                fprintf(stderr, "Error: set cuda_dev_id to >=0.\n");
                exit(1);
            }
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("Scaling a: %" HIPARTI_PRI_VALUE"\n", a); 
    printf("cuda_dev_id: %d\n", cuda_dev_id); fflush(stdout);

    ptiAssert(ptiLoadSparseTensor(&X, 1, Xfname) == 0);

    /* For warm-up caches, timing not included */
    ptiCudaSetDevice(cuda_dev_id);
    ptiAssert(ptiCudaSparseTensorMulScalar(&X, a) == 0);

    ptiStartTimer(timer);
    for(int it=0; it<niters; ++it) {
        ptiAssert(ptiCudaSparseTensorMulScalar(&X, a) == 0);
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
