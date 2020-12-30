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

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -x X INPUT\n");
    printf("         -y Y INPUT\n");
    printf("         -o OUTPUT\n");
    printf("         -d CUDA_DEV_ID\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    char Xfname[1000], Yfname[1000];
    FILE *fo = NULL;
    ptiSparseTensor X, Y, Z;
    int cuda_dev_id = -2;
    int niters = 5;
    int nthreads = 1;

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"x-input", required_argument, 0, 'x'},
            {"y-input", required_argument, 0, 'y'},
            {"output", optional_argument, 0, 'o'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "x:y:o:d:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'x':
            strcpy(Xfname, optarg);
            printf("X input file: %s\n", optarg); fflush(stdout);
            break;
        case 'y':
            strcpy(Yfname, optarg);
            printf("Y input file: %s\n", optarg); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            ptiAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("cuda_dev_id: %d\n", cuda_dev_id);

    ptiAssert(ptiLoadSparseTensor(&X, 1, Xfname) == 0);
    ptiAssert(ptiLoadSparseTensor(&Y, 1, Yfname) == 0);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        ptiAssert(ptiSparseTensorDotMulEq(&Z, &X, &Y) == 0);
    } else if(cuda_dev_id == -1) {
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("nthreads: %d\n", nthreads);
        ptiAssert(ptiOmpSparseTensorDotMulEq(&Z, &X, &Y) == 0);
    }

    for(int it=0; it<niters; ++it) {
        if(cuda_dev_id == -2) {
            ptiAssert(ptiSparseTensorDotMulEq(&Z, &X, &Y) == 0);
        } else if(cuda_dev_id == -1) {
            #pragma omp parallel
            {
                nthreads = omp_get_num_threads();
            }
            printf("nthreads: %d\n", nthreads);
            ptiAssert(ptiOmpSparseTensorDotMulEq(&Z, &X, &Y) == 0);
        }
    }

    if (fo != NULL) {
        ptiAssert(ptiDumpSparseTensor(&Z, 1, fo) == 0);
        fclose(fo);
    }

    ptiFreeSparseTensor(&X);
    ptiFreeSparseTensor(&Y);
    ptiFreeSparseTensor(&Z);

    return 0;
}
