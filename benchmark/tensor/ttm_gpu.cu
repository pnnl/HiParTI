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
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -m MODE, --mode=MODE\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -r RANK\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=CUDA_DEV_ID\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    char ifname[1000];
    FILE *fo = NULL;
    ptiSparseTensor X, spY;
    ptiSemiSparseTensor Y;
    ptiMatrix U;
    ptiIndex mode = 0;
    ptiIndex R = 16;
    int cuda_dev_id = 0;
    int niters = 5;
    int impl_num = 15;
    ptiNnzIndex smem_size = 160000;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    if(argc < 3) {
        print_usage(argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},//input file
            {"mode", optional_argument, 0, 'm'},//
            {"output", optional_argument, 0, 'o'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"impl-num", optional_argument, 0, 'p'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:d:r:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            strcpy(ifname, optarg);
            printf("input file: %s\n", optarg); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            ptiAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'm':
            sscanf(optarg, "%" HIPARTI_SCN_INDEX, &mode);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%" HIPARTI_SCN_INDEX, &R);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    printf("mode: %" HIPARTI_PRI_INDEX "\n", mode);
    printf("cuda_dev_id: %d\n", cuda_dev_id);

    /* Load a sparse tensor from file */
    ptiAssert(ptiLoadSparseTensor(&X, 1, ifname) == 0);
    ptiSparseTensorStatus(&X, stdout);

    printf("ptiRandomizeMatrix(&U, %" HIPARTI_PRI_INDEX ", %" HIPARTI_PRI_INDEX ")\n", X.ndims[mode], R);
    ptiAssert(ptiNewMatrix(&U, X.ndims[mode], R) == 0);
    ptiAssert(ptiConstantMatrix(&U, 1) == 0);
    // ptiAssert(ptiRandomizeMatrix(&U) == 0);

    /* For warm-up caches, timing not included */
    ptiCudaSetDevice(cuda_dev_id);
    // ptiAssert(ptiCudaSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    ptiAssert(ptiCudaSparseTensorMulMatrixOneKernel(&Y, &X, &U, mode, impl_num, smem_size) == 0);

    ptiStartTimer(timer);
    for(int it=0; it<niters; ++it) {
        ptiFreeSemiSparseTensor(&Y);
        ptiCudaSetDevice(cuda_dev_id);
        // ptiAssert(ptiCudaSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        ptiAssert(ptiCudaSparseTensorMulMatrixOneKernel(&Y, &X, &U, mode, impl_num, smem_size) == 0);
    }
    ptiStopTimer(timer);
    double aver_time = ptiPrintAverageElapsedTime(timer, niters, "GPU SpTTM");

    if(fo != NULL) {
        ptiAssert(ptiSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9) == 0);
        ptiAssert(ptiDumpSparseTensor(&spY, 0, fo) == 0);
        fclose(fo);
        ptiFreeSparseTensor(&spY);
    }

    ptiFreeSemiSparseTensor(&Y);
    ptiFreeMatrix(&U);
    ptiFreeSparseTensor(&X);
    ptiFreeTimer(timer);

    return 0;
}
