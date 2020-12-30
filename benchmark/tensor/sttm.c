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
    printf("         -r RANK\n");
    printf("         -d DEV_ID (Only sequential code is supported here)\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    char ifname[1000];
    FILE *fo = NULL;
    ptiSparseTensor spX, spY;
    ptiSemiSparseTensor X, Y;
    ptiMatrix U;
    ptiIndex R = 16;
    ptiIndex mode = 0;
    int dev_id = -2;
    int niters = 5;
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
            fo = fopen(optarg, "aw");
            ptiAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'm':
            sscanf(optarg, "%"HIPARTI_SCN_INDEX, &mode);
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

    /* Load a sparse tensor from file */
    ptiAssert(ptiLoadSparseTensor(&spX, 1, ifname) == 0);
    ptiSparseTensorStatus(&spX, stdout);
    // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);
    ptiAssert(ptiSparseTensorToSemiSparseTensor(&X, &spX, mode) == 0);
    ptiFreeSparseTensor(&spX);

    ptiAssert(ptiNewMatrix(&U, spX.ndims[mode], R) == 0);
    ptiAssert(ptiConstantMatrix(&U, 1) == 0);
    // ptiAssert(ptiRandomizeMatrix(U[m]) == 0);
    ptiDumpMatrix(&U, stdout); fflush(stdout);

    /* Warm up */
    if(dev_id == -2) { // sequential
        ptiAssert(ptiSemiSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    }

    ptiStartTimer(timer);
    for(int i=0; i<niters; ++i) {
        if(dev_id == -2) { // sequential
            ptiAssert(ptiSemiSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
        }
    }
    ptiStopTimer(timer);
    double aver_time = ptiPrintAverageElapsedTime(timer, niters, "CPU SspTTM");

    if(fo != NULL) {
        ptiAssert(ptiSemiSparseTensorToSparseTensor(&spY, &Y, 1e-9) == 0);
        ptiAssert(ptiDumpSparseTensor(&spY, 1, fo) == 0);
        fclose(fo);
        ptiFreeSparseTensor(&spY);
    }

    ptiFreeSemiSparseTensor(&Y);
    ptiFreeMatrix(&U);
    ptiFreeSemiSparseTensor(&X);
    ptiFreeTimer(timer);
    
    return 0;
}
