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
#include <ParTI.h>
#include <assert.h>

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -X FIRST INPUT TENSOR\n");
    printf("         -Y FIRST INPUT TENSOR\n");
    printf("         -Z OUTPUT TENSOR (Optinal)\n");
    printf("         -c NUMBER OF CONTRACT MODES\n");
    printf("         -x CONTRACT MODES FOR TENSOR X (0-based)\n");
    printf("         -y CONTRACT MODES FOR TENSOR Y (0-based)\n");
    printf("         -m MODES FOR TENSOR Z (0-based)\n");
    printf("         -t NTHREADS, --nt=NT (Optinal)\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char *argv[]) {

    char Xfname[1000], Yfname[1000];
    char Xfname2[1000], Yfname2[1000];
    FILE *fZ = NULL, *fZ2 = NULL;
    sptSparseTensor X, Y, Z;
    sptSparseTensor X2, Y2, Z2; // for 2nd contraction
    sptIndex * cmodes_X = NULL, * cmodes_Y = NULL;
    sptIndex * cmodes_X2 = NULL, * cmodes_Y2 = NULL;  // for 2nd contraction
    sptIndex * modes_Z = NULL, * modes_Z2 = NULL;
    sptIndex nmodes_Z, nmodes_Z2;
    sptIndex num_cmodes = 1, num_cmodes_2 = 1;
    int cuda_dev_id = -2;
    int output_sorting=1;
    int niters = 5;
    int placement = 0;
    int nt = 1;
    int opt_summation = 0; // 0: no sum; 1: ours; 2: linear search

    if(argc < 3) {
        print_usage(argv);
        exit(-1);
    }

    static struct option long_options[] = {
        {"X", required_argument, 0, 'X'},
        {"Y", required_argument, 0, 'Y'},
        {"compressed-nmodes", required_argument, 0, 'c'},
        {"modes_Z", required_argument, 0, 'm'},
        {"x", required_argument, 0, 'x'},
        {"y", required_argument, 0, 'y'},
        {"Z", optional_argument, 0, 'Z'},
        {"o", optional_argument, 0, 'o'},
        {"p", optional_argument, 0, 'p'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {"nt", optional_argument, 0, 't'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };

    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "X:Y:c:x:y:m:o:p:Z:d:t:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'X':
            strcpy(Xfname, optarg);
            sscanf(argv[optind], "%s", Xfname2); 
            printf("1st TC: 1st tensor file: %s\n", Xfname);
            printf("1st TC: 2nd tensor file: %s\n", Xfname2);
            sptAssert(sptLoadSparseTensor(&X, 1, Xfname) == 0);
            sptSparseTensorStatus(&X, stdout);
            sptAssert(sptLoadSparseTensor(&X2, 1, Xfname2) == 0);
            sptSparseTensorStatus(&X2, stdout);
            // printf("Original Tensors: \n"); 
            // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);
            // sptAssert(sptDumpSparseTensor(&X2, 0, stdout) == 0);
            break;
        case 'Y':
            strcpy(Yfname, optarg);
            sscanf(argv[optind], "%s", Yfname2); 
            printf("2nd TC: 2nd tensor file: %s\n", Yfname);
            printf("2nd TC: 2nd tensor file: %s\n", Yfname2);
            sptAssert(sptLoadSparseTensor(&Y, 1, Yfname) == 0);
            sptSparseTensorStatus(&Y, stdout);
            sptAssert(sptLoadSparseTensor(&Y2, 1, Yfname2) == 0);
            sptSparseTensorStatus(&Y2, stdout);   
            // sptAssert(sptDumpSparseTensor(&Y, 0, stdout) == 0);  
            // sptAssert(sptDumpSparseTensor(&Y2, 0, stdout) == 0);
            break;
        case 'Z':
            fZ = fopen(optarg, "w");
            sptAssert(fZ != NULL);
            printf("1st TC: output tensor file: %s\n", optarg);
            fZ2 = fopen(argv[optind], "w");
            sptAssert(fZ2 != NULL);
            printf("2nd TC: output tensor file: %s\n", argv[optind]);
            break;
        case 'c':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &num_cmodes);
            cmodes_X = (sptIndex*)malloc(num_cmodes * sizeof(sptIndex));
            cmodes_Y = (sptIndex*)malloc(num_cmodes * sizeof(sptIndex));
            sptAssert(cmodes_X != NULL && cmodes_Y != NULL);
            sscanf(argv[optind], "%"PARTI_SCN_INDEX, &num_cmodes_2);
            cmodes_X2 = (sptIndex*)malloc(num_cmodes_2 * sizeof(sptIndex));
            cmodes_Y2 = (sptIndex*)malloc(num_cmodes_2 * sizeof(sptIndex));
            sptAssert(cmodes_X2 != NULL && cmodes_Y2 != NULL);
            printf("num_cmodes: %u, num_cmodes_2: %u\n", num_cmodes, num_cmodes_2);
            break;
        case 'x':
            for(sptIndex i = 0; i < num_cmodes; ++ i) {
                // Only can input an array directly from argv not optarg.
                sscanf(argv[optind - 1], "%u", &(cmodes_X[i])); 
                ++ optind;
            }
            sptDumpIndexArray(cmodes_X, num_cmodes, stdout);
            for(sptIndex i = 0; i < num_cmodes_2; ++ i) {
                sscanf(argv[optind - 1], "%u", &(cmodes_X2[i])); 
                ++ optind;
            }
            sptDumpIndexArray(cmodes_X2, num_cmodes_2, stdout);
            optind -= num_cmodes;
            optind -= num_cmodes_2;
            break;
        case 'y':
            for(sptIndex i = 0; i < num_cmodes; ++ i) {
                // Only can input an array directly from argv not optarg.
                sscanf(argv[optind - 1], "%u", &(cmodes_Y[i])); 
                ++ optind;
            }
            sptDumpIndexArray(cmodes_Y, num_cmodes, stdout);
            for(sptIndex i = 0; i < num_cmodes_2; ++ i) {
                sscanf(argv[optind - 1], "%u", &(cmodes_Y2[i])); 
                ++ optind;
            }
            sptDumpIndexArray(cmodes_Y2, num_cmodes_2, stdout);
            optind -= num_cmodes;
            optind -= num_cmodes_2;
            break;
        case 'm':
            nmodes_Z = X.nmodes + Y.nmodes - 2 * num_cmodes;
            printf("nmodes_Z: %u\n", nmodes_Z); fflush(stdout);
            modes_Z = malloc(nmodes_Z * sizeof *modes_Z);
            sptAssert(modes_Z != NULL);
            for(sptIndex i = 0; i < nmodes_Z; ++ i) {
                // Only can input an array directly from argv not optarg.
                sscanf(argv[optind - 1], "%u", &(modes_Z[i])); 
                ++ optind;
            }
            sptDumpIndexArray(modes_Z, nmodes_Z, stdout); fflush(stdout);
            nmodes_Z2 = X2.nmodes + Y2.nmodes - 2 * num_cmodes_2;
            modes_Z2 = malloc(nmodes_Z2 * sizeof *modes_Z2);
            sptAssert(modes_Z2 != NULL);
            for(sptIndex i = 0; i < nmodes_Z2; ++ i) {
                // Only can input an array directly from argv not optarg.
                sscanf(argv[optind - 1], "%u", &(modes_Z2[i])); 
                ++ optind;
            }
            sptDumpIndexArray(modes_Z2, nmodes_Z2, stdout); fflush(stdout);
            optind -= nmodes_Z;
            optind -= nmodes_Z2;
            break;
        case 'o':
            sscanf(optarg, "%d", &output_sorting);
            break;    
        case 'p':
            sscanf(optarg, "%d", &placement);
            break;      
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 't':
            sscanf(optarg, "%d", &nt);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    omp_set_num_threads(nt);
    printf("1st TC: #Contraction modes: %"PARTI_PRI_INDEX"\n", num_cmodes);
    printf("2nd TC: #Contraction modes: %"PARTI_PRI_INDEX"\n", num_cmodes_2);

    /* For warm-up caches, timing not included */
    sptAssert(sptSparseTensorMulTensor2TCs(&Z, &X, &Y, num_cmodes, cmodes_X, cmodes_Y, modes_Z, &Z2, &X2, &Y2, num_cmodes_2, cmodes_X2, cmodes_Y2, modes_Z2, nt, output_sorting, opt_summation, placement) == 0);

    // for(int it=0; it<niters; ++it) {
    //     sptFreeSparseTensor(&Z);
    // }

    sptSparseTensorStatus(&Z, stdout);
    sptSparseTensorStatus(&Z2, stdout);
    // sptAssert(sptDumpSparseTensor(&Z, 0, stdout) == 0);

    if(fZ != NULL) {
        // sptSparseTensorSortIndex(&Z, 1, 1);
        //sptAssert(sptDumpSparseTensor(&Z, 0, fZ) == 0);
        fclose(fZ);
    }

    sptFreeSparseTensor(&Y);
    sptFreeSparseTensor(&X);
    sptFreeSparseTensor(&Y2);
    sptFreeSparseTensor(&X2);
    sptFreeSparseTensor(&Z);
    sptFreeSparseTensor(&Z2);

    return 0;
}
