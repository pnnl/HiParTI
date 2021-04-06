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
    FILE *fZ = NULL;
    sptSparseTensor X, Y, Z;
    sptIndex * cmodes_X = NULL, * cmodes_Y = NULL;
    sptIndex * modes_Z = NULL;
    sptIndex nmodes_Z;
    sptIndex num_cmodes = 1;
    int output_sorting=1;
    int niters = 5;
    int placement = 0;
    int nt = 1;
    int opt_summation = 1; // 0: no sum; 1: ours; 2: linear search

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
        {"nt", optional_argument, 0, 't'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };

    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "X:Y:c:x:y:m:o:p:Z:t:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'X':
            strcpy(Xfname, optarg);
            printf("1st tensor file: %s\n", Xfname);
            sptAssert(sptLoadSparseTensor(&X, 1, Xfname) == 0);
            sptSparseTensorStatus(&X, stdout);
            // printf("Original Tensors: \n"); 
            // sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);
            break;
        case 'Y':
            strcpy(Yfname, optarg);
            printf("2nd tensor file: %s\n", Yfname);
            sptAssert(sptLoadSparseTensor(&Y, 1, Yfname) == 0);
            sptSparseTensorStatus(&Y, stdout);   
            // sptAssert(sptDumpSparseTensor(&Y, 0, stdout) == 0); 
            break;
        case 'Z':
            fZ = fopen(optarg, "w");
            sptAssert(fZ != NULL);
            printf("output tensor file: %s\n", optarg);
            break;
        case 'c':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &num_cmodes);
            cmodes_X = (sptIndex*)malloc(num_cmodes * sizeof(sptIndex));
            cmodes_Y = (sptIndex*)malloc(num_cmodes * sizeof(sptIndex));
            sptAssert(cmodes_X != NULL && cmodes_Y != NULL);
            printf("%s\n", optarg);
            break;
        case 'x':
            for(sptIndex i = 0; i < num_cmodes; ++ i) {
                // Only can input an array directly from argv not optarg.
                sscanf(argv[optind - 1], "%u", &(cmodes_X[i])); 
                ++ optind;
            }
            optind -= num_cmodes;
            break;
        case 'y':
            for(sptIndex i = 0; i < num_cmodes; ++ i) {
                // Only can input an array directly from argv not optarg.
                sscanf(argv[optind - 1], "%u", &(cmodes_Y[i])); 
                ++ optind;
            }
            optind -= num_cmodes;
            break;
        case 'm':
            nmodes_Z = X.nmodes + Y.nmodes - 2 * num_cmodes;
            modes_Z = malloc(nmodes_Z * sizeof *modes_Z);
            sptAssert(modes_Z != NULL);
            for(sptIndex i = 0; i < nmodes_Z; ++ i) {
                // Only can input an array directly from argv not optarg.
                sscanf(argv[optind - 1], "%u", &(modes_Z[i])); 
                ++ optind;
            }
            optind -= nmodes_Z;
            break;
        case 'o':
            sscanf(optarg, "%d", &output_sorting);
            break;    
        case 'p':
            sscanf(optarg, "%d", &placement);
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
    printf("#Contraction modes: %"PARTI_PRI_INDEX"\n", num_cmodes);
    // sptDumpIndexArray(cmodes_X, num_cmodes, stdout);
    // sptDumpIndexArray(cmodes_Y, num_cmodes, stdout);
    // sptDumpIndexArray(modes_Z, nmodes_Z, stdout);

    int experiment_modes = 3;
    // sscanf(getenv("EXPERIMENT_MODES"), "%d", &experiment_modes);

    /* For warm-up caches, timing not included */ 
    sptAssert(sptSparseTensorMulTensor(&Z, &X, &Y, num_cmodes, cmodes_X, cmodes_Y, modes_Z, nt, output_sorting, opt_summation, placement) == 0);

    // for(int it=0; it<niters; ++it) {
    //     sptFreeSparseTensor(&Z);
    // }

    sptSparseTensorStatus(&Z, stdout);
    //sptAssert(sptDumpSparseTensor(&Z, 0, stdout) == 0);

    if(fZ != NULL) {
        // sptSparseTensorSortIndex(&Z, 1, 1);
        sptAssert(sptDumpSparseTensor(&Z, 0, fZ) == 0);
        fclose(fZ);
    }

    sptFreeSparseTensor(&Y);
    sptFreeSparseTensor(&X);
    sptFreeSparseTensor(&Z);
    free(modes_Z);
    free(cmodes_X);
    free(cmodes_Y);

    return 0;
}
