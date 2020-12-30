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
#ifdef HIPARTI_USE_OPENMP
    #include <omp.h>
#endif
#include "../src/sptensor/hicoo/hicoo.h"

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -e RENUMBER, --renumber=RENUMBER\n");
    printf("         -n NITERS_RENUM\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t NTHREADS, --nt=NT\n");
    printf("         -u use_reduce, --ur=use_reduce\n");
    printf("         --help\n");
    printf("\n");
}


int main(int argc, char ** argv) {
    char ifname[1000];
    FILE *fo = NULL;
    ptiSparseTensor X;
    ptiIndex R = 16;
    ptiIndex niters = 1; //5; // 50
    double tol = 1e-5;
    ptiKruskalTensor ktensor;
    int nloops = 0;
    int cuda_dev_id = -2;
    int nthreads = 1;
    int use_reduce = 0;
    int impl_num = 0;
    int renumber = 0;
    int niters_renum = 3;
    /* renumber:
     * = 0 : no renumbering.
     * = 1 : renumber with Lexi-order
     * = 2 : renumber with BFS-like
     * = 3 : randomly renumbering, specify niters_renum.
     */

    if(argc < 2) {
        print_usage(argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"output", optional_argument, 0, 'o'},
            {"impl-num", optional_argument, 0, 'p'},
            {"renumber", optional_argument, 0, 'e'},
            {"niters-renum", optional_argument, 0, 'n'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"nt", optional_argument, 0, 't'},
            {"use-reduce", optional_argument, 0, 'u'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:o:p:e:n:d:r:t:u:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            strcpy(ifname, optarg);
            printf("Input file: %s\n", ifname); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            ptiAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'p':
            sscanf(optarg, "%d", &impl_num);
            break;
        case 'e':
            sscanf(optarg, "%d", &renumber);
            break;
        case 'n':
            sscanf(optarg, "%d", &niters_renum);
            break;
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 'r':
            sscanf(optarg, "%u"HIPARTI_SCN_INDEX, &R);
            break;
        case 'u':
            sscanf(optarg, "%d", &use_reduce);
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
    printf("cuda_dev_id: %d\n", cuda_dev_id);
    printf("renumber: %d\n", renumber);
    if (renumber == 1)
        printf("niters_renum: %d\n\n", niters_renum);

    ptiAssert(ptiLoadSparseTensor(&X, 1, ifname) == 0);
    ptiSparseTensorStatus(&X, stdout);
    // ptiDumpSparseTensor(&X, 0, stdout);


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
            #if 0
            orderit(&X, map_inds, renumber, niters_renum);
            #else
            // Fix sb_bits = 7
            ptiIndexRenumber(&X, map_inds, renumber, niters_renum, 7, nthreads, impl_num);
            #endif
            // orderforHiCOO((int)(X.nmodes), (ptiIndex)X.nnz, X.ndims, X.inds, map_inds);
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

        // ptiSparseTensorSortIndex(&X, 1);
        // printf("map_inds:\n");
        // for(ptiIndex m = 0; m < X.nmodes; ++m) {
        //     ptiDumpIndexArray(map_inds[m], X.ndims[m], stdout);
        // }
        // ptiAssert(ptiDumpSparseTensor(&X, 0, stdout) == 0);
    }    

    ptiIndex nmodes = X.nmodes;
    ptiNewKruskalTensor(&ktensor, nmodes, X.ndims, R);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        nthreads = 1;
        ptiAssert(ptiCpdAls(&X, R, niters, tol, &ktensor) == 0);
    } else if(cuda_dev_id == -1) {
        omp_set_num_threads(nthreads);
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("nthreads: %d\n", nthreads);
        printf("use_reduce: %d\n", use_reduce);
        ptiAssert(ptiOmpCpdAls(&X, R, niters, tol, nthreads, use_reduce, &ktensor) == 0);
    }

    // for(int it=0; it<nloops; ++it) {
    //     if(cuda_dev_id == -2) {
    //         nthreads = 1;
    //         ptiAssert(ptiCpdAls(&X, R, niters, tol, &ktensor) == 0);
    //     } else if(cuda_dev_id == -1) {
    //         #pragma omp parallel
    //         {
    //             nthreads = omp_get_num_threads();
    //         }
    //         printf("nthreads: %d\n", nthreads);
    //         ptiAssert(ptiOmpCpdAls(&X, R, niters, tol, nthreads, use_reduce, &ktensor) == 0);
    //     }
    // }

    if(fo != NULL) {
        // Dump ktensor to files
        if (renumber > 0) {
            ptiKruskalTensorInverseShuffleIndices(&ktensor, map_inds);
        }
        ptiAssert( ptiDumpKruskalTensor(&ktensor, fo) == 0 );
        fclose(fo);
    }

    if (renumber > 0) {
        for(ptiIndex m = 0; m < X.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }
    ptiFreeSparseTensor(&X);
    ptiFreeKruskalTensor(&ktensor);

    return 0;
}
