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
#include "../src/sptensor/sptensor.h"
#include "../src/sptensor/hicoo/hicoo.h"
#include <omp.h>


void print_usage(char ** argv) {
    printf("Usage: %s [options] \n", argv[0]);
    printf("Options: -i INPUT, --input=INPUT\n");
    printf("         -o OUTPUT, --output=OUTPUT\n");
    printf("         -b BLOCKSIZE (bits), --blocksize=BLOCKSIZE (bits)\n");
    printf("         -k KERNELSIZE (bits), --kernelsize=KERNELSIZE (bits)\n");
    printf("         -c CHUNKSIZE (bits), --chunksize=CHUNKSIZE (bits, <=9)\n");
    printf("         -e RENUMBER, --renumber=RENUMBER\n");
    printf("         -n NITERS_RENUM\n");
    printf("         -p IMPL_NUM, --impl-num=IMPL_NUM\n");
    printf("         -d CUDA_DEV_ID, --cuda-dev-id=DEV_ID\n");
    printf("         -r RANK\n");
    printf("         -t TK, --tk=TK\n");
    printf("         -l TB, --tb=TB\n");
    printf("         -a balanced\n");
    printf("         --help\n");
    printf("\n");
}


int main(int argc, char ** argv) {
    printf("CPD HiCOO: \n");

    char ifname[1000];
    FILE *fo = NULL;
    ptiSparseTensor tsr;
    ptiSparseTensorHiCOO hitsr;
    ptiRankKruskalTensor ktensor;
    ptiElementIndex sb_bits;
    ptiElementIndex sk_bits;
    ptiElementIndex sc_bits;

    ptiIndex R = 16;
    int cuda_dev_id = -2;
    // int nloops = 1; // 5
    ptiIndex niters = 1; //5; // 50
    double tol = 1e-5;
    int impl_num = 0;
    int renumber = 0;
    int niters_renum = 3;
    /* renumber:
     * = 0 : no renumbering.
     * = 1 : renumber with Lexi-order
     * = 2 : renumber with BFS-like
     * = 3 : randomly renumbering, specify niters_renum.
     */
    int tk = 1;
    int tb = 1;
    int balanced = 0;

    if(argc < 5) { // #Required arguments
        print_usage(argv);
        exit(1);
    }


    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},
            {"bs", required_argument, 0, 'b'},
            {"ks", required_argument, 0, 'k'},
            {"cs", required_argument, 0, 'c'},
            {"output", optional_argument, 0, 'o'},
            {"impl-num", optional_argument, 0, 'p'},
            {"renumber", optional_argument, 0, 'e'},
            {"niters-renum", optional_argument, 0, 'n'},
            {"cuda-dev-id", optional_argument, 0, 'd'},
            {"rank", optional_argument, 0, 'r'},
            {"tk", optional_argument, 0, 't'},
            {"tb", optional_argument, 0, 'l'},
            {"balanced", optional_argument, 0, 'a'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = 0;
        c = getopt_long(argc, argv, "i:b:k:c:o:p:e:n:d:r:t:l:a:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            strcpy(ifname, optarg);
            break;
        case 'o':
            fo = fopen(optarg, "w");
            ptiAssert(fo != NULL);
            break;
        case 'b':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sb_bits);
            break;
        case 'k':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sk_bits);
            break;
        case 'c':
            sscanf(optarg, "%"HIPARTI_SCN_ELEMENT_INDEX, &sc_bits);
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
            sscanf(optarg, "%"HIPARTI_SCN_INDEX, &R);
            break;
        case 't':
            sscanf(optarg, "%d", &tk);
            break;
        case 'l':
            sscanf(optarg, "%d", &tb);
            break;
        case 'a':
            sscanf(optarg, "%d", &balanced);
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
    printf("balanced: %d\n", balanced);

    /* A sorting included in load tensor */
    ptiAssert(ptiLoadSparseTensor(&tsr, 1, ifname) == 0);
    ptiSparseTensorStatus(&tsr, stdout);
    // ptiAssert(ptiDumpSparseTensor(&tsr, 0, stdout) == 0);

    /* Renumber the input tensor */
    ptiIndex ** map_inds;
    if (renumber > 0) {
        map_inds = (ptiIndex **)malloc(tsr.nmodes * sizeof *map_inds);
        pti_CheckOSError(!map_inds, "MTTKRP HiCOO");
        for(ptiIndex m = 0; m < tsr.nmodes; ++m) {
            map_inds[m] = (ptiIndex *)malloc(tsr.ndims[m] * sizeof (ptiIndex));
            pti_CheckError(!map_inds[m], "MTTKRP HiCOO", NULL);
            for(ptiIndex i = 0; i < tsr.ndims[m]; ++i)
                map_inds[m][i] = i;
        }

        ptiTimer renumber_timer;
        ptiNewTimer(&renumber_timer, 0);
        ptiStartTimer(renumber_timer);

        if ( renumber == 1 || renumber == 2) { /* Set the Lexi-order or BFS-like renumbering */
            #if 0
            orderit(&tsr, map_inds, renumber, niters_renum);
            #else
            ptiIndexRenumber(&tsr, map_inds, renumber, niters_renum, sb_bits, tk, impl_num);
            #endif

        }
        if ( renumber == 3) { /* Set randomly renumbering */
            ptiGetRandomShuffledIndices(&tsr, map_inds);
        }
        fflush(stdout);

        ptiStopTimer(renumber_timer);
        ptiPrintElapsedTime(renumber_timer, "Renumbering");
        ptiFreeTimer(renumber_timer);

        ptiTimer shuffle_timer;
        ptiNewTimer(&shuffle_timer, 0);
        ptiStartTimer(shuffle_timer);

        ptiSparseTensorShuffleIndices(&tsr, map_inds);

        ptiStopTimer(shuffle_timer);
        ptiPrintElapsedTime(shuffle_timer, "Shuffling time");
        ptiFreeTimer(shuffle_timer);
        printf("\n");


        // ptiSparseTensorSortIndex(&tsr, 1);   // debug purpose only
        // FILE * debug_fp = fopen("new.txt", "w");
        // ptiAssert(ptiDumpSparseTensor(&tsr, 0, debug_fp) == 0);
        // fprintf(stdout, "\nmap_inds:\n");
        // for(ptiIndex m = 0; m < tsr.nmodes; ++m) {
        //     ptiDumpIndexArray(map_inds[m], tsr.ndims[m], stdout);
        // }
        // fclose(debug_fp);
    }

    /* Convert to HiCOO tensor */
    ptiNnzIndex max_nnzb = 0;
    ptiTimer convert_timer;
    ptiNewTimer(&convert_timer, 0);
    ptiStartTimer(convert_timer);

    ptiAssert(ptiSparseTensorToHiCOO(&hitsr, &max_nnzb, &tsr, sb_bits, sk_bits, sc_bits, tk) == 0);

    ptiStopTimer(convert_timer);
    ptiPrintElapsedTime(convert_timer, "Convert HiCOO");
    ptiFreeTimer(convert_timer);

    ptiSparseTensorStatusHiCOO(&hitsr, stdout);
    // ptiAssert(ptiDumpSparseTensorHiCOO(&hitsr, stdout) == 0);

    ptiIndex nmodes = hitsr.nmodes;
    ptiNewRankKruskalTensor(&ktensor, nmodes, tsr.ndims, R);
    ptiFreeSparseTensor(&tsr);

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {
        tk = 1;
        ptiAssert(ptiCpdAlsHiCOO(&hitsr, R, niters, tol, &ktensor) == 0);
    } else if(cuda_dev_id == -1) {
        omp_set_num_threads(tk);
        #pragma omp parallel
        {
            tk = omp_get_num_threads();
        }
        printf("tk: %d, tb: %d\n", tk, tb);
        ptiAssert(ptiOmpCpdAlsHiCOO(&hitsr, R, niters, tol, tk, tb, balanced, &ktensor) == 0);
    }

    // for(int it=0; it<nloops; ++it) {
    // }

    if(fo != NULL) {
        // Dump ktensor to files
        if (renumber > 0) {
            ptiRankKruskalTensorInverseShuffleIndices(&ktensor, map_inds);
        }
        ptiAssert( ptiDumpRankKruskalTensor(&ktensor, fo) == 0 );
        fclose(fo);
    }

    if (renumber > 0) {
        for(ptiIndex m = 0; m < tsr.nmodes; ++m) {
            free(map_inds[m]);
        }
        free(map_inds);
    }
    ptiFreeSparseTensorHiCOO(&hitsr);
    ptiFreeRankKruskalTensor(&ktensor);

    return 0;
}
