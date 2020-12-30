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
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    char ifname[1000];
    FILE *fo = NULL;
    ptiIndex mode = 0;
    ptiSparseTensor X;
    ptiSemiSparseTensor Y;
    ptiSparseTensor Z;

    if(argc <= 3) { // #Required arguments
        print_usage(argv);
        exit(1);
    }

    int c;
    for(;;) {
        static struct option long_options[] = {
            {"input", required_argument, 0, 'i'},//input file
            {"mode", required_argument, 0, 'm'},//
            {"output", optional_argument, 0, 'o'},
            {"help", no_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "i:m:o:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'i':
            strcpy(ifname, optarg);
            printf("Input file: %s\n", ifname); fflush(stdout);
            break;
        case 'o':
            fo = fopen(optarg, "aw");
            ptiAssert(fo != NULL);
            printf("output file: %s\n", optarg); fflush(stdout);
            break;
        case 'm':
            sscanf(optarg, "%"HIPARTI_SCN_INDEX, &mode);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }

    ptiAssert(ptiLoadSparseTensor(&X, 1, ifname) == 0);

    ptiAssert(ptiSparseTensorToSemiSparseTensor(&Y, &X, mode) == 0);
    ptiFreeSparseTensor(&X);
    ptiAssert(ptiSemiSparseTensorToSparseTensor(&Z, &Y, 1e-6) == 0);
    ptiFreeSemiSparseTensor(&Y);

    if (fo != NULL) {
        ptiAssert(ptiDumpSparseTensor(&Z, 1, fo) == 0);
        fclose(fo);
    }

    ptiFreeSparseTensor(&Z);

    return 0;
}
