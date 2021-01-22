/*
    This file is part of HiParTI!.

    HiParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    HiParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with HiParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <HiParTI.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>


int ptiLoadSparseMatrix(ptiSparseMatrix *mtx, ptiIndex start_index, FILE *fid)
{
    MM_typecode matcode;
    int iores, retval;

    if (mm_read_banner(fid, &matcode) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (!mm_is_valid(matcode)){
        printf("Invalid Matrix Market file.\n");
        exit(1);
    }

    if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode) ) ){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        printf("Only sparse real-valued or pattern coordinate matrices are supported\n");
        exit(1);
    }

    int num_rows, num_cols, num_nonzeros;
    if ( mm_read_mtx_crd_size(fid,&num_rows,&num_cols,&num_nonzeros) !=0)
            exit(1);

    mtx->nrows     = (ptiIndex) num_rows;
    mtx->ncols     = (ptiIndex) num_cols;
    mtx->nnz = (ptiNnzIndex) num_nonzeros;

    retval = ptiNewIndexVector(&mtx->rowind, mtx->nnz, mtx->nnz);
    pti_CheckOSError(iores < 0, "SpMtx Load");
    retval = ptiNewIndexVector(&mtx->colind, mtx->nnz, mtx->nnz);
    pti_CheckOSError(iores < 0, "SpMtx Load");
    retval = ptiNewValueVector(&mtx->values, mtx->nnz, mtx->nnz);
    pti_CheckError(retval, "SpMtx Load", NULL);

    char strI[64], strJ[64], strV[64];
    if (mm_is_pattern(matcode)){
        // pattern matrix defines sparsity pattern, but not values
        for( ptiNnzIndex i = 0; i < mtx->nnz; i++ ) {
            fscanf(fid, "%s %s\n", strI, strJ);
            mtx->rowind.data[i] = (ptiIndex) (atoi(strI)) - 1;
            mtx->colind.data[i] = (ptiIndex) (atoi(strJ)) - 1;
            mtx->values.data[i] = 1.0;  //use value 1.0 for all nonzero entries 
        }
    } else if (mm_is_real(matcode) || mm_is_integer(matcode)){
        for( ptiNnzIndex i = 0; i < mtx->nnz; i++ ) {
            fscanf(fid, "%s %s %s\n", strI, strJ, strV);
            mtx->rowind.data[i] = (ptiIndex) (atoi(strI)) - 1;
            mtx->colind.data[i] = (ptiIndex) (atoi(strJ)) - 1;
            mtx->values.data[i] = (ptiValue) (atof(strV));
        }
    } else {
        printf("Unrecognized data type\n");
        exit(1);
    }

    if( mm_is_symmetric(matcode) ){ //duplicate off diagonal entries
        ptiIndex off_diagonals = 0;
        for( ptiNnzIndex i = 0; i < mtx->nnz; i++ ){
            if( mtx->rowind.data[i] != mtx->colind.data[i] )
                off_diagonals++;
        }

        ptiNnzIndex true_nonzeros = 2*off_diagonals + (mtx->nnz - off_diagonals);

        ptiIndex* new_I = (ptiIndex*)malloc(true_nonzeros * sizeof(ptiIndex));
        ptiIndex* new_J = (ptiIndex*)malloc(true_nonzeros * sizeof(ptiIndex));
        ptiValue * new_V = (ptiValue*)malloc(true_nonzeros * sizeof(ptiValue));

        ptiNnzIndex ptr = 0;
        for( ptiNnzIndex i = 0; i < mtx->nnz; i++ ){
            if( mtx->rowind.data[i] != mtx->colind.data[i] ){
                new_I[ptr] = mtx->rowind.data[i];  new_J[ptr] = mtx->colind.data[i];  new_V[ptr] = mtx->values.data[i];
                ptr++;
                new_J[ptr] = mtx->rowind.data[i];  new_I[ptr] = mtx->colind.data[i];  new_V[ptr] = mtx->values.data[i];
                ptr++;
            } else {
                new_I[ptr] = mtx->rowind.data[i];  new_J[ptr] = mtx->colind.data[i];  new_V[ptr] = mtx->values.data[i];
                ptr++;
            }
        }       
         free(mtx->rowind.data); free(mtx->colind.data); free(mtx->values.data);
         mtx->rowind.data = new_I;  mtx->colind.data = new_J; mtx->values.data = new_V;      
         mtx->nnz = true_nonzeros;
         mtx->rowind.len = true_nonzeros;
         mtx->rowind.cap = true_nonzeros;
         mtx->colind.len = true_nonzeros;
         mtx->colind.cap = true_nonzeros;
         mtx->values.len = true_nonzeros;
         mtx->values.cap = true_nonzeros;
    } //end symmetric case

    return 0;  

}

