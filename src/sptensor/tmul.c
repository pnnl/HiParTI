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

#include <ParTI.h>
#include <stdlib.h>
#include "sptensor.h"
#include <string.h>
#include <limits.h>
#include <numa.h>


void summation(sptIndex nmodes_Z, sptSparseTensor *Z_tmp, sptSparseTensor *Z_input, int tk, sptTimer timer, double* summation_time, sptIndex * ndims_buf, unsigned long long Z_total_size, unsigned long long * Z_tmp_start, sptSparseTensor *Z, int opt_summation);

/** All combined:
 * 0: COOY + SPA
 * 1: COOY + HTA
 * 2: HTY + SPA
 * 3: HTY + HTA
 **/
int sptSparseTensorMulTensor(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, sptIndex * modes_Z, int tk, int output_sorting, int opt_summation, int placement)
{
    // Experiment modes
    int experiment_modes = 3;
    // sscanf(getenv("EXPERIMENT_MODES"), "%d", &experiment_modes);

    if(X->nnz == 0 || Y->nnz == 0) {
        // printf("No contraction needed.\n");
        sptIndex nmodes_Z = X->nmodes + Y->nmodes - 2 * num_cmodes;
        // sptDumpIndexArray(cmodes_X, num_cmodes, stdout);
        // sptDumpIndexArray(cmodes_Y, num_cmodes, stdout);
        // sptDumpIndexArray(modes_Z, nmodes_Z, stdout);
        sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
        spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
        for(sptIndex m = 0; m < X->nmodes - num_cmodes; ++m) {
            // ndims_buf[m] = X->ndims[m];
            ndims_buf[m] = X->ndims[modes_Z[m]];
        }
        for(sptIndex m = X->nmodes - num_cmodes; m < nmodes_Z; ++m) {
            // ndims_buf[(m - num_cmodes) + X->nmodes - num_cmodes] = Y->ndims[m];
            ndims_buf[m] = Y->ndims[modes_Z[m]];
        }
        // printf("ndims_buf:\n");
        // sptDumpIndexArray(ndims_buf, nmodes_Z, stdout); fflush(stdout);
        int result = sptNewSparseTensor(Z, nmodes_Z, ndims_buf); 
        free(ndims_buf);
        return 0;
    }

//0: COOY + SPA
if(experiment_modes == 0) {
    int result;
    /// The number of threads
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    /// Shuffle tensor indices according to mode_order_X
    sptSparseTensorShuffleModes(X, mode_order_X);
    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    sptSparseTensorSortIndex(X, 1, tk);
    
    sptStopTimer(timer);
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    /// Shuffle Y indices and sort Y as the order of free modes -> contract modes
    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_Y);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }
    sptAssert(ci == num_cmodes);
    /// Shuffle tensor indices according to mode_order_Y
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    // printf("Permuted Y:\n");
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order
    sptSparseTensorSortIndex(Y, 1, tk);
    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);  
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time );

    //printf("Sorted X:\n");
    //sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    //printf("Sorted Y:\n");
    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);

    /// Set fidx_X: indexing the combined free indices and fidx_Y: indexing the combined contract indices
    sptNnzIndexVector fidx_X, fidx_Y;
    //sptStartTimer(timer);
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    /// Set indices for contract modes, use Y
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);
    //sptStopTimer(timer);
    //sptPrintElapsedTime(timer, "Set fidx X,Y");
    //sptPrintElapsedTime(timer, "Set fidx X");
    //printf("fidx_X: \n");
    //sptDumpNnzIndexVector(&fidx_X, stdout);
    //printf("fidx_Y: \n");
    //sptDumpNnzIndexVector(&fidx_Y, stdout);
    free(mode_order_X);
    free(mode_order_Y);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[m];
    }
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[m];
    }   
    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = malloc(tk * sizeof (sptSparseTensor));

    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;

    sptNewTimer(&timer_SPA, 0);

    sptStartTimer(timer);

    // For the progress
    int fx_counter = fidx_X.len;

#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, fidx_Y, nmodes_X, nmodes_Y, num_cmodes, Z_tmp, fx_counter)   
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
        int tid = omp_get_thread_num();
        //Print the progress 
        fx_counter--;
        //if (fx_counter % 1 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        if (tid == 0){
            sptStartTimer(timer_SPA);
        }
        /// Allocate the SPA buffer
        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
        sptIndexVector * spa_inds = (sptIndexVector*)malloc(nmodes_spa * sizeof(sptIndexVector));
        sptValueVector spa_vals;
        for(sptIndex m = 0; m < nmodes_spa; ++m)
            sptNewIndexVector(&spa_inds[m], 0, 0);
        sptNewValueVector(&spa_vals, 0, 0);

        /// Allocate a small index buffer
        sptIndexVector inds_buf;
        sptNewIndexVector(&inds_buf, (nmodes_Y - num_cmodes), (nmodes_Y - num_cmodes));
        //printf("\nzX: [%lu, %lu]\n", fx_begin, fx_end);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_prep += sptElapsedTime(timer_SPA);
            sptStartTimer(timer_SPA);
        }

        /// zX has common free indices
        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
            if (tid == 0) {
                sptStartTimer(timer_SPA); 
            }
            sptValue valX = X->values.data[zX];
            sptIndexVector cmode_index_X; 
            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
            for(sptIndex i = 0; i < num_cmodes; ++i){
                 cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
                 //printf("\ncmode_index_X[%lu]: %lu", i, cmode_index_X[i]);
             }

            sptNnzIndex fy_begin = -1;
            sptNnzIndex fy_end = -1;
            unsigned int current_idx = 0;
            
            for(sptIndex j = 0; j < fidx_Y.len; j++){
                for(sptIndex i = 0; i< num_cmodes; i++){
                    if(cmode_index_X.data[i] != Y->inds[i].data[fidx_Y.data[j]]) break;
                    if(i == (num_cmodes - 1)){
                        fy_begin = fidx_Y.data[j];
                        fy_end = fidx_Y.data[j+1];
                        break;
                    }
                    //printf("\ni: %lu, current_idx: %lu, Y->inds[i].data[fidx_Y.data[current_idx]]: %lu\n", i, current_idx, Y->inds[i].data[fidx_Y.data[current_idx]]);
                }
                if (fy_begin != -1 || fy_end != -1) break;
            }
            
            
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_free_mode += sptElapsedTime(timer_SPA);
            }
        
            if (fy_begin == -1 || fy_end == -1) continue;
            //printf("zX: %lu, valX: %.2f, cmode_index_X[0]: %u, zY: [%lu, %lu]\n", zX, valX, cmode_index_X.data[0], fy_begin, fy_end);

            if (tid == 0){
                sptStartTimer(timer_SPA);               
            }

            /// zY has common contraction indices
            char tmp[32];
            char index_str[128]; 
            long int tmp_key;
            for(sptNnzIndex zY = fy_begin; zY < fy_end; ++ zY) {    // Loop nnzs inside a Y fiber
                for(sptIndex m = 0; m < nmodes_spa; ++m)
                    inds_buf.data[m] = Y->inds[m + num_cmodes].data[zY];
                //printf("inds_buf:\n");
                //sptDumpIndexVector(&inds_buf, stdout);
                long int found = sptInIndexVector(spa_inds, nmodes_spa, spa_inds[0].len, &inds_buf);
                if( found == -1) {
                    for(sptIndex m = 0; m < nmodes_spa; ++m)
                        sptAppendIndexVector(&spa_inds[m], Y->inds[m + num_cmodes].data[zY]);
                    sptAppendValueVector(&spa_vals, Y->values.data[zY] * valX);
                } else {
                    spa_vals.data[found] += Y->values.data[zY] * valX;
                }
            }   // End Loop nnzs inside a Y fiber
            //printf("spa_inds:\n");
            //for(sptIndex m = 0; m < nmodes_spa; ++m) {
            //    printf("[m%u]:\n", m);
            //    sptDumpIndexVector(&spa_inds[m], stdout);
            //}
            //printf("spa_vals:\n");
            //sptDumpValueVector(&spa_vals, stdout);
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_spa += sptElapsedTime(timer_SPA);
            }

        }   // End Loop nnzs inside a X fiber
    
        if (tid == 0){
            sptStartTimer(timer_SPA);   
        }

        /// Write back to Z
        Z_tmp[tid].nnz += spa_vals.len;
     
        for(sptIndex i = 0; i < spa_vals.len; ++i) {
            for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
                sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
            }
        }
        for(sptIndex m = 0; m < nmodes_spa; ++m) 
            sptAppendIndexVectorWithVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], &spa_inds[m]);
        sptAppendValueVectorWithVector(&Z_tmp[tid].values, &spa_vals);  
             
        //printf("Z:\n");
        //sptDumpSparseTensor(&Z_tmp[tid], 0, stdout);
        /// Free SPA buffer
        for(sptIndex m = 0; m < nmodes_spa; ++m){
            sptFreeIndexVector(&(spa_inds[m]));   
         }     
         sptFreeValueVector(&spa_vals);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_accumulate_z += sptElapsedTime(timer_SPA);
        }
    }   // End Loop fiber pointers of X

sptStopTimer(timer);
double main_computation = sptElapsedTime(timer);
total_time += main_computation;
double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

    sptStartTimer(timer);

    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
    }
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

#pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
    for(int i = 0; i < tk; i++){
        int tid = omp_get_thread_num();
        if(Z_tmp[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m) 
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);        
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);  
        }
    } 

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Writeback");
    sptStartTimer(timer);


    // summation
    sptStartTimer(timer);
    double summation_time = 0;
    sptSparseTensor *Z_input;
    summation(nmodes_Z, Z_tmp, Z_input, tk, timer, &summation_time, ndims_buf, Z_total_size, Z_tmp_start, Z, opt_summation);  // s5
    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Summation");
    sptStartTimer(timer);

    sptSparseTensorSortIndex(Z, 1, tk);

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Output Sorting");
    printf("[Total time]: %.6f s\n", total_time);
    printf("\n");
} 

//1: COOY + HTA
if(experiment_modes == 1){
    int result;
    /// The number of threads
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    /// Shuffle tensor indices according to mode_order_X
    sptSparseTensorShuffleModes(X, mode_order_X);
    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    sptSparseTensorSortIndex(X, 1, tk);
    
    sptStopTimer(timer);
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    /// Shuffle Y indices and sort Y as the order of free modes -> contract modes
    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_Y);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }
    sptAssert(ci == num_cmodes);
    /// Shuffle tensor indices according to mode_order_Y
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    // printf("Permuted Y:\n");
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order
    sptSparseTensorSortIndex(Y, 1, tk);
    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    printf("[Input Processing]: %.6f s\n", X_time + sptElapsedTime(timer));

    //printf("Sorted X:\n");
    //sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    //printf("Sorted Y:\n");
    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);

    /// Set fidx_X: indexing the combined free indices and fidx_Y: indexing the combined contract indices
    sptNnzIndexVector fidx_X, fidx_Y;
    //sptStartTimer(timer);
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    /// Set indices for contract modes, use Y
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);
    //sptStopTimer(timer);
    //sptPrintElapsedTime(timer, "Set fidx X,Y");
    //sptPrintElapsedTime(timer, "Set fidx X");
    //printf("fidx_X: \n");
    //sptDumpNnzIndexVector(&fidx_X, stdout);
    //printf("fidx_Y: \n");
    //sptDumpNnzIndexVector(&fidx_Y, stdout);
    free(mode_order_X);
    free(mode_order_Y);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[m];
    }
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[m];
    }   
    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = malloc(tk * sizeof (sptSparseTensor));

    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;

    sptNewTimer(&timer_SPA, 0);

    sptStartTimer(timer);

    // For the progress
    int fx_counter = fidx_X.len;

#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, fidx_Y, nmodes_X, nmodes_Y, num_cmodes, Z_tmp, fx_counter)   
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
        int tid = omp_get_thread_num();
        //Print the progress 
        fx_counter--;
        //if (fx_counter % 1 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
        if (tid == 0){
            sptStartTimer(timer_SPA);
        }

        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
        long int nnz_counter = 0;
        
        /// Calculate key range for hashtable
        sptIndex* inds_buf = (sptIndex*)malloc((nmodes_spa + 1) * sizeof(sptIndex));
        sptIndex current_idx = 0;
        for(sptIndex i = 0; i < nmodes_spa + 1; i++) inds_buf[i] = 1;
        for(sptIndex i = 0; i < nmodes_spa;i++){
            for(sptIndex j = i; j < nmodes_spa;j++)
                inds_buf[i] = inds_buf[i] * Y->ndims[j + num_cmodes];
        }

        /// Create a hashtable for SPAs
        table_t *ht;
        const unsigned int ht_size = 10000;
        ht = htCreate(ht_size);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_prep += sptElapsedTime(timer_SPA);
        }

        /// zX has common free indices
        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }

            sptValue valX = X->values.data[zX];
            sptIndexVector cmode_index_X; 
            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
            for(sptIndex i = 0; i < num_cmodes; ++i){
                 cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
                 //printf("\ncmode_index_X[%lu]: %lu", i, cmode_index_X[i]);
             }

            sptNnzIndex fy_begin = -1;
            sptNnzIndex fy_end = -1;
            
            for(sptIndex j = 0; j < fidx_Y.len; j++){
                for(sptIndex i = 0; i< num_cmodes; i++){
                    if(cmode_index_X.data[i] != Y->inds[i].data[fidx_Y.data[j]]) break;
                    if(i == (num_cmodes - 1)){
                        fy_begin = fidx_Y.data[j];
                        fy_end = fidx_Y.data[j+1];
                        break;
                    }
                    //printf("\ni: %lu, current_idx: %lu, Y->inds[i].data[fidx_Y.data[current_idx]]: %lu\n", i, current_idx, Y->inds[i].data[fidx_Y.data[current_idx]]);
                }
                if (fy_begin != -1 || fy_end != -1) break;
            }
            
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_free_mode += sptElapsedTime(timer_SPA);
            }
        
            if (fy_begin == -1 || fy_end == -1) continue;
            //printf("zX: %lu, valX: %.2f, cmode_index_X[0]: %u, zY: [%lu, %lu]\n", zX, valX, cmode_index_X.data[0], fy_begin, fy_end);

            if (tid == 0) sptStartTimer(timer_SPA);               

            /// zY has common contraction indices
            for(sptNnzIndex zY = fy_begin; zY < fy_end; ++ zY) {    // Loop nnzs inside a Y fiber
                long int tmp_key = 0;    
                for(sptIndex m = 0; m < nmodes_spa; ++m)
                    tmp_key += Y->inds[m + num_cmodes].data[zY] * inds_buf[m + 1];
                sptValue val = htGet(ht, tmp_key);
                if(val == LONG_MIN) 
                    htInsert(ht, tmp_key, Y->values.data[zY] * valX);
                else    
                    htUpdate(ht, tmp_key, val + (Y->values.data[zY] * valX));
                //printf("val: %f\n", val);
            }
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_spa += sptElapsedTime(timer_SPA);
            }

        }   // End Loop nnzs inside a X fiber
    
        if (tid == 0){
            sptStartTimer(timer_SPA);   
        }

        /// Write back to Z
        for(int i = 0; i < ht->size; i++){
            node_t *temp = ht->list[i];
            while(temp){
                long int idx_tmp = temp->key;
                nnz_counter++;
                for(sptIndex m = 0; m < nmodes_spa; ++m) {
                    //printf("idx_tmp: %lu, m: %d, (idx_tmp inds_buf[m])/inds_buf[m+1]): %d\n", idx_tmp, m, (idx_tmp%inds_buf[m])/inds_buf[m+1]);
                    sptAppendIndexVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%inds_buf[m])/inds_buf[m+1]);
                }
                //printf("val: %f\n", temp->val);
                sptAppendValueVector(&Z_tmp[tid].values, temp->val);
                node_t* pre = temp;
                temp = temp->next;
                free(pre);
            }
        }

        Z_tmp[tid].nnz += nnz_counter;
        for(sptIndex i = 0; i < nnz_counter; ++i) {
            for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
                sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
            }
        }

        // release spa hashtable
        htFree(ht);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_accumulate_z += sptElapsedTime(timer_SPA);
        }
    }   // End Loop fiber pointers of X

sptStopTimer(timer);
double main_computation = sptElapsedTime(timer);
total_time += main_computation;
double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
printf("[Index Search]: %.2f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
printf("[Accumulation]: %.2f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

sptStartTimer(timer);

/// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
    }
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

#pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
    for(int i = 0; i < tk; i++){
        int tid = omp_get_thread_num();
        if(Z_tmp[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m) 
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);        
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);  
        }
    } 

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Writeback");
    sptStartTimer(timer);

    sptSparseTensorSortIndex(Z, 1, tk);

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Output Sorting");
    printf("[Total time]: %.6f s\n", total_time);
    printf("\n");
}

//2: HTY + SPA
if(experiment_modes == 2) {
    int result;
    /// The number of threads
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    /// Shuffle tensor indices according to mode_order_X
    sptSparseTensorShuffleModes(X, mode_order_X);

    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
    sptSparseTensorSortIndex(X, 1, tk);
    
    sptStopTimer(timer);
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    /// Convert Y into a hashtable
    /// Create a hashtable 
    table_t *Y_ht;
    unsigned int Y_ht_size = Y->nnz;
    Y_ht = tensor_htCreate(Y_ht_size);    
    
    // omp lock
    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

    /// Calculate key range for Y hashtable
    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    sptNnzIndex Y_nnz = Y->nnz;
#pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
        /// Contract modes of Y
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];    

        /// Free modes of Y
        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < Y_num_fmodes; ++m)
            key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
        unsigned pos = tensor_htHashCode(key_cmodes);
        omp_set_lock(&locks[pos]);    
        tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(Y_val.len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    // Release omp lock
    for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);

    sptStopTimer(timer);   
    total_time += sptElapsedTime(timer);  
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time );
    
    /// Set fidx_X: indexing the combined free indices
    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    //printf("fidx_X: \n");
    //sptDumpNnzIndexVector(&fidx_X, stdout);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[m];
    }

    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }

    free(mode_order_X);
    free(mode_order_Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    sptStartTimer(timer);

    // For the progress
    int fx_counter = fidx_X.len;

#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, fx_counter)       
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
        int tid = omp_get_thread_num();
        //Print the progress 
        fx_counter--;
        //if (fx_counter % 100 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        if (tid == 0){
            sptStartTimer(timer_SPA);
        }
        /// Allocate the SPA buffer
        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
        sptIndexVector * spa_inds = (sptIndexVector*)malloc(nmodes_spa * sizeof(sptIndexVector));
        sptValueVector spa_vals;
        for(sptIndex m = 0; m < nmodes_spa; ++m)
            sptNewIndexVector(&spa_inds[m], 0, 0);
        sptNewValueVector(&spa_vals, 0, 0);

        /// Allocate a small index buffer
        sptIndexVector inds_buf;
        sptNewIndexVector(&inds_buf, (nmodes_Y - num_cmodes), (nmodes_Y - num_cmodes));
        //printf("\nzX: [%lu, %lu]\n", fx_begin, fx_end);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_prep += sptElapsedTime(timer_SPA);
        }

        /// zX has common free indices
        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
            if (tid == 0) {
                sptStartTimer(timer_SPA);  
            }             
            sptValue valX = X->values.data[zX];
            sptIndexVector cmode_index_X; 
            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
            for(sptIndex i = 0; i < num_cmodes; ++i){
                cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
                //printf("\ncmode_index_X[%lu]: %lu\n", i, cmode_index_X.data[i]);
            }

            unsigned long long key_cmodes = 0;    
            for(sptIndex m = 0; m < num_cmodes; ++m)
                key_cmodes += cmode_index_X.data[m] * Y_cmode_inds[m + 1];
            //printf("key_cmodes: %d\n", key_cmodes);    

            tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);  
            //printf("Y_val.len: %d\n", Y_val.len);
            unsigned int my_len = Y_val.len;
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_free_mode += sptElapsedTime(timer_SPA);
            }
            if(my_len == 0) continue;

            if (tid == 0) sptStartTimer(timer_SPA);               

            for(int i = 0; i < my_len; i++){
                unsigned long long fmode =  Y_val.key_FM[i];
                float result = Y_val.val[i] * valX;

                for(sptIndex m = 0; m < nmodes_spa; ++m)
                    inds_buf.data[m] =  (fmode%Y_fmode_inds[m])/Y_fmode_inds[m+1];
                //printf("inds_buf:\n");
                //sptDumpIndexVector(&inds_buf, stdout);
                long int found = sptInIndexVector(spa_inds, nmodes_spa, spa_inds[0].len, &inds_buf);
                if( found == -1) {
                    for(sptIndex m = 0; m < nmodes_spa; ++m)
                        sptAppendIndexVector(&spa_inds[m], (fmode%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                    sptAppendValueVector(&spa_vals, result);
                } else {
                    spa_vals.data[found] += result;
                }
            }

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_spa += sptElapsedTime(timer_SPA);
            }
            
        }   // End Loop nnzs inside a X fiber

        if (tid == 0) sptStartTimer(timer_SPA);    

        /// Write back to Z
        Z_tmp[tid].nnz += spa_vals.len;
     
        for(sptIndex i = 0; i < spa_vals.len; ++i) {
            for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
                sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
            }
        }
        for(sptIndex m = 0; m < nmodes_spa; ++m) 
            sptAppendIndexVectorWithVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], &spa_inds[m]);
        sptAppendValueVectorWithVector(&Z_tmp[tid].values, &spa_vals);  
             
        //printf("Z:\n");
        //sptDumpSparseTensor(&Z_tmp[tid], 0, stdout);
        /// Free SPA buffer
        for(sptIndex m = 0; m < nmodes_spa; ++m){
            sptFreeIndexVector(&(spa_inds[m]));   
         }     
         sptFreeValueVector(&spa_vals);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_accumulate_z += sptElapsedTime(timer_SPA);
        }
    }

sptStopTimer(timer);
double main_computation = sptElapsedTime(timer);
total_time += main_computation;
double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

sptStartTimer(timer);
/// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

#pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
    for(int i = 0; i < tk; i++){
        int tid = omp_get_thread_num();
        if(Z_tmp[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m) 
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);        
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);  
            //sptDumpSparseTensor(&Z_tmp[tid], 0, stdout);
        }
    } 

    //  for(int i = 0; i < tk; i++)
    //      sptFreeSparseTensor(&Z_tmp[i]);
    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Writeback");
    sptStartTimer(timer);

    sptSparseTensorSortIndex(Z, 1, tk);

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Output Sorting");
    printf("[Total time]: %.6f s\n", total_time);
    printf("\n");

    //sptFreeTimer(timer);
    //sptFreeNnzIndexVector(&fidx_X);

    return 0;
}  

    //3: HTY + HTA
    if(experiment_modes == 3)
    {
        int result;
        sptIndex nmodes_X = X->nmodes;
        sptIndex nmodes_Y = Y->nmodes;
        sptTimer timer;
        double total_time = 0;
        sptNewTimer(&timer, 0);

        if(num_cmodes > X->nmodes) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
        for(sptIndex m = 0; m < num_cmodes; ++m) {
            if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
                spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
            }
        }

        sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;

        sptStartTimer(timer);
        /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
        sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
        sptIndex ci = nmodes_X - num_cmodes, fi = 0;
        // for(sptIndex m = 0; m < nmodes_X; ++m) {
        //     if(sptInArray(cmodes_X, num_cmodes, m) == -1) { // free modes
        //         mode_order_X[fi] = m;
        //         ++ fi;
        //     }
        // }
        // sptAssert(fi == nmodes_X - num_cmodes);
        for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
            mode_order_X[m] = modes_Z[m];
        }
        /// Copy the contract modes while keeping the contraction mode order
        for(sptIndex m = 0; m < num_cmodes; ++m) {
            mode_order_X[ci] = cmodes_X[m];
            ++ ci;
        }
        sptAssert(ci == nmodes_X);
        // printf("mode_order_X:\n");
        // sptDumpIndexArray(mode_order_X, nmodes_X, stdout); fflush(stdout);
        /// Shuffle tensor indices according to mode_order_X
        sptSparseTensorShuffleModes(X, mode_order_X);

        // printf("Permuted X:\n");
        // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
        for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
        // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
        sptSparseTensorSortIndex(X, 1, tk);
        // printf("Sorted X:\n");
        // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
        
        sptStopTimer(timer);
        //total_time += sptPrintElapsedTime(timer, "Sort X");
        double X_time = sptElapsedTime(timer);
        total_time += X_time;
        sptStartTimer(timer);

        //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
        sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
        ci = 0;
        fi = num_cmodes;
        // for(sptIndex m = 0; m < nmodes_Y; ++m) {
        //     if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
        //         mode_order_Y[fi] = m;
        //         ++ fi;
        //     }
        // }
        for(sptIndex m = nmodes_X - num_cmodes; m < nmodes_Z; ++m) {
            mode_order_Y[fi] = modes_Z[m];
            ++ fi;
        }
        sptAssert(fi == nmodes_Y);
        /// Copy the contract modes while keeping the contraction mode order
        for(sptIndex m = 0; m < num_cmodes; ++m) {
            mode_order_Y[ci] = cmodes_Y[m];
            ++ ci;
        }
        // printf("mode_order_Y:\n");
        // sptDumpIndexArray(mode_order_Y, nmodes_Y, stdout); fflush(stdout);

        table_t *Y_ht;
        unsigned int Y_ht_size = Y->nnz;
        Y_ht = tensor_htCreate(Y_ht_size);    
        
        omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
        for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

        sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
        for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
        for(sptIndex i = 0; i < num_cmodes;i++){
            for(sptIndex j = i; j < num_cmodes;j++)
                Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
        }

        sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
        sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
        for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
        for(sptIndex i = 0; i < Y_num_fmodes;i++){
            for(sptIndex j = i; j < Y_num_fmodes;j++)
                Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
        }

        sptNnzIndex Y_nnz = Y->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
        for(sptNnzIndex i = 0; i < Y_nnz; i++){
            //if (Y->values.data[i] <0.00000001) continue;
            unsigned long long key_cmodes = 0;    
            for(sptIndex m = 0; m < num_cmodes; ++m)
                key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];    

            unsigned long long key_fmodes = 0;    
            for(sptIndex m = 0; m < Y_num_fmodes; ++m)
                key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
            unsigned pos = tensor_htHashCode(key_cmodes);
            omp_set_lock(&locks[pos]);    
            tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
            //printf("Y_val.len: %d\n", Y_val.len); 
            if(Y_val.len == 0) {
                tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            }
            else  {
                tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
                //for(int i = 0; i < Y_val.len; i++)
                //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
            }
            omp_unset_lock(&locks[pos]);    
            //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
        }

        for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);

        sptStopTimer(timer);     
        total_time += sptElapsedTime(timer);
        // printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

        sptNnzIndexVector fidx_X;
        /// Set indices for free modes, use X
        sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
        //printf("fidx_X: \n");
        //sptDumpNnzIndexVector(&fidx_X, stdout);

        /// Allocate the output tensor
        sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
        spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
        for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
            ndims_buf[m] = X->ndims[m];
        }
        /// For non-sorted Y 
        for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
            ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
        }

        free(mode_order_X);
        free(mode_order_Y);

        /// Each thread with a local Z_tmp
        sptSparseTensor *Z_tmp = malloc(tk * sizeof (sptSparseTensor));
        for (int i = 0; i < tk; i++){
            result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
        }

        //free(ndims_buf);
        spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
        
        sptTimer timer_SPA;
        double time_prep = 0;
        double time_free_mode = 0;
        double time_spa = 0;
        double time_accumulate_z = 0;
        sptNewTimer(&timer_SPA, 0);
        sptStartTimer(timer);

        // For the progress
        int fx_counter = fidx_X.len;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds)       
        for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
            int tid = omp_get_thread_num();
            fx_counter--;
            //if (fx_counter % 100 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
            sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];

            /// hashtable size
            const unsigned int ht_size = 10000;
            sptIndex nmodes_spa = nmodes_Y - num_cmodes;
            long int nnz_counter = 0;
            sptIndex current_idx = 0;

            table_t *ht;
            ht = htCreate(ht_size);

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_prep += sptElapsedTime(timer_SPA);
            }

            for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {   
                sptValue valX = X->values.data[zX];  
                if (tid == 0) {
                    sptStartTimer(timer_SPA);
                }       
                sptIndexVector cmode_index_X; 
                sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
                for(sptIndex i = 0; i < num_cmodes; ++i){
                    cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
                    //printf("\ncmode_index_X[%lu]: %lu\n", i, cmode_index_X.data[i]);
                }

                unsigned long long key_cmodes = 0;    
                for(sptIndex m = 0; m < num_cmodes; ++m)
                    key_cmodes += cmode_index_X.data[m] * Y_cmode_inds[m + 1];  

                tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);  
                //printf("Y_val.len: %d\n", Y_val.len);
                unsigned int my_len = Y_val.len;
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_free_mode += sptElapsedTime(timer_SPA);
                }
                if(my_len == 0) continue;

                if (tid == 0) {
                    sptStartTimer(timer_SPA);       
                }        

                for(int i = 0; i < my_len; i++){
                    unsigned long long fmode =  Y_val.key_FM[i];
                    //printf("i: %d, Y_val.key_FM[i]: %lu, Y_val.val[i]: %f\n", i, Y_val.key_FM[i], Y_val.val[i]);
                    sptValue spa_val = htGet(ht, fmode);
                    float result = Y_val.val[i] * valX;
                    if(spa_val == LONG_MIN) {
                        htInsert(ht, fmode, result);
                        nnz_counter++;
                    }
                    else    
                        htUpdate(ht, fmode, spa_val + result);
                }

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_spa += sptElapsedTime(timer_SPA);
                }
                
            }   // End Loop nnzs inside a X fiber

            if (tid == 0) {
                sptStartTimer(timer_SPA);    
            }

            for(int i = 0; i < ht->size; i++){
                node_t *temp = ht->list[i];
                while(temp){
                    unsigned long long idx_tmp = temp->key;
                    //nnz_counter++;
                    for(sptIndex m = 0; m < nmodes_spa; ++m) {
                        //printf("idx_tmp: %lu, m: %d, (idx_tmp inds_buf[m])/inds_buf[m+1]): %d\n", idx_tmp, m, (idx_tmp%inds_buf[m])/inds_buf[m+1]);                   
                        sptAppendIndexVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                    }
                    //printf("val: %f\n", temp->val);
                    sptAppendValueVector(&Z_tmp[tid].values, temp->val);
                    node_t* pre = temp;
                    temp = temp->next;
                    free(pre);
                }
            }
            Z_tmp[tid].nnz += nnz_counter;
            for(sptIndex i = 0; i < nnz_counter; ++i) {
                for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {               
                    sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
                }
            }
            htFree(ht);
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_accumulate_z += sptElapsedTime(timer_SPA);
            }
        }

    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    // printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    // printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
        //Calculate the indecies of Z
        unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
        unsigned long long Z_total_size = 0;

        Z_tmp_start[0] = 0;
        for(int i = 0; i < tk; i++){
            Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
            Z_total_size +=  Z_tmp[i].nnz;
            //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
        }
        //printf("%d\n", Z_total_size);
        result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

    #pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
        for(int i = 0; i < tk; i++){
            int tid = omp_get_thread_num();
            if(Z_tmp[tid].nnz > 0){
                for(sptIndex m = 0; m < nmodes_Z; ++m) 
                    sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);        
                sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);  
                //sptDumpSparseTensor(&Z_tmp[tid], 0, stdout);
            }
        } 

        //  for(int i = 0; i < tk; i++)
        //      sptFreeSparseTensor(&Z_tmp[i]);
        sptStopTimer(timer);
        // total_time += sptPrintElapsedTime(timer, "Writeback");
        total_time += sptElapsedTime(timer);

        sptStartTimer(timer);
        if(output_sorting == 1){
            sptSparseTensorSortIndex(Z, 1, tk);
        }
        sptStopTimer(timer);
        // total_time += sptPrintElapsedTime(timer, "Output Sorting");
        total_time += sptElapsedTime(timer);
        // printf("[Total time]: %.6f s\n", total_time);
        printf("\n");
    }  

    return 0;
}


/* Two Tensor Contractions */
void buildHtY(int tk,  table_t * Y_ht, sptIndex Y_num_fmodes, sptIndex *mode_order_Y, sptIndex num_cmodes, sptIndex * Y_cmode_inds, sptIndex * Y_fmode_inds, sptNnzIndex Y_nnz, sptSparseTensor * Y, omp_lock_t *locks)
{
   #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
        //if (Y->values.data[i] <0.00000001) continue;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < Y_num_fmodes; ++m)
            key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
        unsigned pos = tensor_htHashCode(key_cmodes);
        omp_set_lock(&locks[pos]);    
        tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(Y_val.len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }
}

void computeTC(int tk, sptNnzIndexVector fidx_X, sptIndex nmodes_X, sptIndex nmodes_Y, sptIndex num_cmodes, sptSparseTensor * Z_tmp, sptIndex * Y_fmode_inds, table_t * Y_ht, sptIndex * Y_cmode_inds, int fx_counter, sptTimer timer_SPA, sptSparseTensor* X, double* time_prep, double* time_index_search, double* time_spa, double* time_accumulate_z, sptIndex nmodes_Z)
{
#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds)       
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) 
    {    // Loop fiber pointers of X
        int tid = omp_get_thread_num();
        fx_counter--;
        //if (fx_counter % 100 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
        if (tid == 0){
            sptStartTimer(timer_SPA);
        }
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];

        /// hashtable size
        const unsigned int ht_size = 10000;
        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
        long int nnz_counter = 0;
        sptIndex current_idx = 0;

        table_t *ht;
        ht = htCreate(ht_size);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            *time_prep += sptElapsedTime(timer_SPA);
        }

        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {   
            sptValue valX = X->values.data[zX];  
            if (tid == 0) {
                sptStartTimer(timer_SPA);
            }       
            sptIndexVector cmode_index_X; 
            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
            for(sptIndex i = 0; i < num_cmodes; ++i){
                cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
                //printf("\ncmode_index_X[%lu]: %lu\n", i, cmode_index_X.data[i]);
            }

            unsigned long long key_cmodes = 0;    
            for(sptIndex m = 0; m < num_cmodes; ++m)
                key_cmodes += cmode_index_X.data[m] * Y_cmode_inds[m + 1];  

            tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);  
            //printf("Y_val.len: %d\n", Y_val.len);
            unsigned int my_len = Y_val.len;
            if (tid == 0){
                sptStopTimer(timer_SPA);
                *time_index_search += sptElapsedTime(timer_SPA);
            }
            if(my_len == 0) continue;

            if (tid == 0) {
                sptStartTimer(timer_SPA);       
            }        

            for(int i = 0; i < my_len; i++){
                unsigned long long fmode =  Y_val.key_FM[i];
                //printf("i: %d, Y_val.key_FM[i]: %lu, Y_val.val[i]: %f\n", i, Y_val.key_FM[i], Y_val.val[i]);
                sptValue spa_val = htGet(ht, fmode);
                float result = Y_val.val[i] * valX;
                if(spa_val == LONG_MIN) {
                    htInsert(ht, fmode, result);
                    nnz_counter++;
                }
                else    
                    htUpdate(ht, fmode, spa_val + result);
            }

            if (tid == 0){
                sptStopTimer(timer_SPA);
                *time_spa += sptElapsedTime(timer_SPA);
            }
            
        }   // End Loop nnzs inside a X fiber

        if (tid == 0) {
            sptStartTimer(timer_SPA);    
        }

        for(int i = 0; i < ht->size; i++){
            node_t *temp = ht->list[i];
            while(temp){
                unsigned long long idx_tmp = temp->key;
                //nnz_counter++;
                for(sptIndex m = 0; m < nmodes_spa; ++m) {
                    //printf("idx_tmp: %lu, m: %d, (idx_tmp inds_buf[m])/inds_buf[m+1]): %d\n", idx_tmp, m, (idx_tmp%inds_buf[m])/inds_buf[m+1]);                   
                    sptAppendIndexVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                }
                //printf("val: %f\n", temp->val);
                sptAppendValueVector(&Z_tmp[tid].values, temp->val);
                node_t* pre = temp;
                temp = temp->next;
                free(pre);
            }
        }
        Z_tmp[tid].nnz += nnz_counter;
        for(sptIndex i = 0; i < nnz_counter; ++i) {
            for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {               
                sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
            }
        }
        htFree(ht);
        if (tid == 0){
            sptStopTimer(timer_SPA);
            *time_accumulate_z += sptElapsedTime(timer_SPA);
        }
    }
}


void writeback (sptSparseTensor * Z_tmp, sptSparseTensor * Z, sptIndex nmodes_Z, unsigned long long *Z_tmp_start, int tk)
{
#pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
    for(int i = 0; i < tk; i++)
    {
        int tid = omp_get_thread_num();
        if(Z_tmp[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m) 
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);        
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);  
            //sptDumpSparseTensor(&Z_tmp[tid], 0, stdout);
        }
    } 
}



// Summation
void summation_original(int tk, sptIndex nmodes_Z, sptIndex * ndims_buf, unsigned long long Z_total_size, sptSparseTensor *Z_tmp, sptSparseTensor * Z_input, sptSparseTensor * Z)
{
    printf("Z_input: \n");
    sptDumpSparseTensor(Z_input, 0, stdout);
    long int Z_counter = Z_input->nnz;
#pragma omp parallel for schedule(static) num_threads(tk) shared(Z, Z_counter, Z_input)
    for(sptNnzIndex i = 0; i < Z_input->nnz; i++){
        Z_counter--;
        if (Z_counter % 10000 == 0) {
            printf("Progress: %ld\/%lu\n", Z_counter, Z_input->nnz); 
            fflush(stdout);
        }
        sptIndexVector inds_buf;
        sptNewIndexVector(&inds_buf, nmodes_Z, nmodes_Z);
        for(sptIndex m = 0; m < nmodes_Z; ++m)
            inds_buf.data[m] = Z_input->inds[m].data[i];
        long int found = sptInIndexVector(Z->inds, nmodes_Z, Z->nnz, &inds_buf);    
        if(found == -1) 
        {
            #pragma omp critical 
            {
                for(sptIndex m = 0; m < nmodes_Z; ++m)
                    sptAppendIndexVector(&Z->inds[m], inds_buf.data[m]);
                sptAppendValueVector(&Z->values, Z_input->values.data[i]);
            }
        } else {
            Z->values.data[found] += Z_input->values.data[i];
        }
    }
}


void writebackResults(sptSparseTensor * Z_tmp, sptIndex nmodes_Z, unsigned long long *Z_tmp_start, sptSparseTensor * Z_input_backup, sptSparseTensor * Z_backup, sptSparseTensor * Z_input, int tk)
{
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Z_tmp, nmodes_Z, Z_tmp_start)
    for (int i = 0; i < tk; i++)
    {
        int tid = omp_get_thread_num();
        if (Z_tmp[tid].nnz > 0)
        {
            for (sptIndex m = 0; m < nmodes_Z; ++m)
                sptAppendIndexVectorWithVectorStartFromNuma(&Z_input_backup->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);
            sptAppendValueVectorWithVectorStartFromNuma(&Z_input_backup->values, &Z_tmp[tid].values, Z_tmp_start[tid]);
        }
    }   

    #pragma omp parallel for schedule(static) num_threads(tk) shared(Z_tmp, nmodes_Z, Z_tmp_start)
    for (int i = 0; i < tk; i++)
    {
        int tid = omp_get_thread_num();
        if (Z_tmp[tid].nnz > 0)
        {
            for (sptIndex m = 0; m < nmodes_Z; ++m)
                sptAppendIndexVectorWithVectorStartFromNuma(&Z_backup->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);
            sptAppendValueVectorWithVectorStartFromNuma(&Z_backup->values, &Z_tmp[tid].values, Z_tmp_start[tid]);
        }
    }   

    #pragma omp parallel for schedule(static) num_threads(tk) shared(Z_tmp, nmodes_Z, Z_tmp_start)
    for (int i = 0; i < tk; i++)
    {
        int tid = omp_get_thread_num();
        if (Z_tmp[tid].nnz > 0)
        {
            for (sptIndex m = 0; m < nmodes_Z; ++m)
                sptAppendIndexVectorWithVectorStartFromNuma(&Z_input->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);
            sptAppendValueVectorWithVectorStartFromNuma(&Z_input->values, &Z_tmp[tid].values, Z_tmp_start[tid]);
        }
    } 
}


void summation(sptIndex nmodes_Z, sptSparseTensor *Z_tmp, sptSparseTensor *Z_input, int tk, sptTimer timer, double* summation_time, sptIndex * ndims_buf, unsigned long long Z_total_size, unsigned long long * Z_tmp_start, sptSparseTensor *Z, int opt_summation)
{
    sptSparseTensor *Z_input_backup = malloc(sizeof (sptSparseTensor));
    sptSparseTensor *Z_backup = malloc(sizeof (sptSparseTensor));
    int result = sptNewSparseTensorWithSize(Z_input_backup, nmodes_Z, ndims_buf, Z_total_size);
    result = sptNewSparseTensorWithSize(Z_backup, nmodes_Z, ndims_buf, Z_total_size);

    Z_input = malloc(sizeof(sptSparseTensor));
    result = sptNewSparseTensorWithSize(Z_input, nmodes_Z, ndims_buf, Z_total_size);
    
    writebackResults (Z_tmp, nmodes_Z, Z_tmp_start, Z_input_backup, Z_backup, Z_input, tk);

    #pragma omp parallel for schedule(static) num_threads(tk)
    for(int i = 0; i < tk; i++)
        sptFreeSparseTensor(&Z_tmp[i]);   
    
    if(opt_summation == 2) 
        summation_original(tk, nmodes_Z, ndims_buf, Z_total_size, Z_tmp, Z_input, Z);
    else if (opt_summation == 1)
    {
        sptStartTimer(timer);    
        unsigned long long Z_nnz = 0;
        sptIndex *Z_mode_inds = (sptIndex *)malloc((nmodes_Z + 1) * sizeof(sptIndex));
        for (sptIndex i = 0; i < nmodes_Z + 1; i++)
            Z_mode_inds[i] = 1;
        for (sptIndex i = 0; i < nmodes_Z; i++)
        {
            for (sptIndex j = i; j < nmodes_Z; j++)
                Z_mode_inds[i] = Z_mode_inds[i] * Z_tmp[0].ndims[j];
        }

        table_t *Z_ht;
        for (int i = 0; i < tk; i++)
        {
            Z_nnz += Z_tmp[i].nnz;
        }    

        unsigned long long  Z_ht_size = Z_nnz / 40;
        // Z_nnz = 0;
        // for (int i = 0; i < tk; i++)
        // {
        //     Z_nnz += Z_tmp_dram[i].nnz + Z_tmp_optane[i].nnz;
        // }

        omp_lock_t *locks_Z = (omp_lock_t *)malloc(Z_ht_size * sizeof(omp_lock_t));
        for (size_t i = 0; i < Z_ht_size; i++)
        {
            omp_init_lock(&locks_Z[i]);
        }
        sptStopTimer(timer);
        *summation_time += sptElapsedTime(timer);

        sptStartTimer(timer);
        bucket_s *buckets_s = (bucket_s *)malloc(Z_ht_size * sizeof(bucket_s));
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Z_ht_size, buckets_s)
        for (int i = 0; i < Z_ht_size; i++)
        {
            buckets_s[i].frequency = 0;
            buckets_s[i].pos = i;
            sptNewIndexVector(&buckets_s[i].idx, 1, 1);
        }
        sptStopTimer(timer);
        *summation_time += sptElapsedTime(timer);

        sptStartTimer(timer);
        unsigned long long Z_input_nnz = Z_input->nnz;


    #pragma omp parallel for schedule(static, tk) num_threads(tk) shared(Z_input, Z_input_nnz, Z_mode_inds, buckets_s, locks_Z)
        for (int i = 0; i < Z_input_nnz; i++)
        {
            unsigned long long key_modes = 0;
            for (sptIndex m = 0; m < nmodes_Z; ++m)
                key_modes += Z_input->inds[m].data[i] * Z_mode_inds[m + 1];
            unsigned bucket = key_modes%Z_ht_size;
            
            omp_set_lock(&locks_Z[bucket]);
            buckets_s[bucket].frequency++;
            sptAppendIndexVector(&buckets_s[bucket].idx, i);
            omp_unset_lock(&locks_Z[bucket]);
        }

        sptQuickSortBucketS(buckets_s, 0, Z_ht_size, tk);
        sptStopTimer(timer);
        *summation_time += sptElapsedTime(timer);

        sptStartTimer(timer);
        Z_ht = htCreate(Z_ht_size);
        sptStopTimer(timer);
        *summation_time += sptElapsedTime(timer);


        sptStartTimer(timer); 
        unsigned long long tmp_counter;
        tmp_counter = Z_nnz;
        #pragma omp parallel for schedule(dynamic, 1) num_threads(tk) shared(Z_mode_inds, Z_ht, locks_Z, Z_input_nnz, buckets_s)
        for(sptNnzIndex i = 0; i < Z_nnz; i++) {
            tmp_counter--;
            unsigned long long key_modes = 0;  
            for(sptIndex m = 0; m < nmodes_Z; ++m){
                key_modes += Z_input->inds[m].data[i] * Z_mode_inds[m + 1];    
            }

            unsigned int bucket = buckets_s[htHashCode(key_modes)].pos;
            omp_set_lock(&locks_Z[bucket]);    
            sptValue Z_val = htGet(Z_ht, key_modes);
            if(Z_val == LONG_MIN) {
                htInsertS(Z_ht, key_modes, Z_input->values.data[i]);
            }
            else    
                htUpdateS(Z_ht, key_modes, Z_val + Z_input->values.data[i]);
            omp_unset_lock(&locks_Z[bucket]);  
        }

        sptStopTimer(timer);
        *summation_time += sptElapsedTime(timer);

        tmp_counter = Z_input_nnz;
        #pragma omp parallel for schedule(dynamic, 1) num_threads(tk) shared(Z_mode_inds, Z_ht, locks_Z, Z_input_nnz, buckets_s)
        for(sptNnzIndex i = 0; i < Z_input_nnz; i++) 
        {
            tmp_counter--;
            unsigned long long key_modes = 0;  
            for(sptIndex m = 0; m < nmodes_Z; ++m){
                key_modes += Z_input->inds[m].data[i] * Z_mode_inds[m + 1];    
            }

            unsigned int bucket = buckets_s[htHashCode(key_modes)].pos;
            omp_set_lock(&locks_Z[bucket]);    
            sptValue Z_val = htGet(Z_ht, key_modes);
            if(Z_val == LONG_MIN) {
                htInsertS(Z_ht, key_modes, Z_input->values.data[i]);
            }
            else    
                htUpdateS(Z_ht, key_modes, Z_val + Z_input->values.data[i]);
            omp_unset_lock(&locks_Z[bucket]);  
        }

        for (size_t i = 0; i < Z_ht_size; i++)
        {
            omp_destroy_lock(&locks_Z[i]);
        }

        sptStopTimer(timer);
        *summation_time += sptElapsedTime(timer);
    }
}


//opt_summation: 0: no summation; 1: ours; 2: linear search summation
int sptSparseTensorMulTensor2TCs(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, sptIndex * modes_Z,
    sptSparseTensor *Z2, sptSparseTensor * const X2, sptSparseTensor *const Y2, sptIndex num_cmodes_2, sptIndex * cmodes_X2, sptIndex * cmodes_Y2, sptIndex * modes_Z2,
    int tk, int output_sorting, int opt_summation, int placement)
{
    if((X->nnz == 0 || Y->nnz == 0) && (X2->nnz == 0 || Y2->nnz == 0)) {
        // printf("No contraction needed.\n");
        // sptDumpIndexArray(cmodes_X, num_cmodes, stdout);
        // sptDumpIndexArray(cmodes_Y, num_cmodes, stdout);
        // sptDumpIndexArray(modes_Z, nmodes_Z, stdout);
        sptIndex nmodes_Z = X->nmodes + Y->nmodes - 2 * num_cmodes;
        sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
        spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
        for(sptIndex m = 0; m < X->nmodes - num_cmodes; ++m) {
            ndims_buf[m] = X->ndims[modes_Z[m]];
        }
        // for(sptIndex m = num_cmodes; m < Y->nmodes; ++m) {
        //     ndims_buf[(m - num_cmodes) + X->nmodes - num_cmodes] = Y->ndims[m];
        // }   
        for(sptIndex m = X->nmodes - num_cmodes; m < nmodes_Z; ++m) {
            // ndims_buf[(m - num_cmodes) + X->nmodes - num_cmodes] = Y->ndims[m];
            ndims_buf[m] = Y->ndims[modes_Z[m]];
        }
        // printf("ndims_buf:\n");
        // sptDumpIndexArray(ndims_buf, nmodes_Z, stdout); fflush(stdout);
        int result = sptNewSparseTensor(Z, nmodes_Z, ndims_buf); 
        free(ndims_buf);

        sptIndex nmodes_Z2 = X2->nmodes + Y2->nmodes - 2 * num_cmodes_2;
        sptIndex *ndims_buf_2 = malloc(nmodes_Z2 * sizeof *ndims_buf_2);
        spt_CheckOSError(!ndims_buf_2, "CPU  SpTns * SpTns");
        for(sptIndex m = 0; m < X2->nmodes - num_cmodes_2; ++m) {
            ndims_buf_2[m] = X2->ndims[modes_Z2[m]];
        }
        for(sptIndex m = X2->nmodes - num_cmodes_2; m < nmodes_Z2; ++m) {
            // ndims_buf[(m - num_cmodes) + X->nmodes - num_cmodes] = Y->ndims[m];
            ndims_buf_2[m] = Y2->ndims[modes_Z2[m]];
        }
        result = sptNewSparseTensor(Z2, nmodes_Z2, ndims_buf_2); 
        free(ndims_buf_2);
        return 0;
    } else if(X->nnz == 0 || Y->nnz == 0) {
        printf("Only 2nd contraction performed.\n");
        sptIndex nmodes_Z2 = X2->nmodes + Y2->nmodes - 2 * num_cmodes_2;
        sptIndex *ndims_buf_2 = malloc(nmodes_Z2 * sizeof *ndims_buf_2);
        spt_CheckOSError(!ndims_buf_2, "CPU  SpTns * SpTns");
        for(sptIndex m = 0; m < X2->nmodes - num_cmodes_2; ++m) {
            ndims_buf_2[m] = X2->ndims[modes_Z2[m]];
        }
        for(sptIndex m = X2->nmodes - num_cmodes_2; m < nmodes_Z2; ++m) {
            // ndims_buf[(m - num_cmodes) + X->nmodes - num_cmodes] = Y->ndims[m];
            ndims_buf_2[m] = Y2->ndims[modes_Z2[m]];
        }
        int result = sptNewSparseTensor(Z2, nmodes_Z2, ndims_buf_2); 
        free(ndims_buf_2);

        sptSparseTensorMulTensor(Z2, X2, Y2, num_cmodes_2, cmodes_X2, cmodes_Y2, modes_Z2, tk, output_sorting, opt_summation, placement);
        return 0;
    } else if(X2->nnz == 0 || Y2->nnz == 0) {
        printf("Only 1st contraction performed.\n");
        sptIndex nmodes_Z = X->nmodes + Y->nmodes - 2 * num_cmodes;
        sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
        spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
        for(sptIndex m = 0; m < X->nmodes - num_cmodes; ++m) {
            ndims_buf[m] = X->ndims[modes_Z[m]];
        }
        // for(sptIndex m = num_cmodes; m < Y->nmodes; ++m) {
        //     ndims_buf[(m - num_cmodes) + X->nmodes - num_cmodes] = Y->ndims[m];
        // }   
        for(sptIndex m = X->nmodes - num_cmodes; m < nmodes_Z; ++m) {
            // ndims_buf[(m - num_cmodes) + X->nmodes - num_cmodes] = Y->ndims[m];
            ndims_buf[m] = Y->ndims[modes_Z[m]];
        }
        // printf("ndims_buf:\n");
        // sptDumpIndexArray(ndims_buf, nmodes_Z, stdout); fflush(stdout);
        int result = sptNewSparseTensor(Z, nmodes_Z, ndims_buf); 
        free(ndims_buf);

        sptSparseTensorMulTensor(Z, X, Y, num_cmodes, cmodes_X, cmodes_Y, modes_Z, tk, output_sorting, opt_summation, placement);
        return 0;
    }

    //3: HTY + HTA
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptIndex nmodes_X2 = X2->nmodes;
    sptIndex nmodes_Y2 = Y2->nmodes;
    sptTimer timer, stage1, stage2;
    double total_time = 0;
    sptNewTimer(&timer, 0);
    sptNewTimer(&stage1, 0);
    sptNewTimer(&stage2, 0);

    if( (num_cmodes >= X->nmodes || num_cmodes >= Y->nmodes) || (num_cmodes_2 >= X2->nmodes || num_cmodes_2 >= Y2->nmodes)) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }
    for(sptIndex m = 0; m < num_cmodes_2; ++m) {
        if(X2->ndims[cmodes_X2[m]] != Y2->ndims[cmodes_Y2[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        mode_order_X[m] = modes_Z[m];
    }
    sptIndex * mode_order_X2 = (sptIndex *)malloc(nmodes_X2 * sizeof(sptIndex));
    sptIndex ci2 = nmodes_X2 - num_cmodes_2, fi2 = 0;
    for(sptIndex m = 0; m < nmodes_X2 - num_cmodes_2; ++m) {
        mode_order_X2[m] = modes_Z2[m];
    }
    
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    for(sptIndex m = 0; m < num_cmodes_2; ++m) {
        mode_order_X2[ci2] = cmodes_X2[m];
        ++ ci2;
    }
    sptAssert(ci2 == nmodes_X2);
    // printf("mode_order_X:\n");
    // sptDumpIndexArray(mode_order_X, nmodes_X, stdout);
    // printf("mode_order_X2:\n");
    // sptDumpIndexArray(mode_order_X2, nmodes_X2, stdout);

    /// Shuffle tensor indices according to mode_order_X
    /****** Start: TC1 & TC2-stage 1 **********/
    sptSparseTensorShuffleModes(X, mode_order_X);   // tc1-s1
    sptSparseTensorShuffleModes(X2, mode_order_X2);   // tc2-s1

    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    for(sptIndex m = 0; m < nmodes_X2; ++m) mode_order_X2[m] = m; // reset mode_order
    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
    sptSparseTensorSortIndex(X, 1, tk);   // tc1-s1
    // printf("Sorted X:\n");
    // sptSparseTensorStatus(X, stdout);

    sptStopTimer(timer);
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    sptIndex nmodes_Z = X->nmodes + Y->nmodes - 2 * num_cmodes;
    sptIndex nmodes_Z2 = X2->nmodes + Y2->nmodes - 2 * num_cmodes_2;

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    sptIndex * mode_order_Y2 = (sptIndex *)malloc(nmodes_Y2 * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    ci2 = 0;
    fi2 = num_cmodes_2;
    for(sptIndex m = nmodes_X - num_cmodes; m < nmodes_Z; ++m) {
        mode_order_Y[fi] = modes_Z[m];
        ++ fi;
    }
    sptAssert(fi == nmodes_Y);
    for(sptIndex m = nmodes_X2 - num_cmodes_2; m < nmodes_Z2; ++m) {
        mode_order_Y2[fi2] = modes_Z2[m];
        ++ fi2;
    }
    sptAssert(fi2 == nmodes_Y2);
    
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }
    sptAssert(ci == num_cmodes);
    for(sptIndex m = 0; m < num_cmodes_2; ++m) {
        mode_order_Y2[ci2] = cmodes_Y2[m];
        ++ ci2;
    }
    sptAssert(ci2 == num_cmodes_2);
    // printf("mode_order_Y: \n");
    // sptDumpIndexArray(mode_order_Y, nmodes_Y, stdout);
    // printf("mode_order_Y2: \n");
    // sptDumpIndexArray(mode_order_Y2, nmodes_Y2, stdout);
    
    // tc1-s1
    table_t *Y_ht, *Y2_ht;
    unsigned int Y_ht_size = Y->nnz;
    unsigned int Y2_ht_size = Y2->nnz;
    Y_ht = tensor_htCreate(Y_ht_size);
    Y2_ht = tensor_htCreate(Y2_ht_size);
    
    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);
    omp_lock_t *locks_2 = (omp_lock_t *)malloc(Y2_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y2_ht_size; i++) omp_init_lock(&locks_2[i]);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }
    sptIndex* Y2_cmode_inds = (sptIndex*)malloc((num_cmodes_2 + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes_2 + 1; i++) Y2_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes_2; i++){
        for(sptIndex j = i; j < num_cmodes_2; j++)
            Y2_cmode_inds[i] = Y2_cmode_inds[i] * Y2->ndims[mode_order_Y2[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }
    sptIndex Y2_num_fmodes = nmodes_Y2 - num_cmodes_2;
    sptIndex* Y2_fmode_inds = (sptIndex*)malloc((Y2_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y2_num_fmodes + 1; i++) Y2_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y2_num_fmodes;i++){
        for(sptIndex j = i; j < Y2_num_fmodes;j++)
            Y2_fmode_inds[i] = Y2_fmode_inds[i] * Y2->ndims[mode_order_Y2[j + num_cmodes_2]]; 
    }

    sptNnzIndex Y_nnz = Y->nnz, Y2_nnz = Y2->nnz;

    buildHtY(tk, Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds, Y_nnz, Y, locks);    // tc1-s1
    /****** End: TC1-stage 1 **********/

    for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);

    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    // printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    sptNnzIndexVector fidx_X, fidx_X2;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    // printf("fidx_X: \n");
    // sptDumpNnzIndexVector(&fidx_X, stdout);

    /// Allocate the output tensor
    sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[m];
    }
    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }
    // for(sptIndex m = 0; m < X->nmodes - num_cmodes; ++m) {
    //     ndims_buf[m] = X->ndims[modes_Z[m]];
    // }
    // /// For non-sorted Y 
    // for(sptIndex m = X->nmodes - num_cmodes; m < nmodes_Z; ++m) {
    //     // ndims_buf[(m - num_cmodes) + X->nmodes - num_cmodes] = Y->ndims[m];
    //     ndims_buf[m] = Y->ndims[modes_Z[m]];
    // }
    // printf("ndims_buf: \n");
    // sptDumpIndexArray(ndims_buf, nmodes_Z, stdout);

    sptIndex *ndims_buf_2 = malloc(nmodes_Z2 * sizeof *ndims_buf_2);
    spt_CheckOSError(!ndims_buf_2, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X2 - num_cmodes_2; ++m) {
        ndims_buf_2[m] = X2->ndims[m];
    }
    /// For non-sorted Y 
    for(sptIndex m = num_cmodes_2; m < nmodes_Y2; ++m) {
        ndims_buf_2[(m - num_cmodes_2) + nmodes_X2 - num_cmodes_2] = Y2->ndims[mode_order_Y2[m]];
    }
    // for(sptIndex m = 0; m < X2->nmodes - num_cmodes_2; ++m) {
    //     ndims_buf_2[m] = X2->ndims[modes_Z2[m]];
    // }
    // for(sptIndex m = X2->nmodes - num_cmodes_2; m < nmodes_Z2; ++m) {
    //     // ndims_buf[(m - num_cmodes) + X->nmodes - num_cmodes] = Y->ndims[m];
    //     ndims_buf_2[m] = Y2->ndims[modes_Z2[m]];
    // }
    // printf("ndims_buf_2: \n");
    // sptDumpIndexArray(ndims_buf_2, nmodes_Z2, stdout);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }
    sptSparseTensor *Z2_tmp = malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z2_tmp[i]), nmodes_Z2, ndims_buf_2);
    }
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA, timer_SPA_2;
    double time_prep = 0, time_prep_2 = 0;
    double time_index_search = 0, time_index_search_2 = 0;
    double time_spa = 0, time_spa_2 = 0;
    double time_accumulate_z = 0, time_accumulate_z2 = 0;
    sptNewTimer(&timer_SPA, 0);
    sptNewTimer(&timer_SPA_2, 0);
    sptStartTimer(timer);

    // For the progress
    int fx_counter = fidx_X.len;

    // For stage parallelism
    int tk_other = 8;
    int new_tk = tk+tk_other;
    omp_set_num_threads(new_tk);
    omp_set_nested(1);
    #pragma omp parallel sections num_threads(2) shared(Z_tmp, Z2_tmp)
    {
        #pragma omp section
        {
            sptStartTimer(stage1);
            // TC1: comp
            /****** Start: TC1-stage 2 **********/
            computeTC(tk, fidx_X, nmodes_X, nmodes_Y, num_cmodes, Z_tmp, Y_fmode_inds, Y_ht, Y_cmode_inds, fx_counter, timer_SPA, X, &time_prep, &time_index_search, &time_spa, &time_accumulate_z, nmodes_Z);
            /****** End: TC1-stage 2 **********/
            sptStopTimer(stage1);
        }

        #pragma omp section
        {
            // s1,2,3 TC2
            sptStartTimer(stage2);
            sptSparseTensorSortIndex(X2, 1, tk_other);   // tc2-s1
            // printf("Sorted X2:\n");
            // sptSparseTensorStatus(X2, stdout);

            buildHtY(tk_other, Y2_ht, Y2_num_fmodes, mode_order_Y2, num_cmodes_2, Y2_cmode_inds, Y2_fmode_inds, Y2_nnz, Y2, locks_2);    // tc2-s1
            for(size_t i = 0; i < Y2_ht_size; i++) omp_destroy_lock(&locks_2[i]);

            sptSparseTensorSetIndices(X2, mode_order_X2, nmodes_X2 - num_cmodes_2, &fidx_X2);
            // printf("fidx_X2: \n");
            // sptDumpNnzIndexVector(&fidx_X2, stdout);
            int fx_counter_2 = fidx_X2.len;
            // printf("X2: \n");
            // sptAssert(sptDumpSparseTensor(X2, 0, stdout) == 0);
            // printf("Y2: \n");
            // sptAssert(sptDumpSparseTensor(Y2, 0, stdout) == 0);
            
            /****** End: TC2-stage 1 **********/
            /****** Start: TC2-stage 2 **********/
            computeTC(tk, fidx_X2, nmodes_X2, nmodes_Y2, num_cmodes_2, Z2_tmp, Y2_fmode_inds, Y2_ht, Y2_cmode_inds, fx_counter_2, timer_SPA_2, X2, &time_prep_2, &time_index_search_2, &time_spa_2, &time_accumulate_z2,nmodes_Z2); 
            /****** End: TC2-stage 2 **********/
            sptStopTimer(stage2);
        }
    }
    omp_set_num_threads(tk);

    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_index_search + time_spa + time_accumulate_z;
    // printf("[Index Search]: %.6f s\n", (time_index_search + time_prep)/spa_total * main_computation);
    // printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    // Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long* Z2_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0, Z2_total_size = 0;
    sptSparseTensor *Z_input, *Z2_input;

    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    sptAssert(Z_tmp_start[tk] == Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

    // printf("Z2_tmp:\n");
    // for (int i = 0; i < tk; i++) {
    //     printf("%d: %lu\n", i, Z2_tmp[i].nnz);
    //     // result = sptDumpSparseTensor(&(Z2_tmp[i]), 0, stdout);
    // }
    Z2_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z2_tmp_start[i + 1] = Z2_tmp[i].nnz + Z2_tmp_start[i];
        Z2_total_size +=  Z2_tmp[i].nnz;
        // printf("Z2_tmp_start[i + 1]: %llu, i: %d\n", Z2_tmp_start[i + 1], i);
    }
    sptAssert(Z2_tmp_start[tk] == Z2_total_size);
    result = sptNewSparseTensorWithSize(Z2, nmodes_Z2, ndims_buf_2, Z2_total_size); 
    // printf("nmodes_Z2: %u\n", nmodes_Z2);

    writeback(Z_tmp, Z, nmodes_Z, Z_tmp_start, tk); // tc1
    // printf("Z: \n");
    // sptSparseTensorStatus(Z, stdout);
    sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");
    total_time += sptElapsedTime(timer);

    sptStartTimer(timer);
    double summation_time = 0;
    omp_set_num_threads(new_tk);
    omp_set_nested(1);
    
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            sptStartTimer(stage1);
            // TC1 summation
            /****** Start: TC1-stage 3 **********/
            summation(nmodes_Z, Z_tmp, Z_input, tk, timer, &summation_time, ndims_buf, Z_total_size, Z_tmp_start, Z, opt_summation);  // s5
            // printf("Summed Z: \n");
            // sptSparseTensorStatus(Z, stdout);
            /****** End: TC1-stage 3 **********/
            
            /****** Start: TC1-stage 4 **********/
            if(output_sorting == 1) sptSparseTensorSortIndex(Z, 1, tk); // s6
            /****** End: TC1-stage 4 **********/
            // printf("Sorted Z: \n");
            // sptDumpSparseTensor(Z, 0, stdout);
            sptStopTimer(stage1);
        }

        #pragma omp section
        {
            sptStartTimer(stage2);
            writeback(Z2_tmp, Z2, nmodes_Z2, Z2_tmp_start, tk); // tc2
            // printf("Z2: \n");
            // sptSparseTensorStatus(Z2, stdout);
            // printf("nmodes_Z2: %u\n", nmodes_Z2);
            // printf("Z2_tmp_start:\n");
            // for(int i=0; i< (tk + 1); ++i)
            //     printf("%llu, ", Z2_tmp_start[i]);

            /****** Start: TC2-stage 3 **********/
            summation(nmodes_Z2, Z2_tmp, Z2_input, tk, timer, &summation_time, ndims_buf_2, Z2_total_size, Z2_tmp_start, &Z2, opt_summation);  
            // printf("Summed Z: \n");
            // sptSparseTensorStatus(Z2, stdout);
            /****** End: TC2-stage 3 **********/

            /****** Start: TC2-stage 4 **********/
            if(output_sorting == 1) sptSparseTensorSortIndex(Z2, 1, tk);
            /****** End: TC2-stage 4 **********/
            // printf("\nZ2: \n");
            // sptDumpSparseTensor(Z2, 0, stdout);
            sptStopTimer(stage2);

        }
    }
    omp_set_num_threads(tk); 
    sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Summation+sorting");
    total_time += sptElapsedTime(timer);
    // printf("[Total time]: %.6f s\n", total_time);
    printf("\n");

    free(mode_order_X); free(mode_order_Y);
    free(mode_order_X2); free(mode_order_Y2);
    sptFreeTimer(timer);
    sptFreeTimer(stage1);
    sptFreeTimer(stage2);
    return 0;
}



