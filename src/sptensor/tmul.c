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

/** All combined:
 * 0: COOY + SPA
 * 1: COOY + HTA
 * 2: HTY + SPA
 * 3: HTY + HTA
 * 4: HTY + HTA on HM
 **/
int sptSparseTensorMulTensor(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, int tk, int output_sorting, int placement)
{
    // Experiment modes
    int experiment_modes;
    sscanf(getenv("EXPERIMENT_MODES"), "%d", &experiment_modes);

//0: COOY + SPA
if(experiment_modes == 0){
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
if(experiment_modes == 2){
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
if(experiment_modes == 3){
    int result;
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
    //total_time += sptPrintElapsedTime(timer, "Sort X");
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
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

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
    if(output_sorting == 1){
        sptSparseTensorSortIndex(Z, 1, tk);
    }
    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Output Sorting");
    printf("[Total time]: %.6f s\n", total_time);
    printf("\n");
}  

//4: HTY + HTA on HM
if(experiment_modes == 4){
    int result;
    int dram_node;
    int optane_node;
    sscanf(getenv("DRAM_NODE"), "%d", &dram_node);
    sscanf(getenv("OPTANE_NODE"), "%d", &optane_node);
    int numa_node = dram_node;

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

    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    sptSparseTensorSortIndex(X, 1, tk);
    
    sptStopTimer(timer);
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    unsigned long long tmp_dram_size = 0;
    FILE *fp;
    char *s;
    char path[1035];
    unsigned long long  i1, i2, i3, i4, i5, i6, i7, i8;
    fp = popen("numactl -H", "r");  
    while (fgets(path, sizeof(path), fp) != NULL) {  
        s = strstr(path, "node 0 free:");      
        if (s != NULL)                     
            if (2 == sscanf(s, "%*[^0123456789]%llu%*[^0123456789]%llu", &i1, &i2)){
                tmp_dram_size = i2 * 1024 * 1024; 
                //printf("test: %llu B\n", dram_cap);
                break;
            }
    }
    pclose(fp);

    unsigned int node_size = sizeof(unsigned long long) + sizeof(unsigned int) + sizeof(unsigned int) + sizeof(unsigned long long*) + sizeof(sptValue*) + sizeof(tensor_node_t*);
    unsigned long long Y_upper_size = node_size * (Y->nnz + Y->nnz);
    //printf("%lu\n", Y_upper_size);
    if (Y_upper_size < tmp_dram_size) numa_set_preferred(dram_node);
    else numa_set_preferred(numa_node);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { 
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_Y);

    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }
    sptAssert(ci == num_cmodes);
    //for(sptIndex m = 0; m < nmodes_Y; ++m) 
    //    printf ("mode_order_Y[m]: %d\n", mode_order_Y[m]);

    table_t *Y_ht;
    unsigned int Y_ht_size = Y->nnz;
    Y_ht = tensor_htCreate(Y_ht_size);    
    
    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) {
        omp_init_lock(&locks[i]);
    }

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }
    //for(sptIndex i = 0; i <= num_cmodes;i++)
    //    printf("%d ", Y_cmode_inds[i]);
    //printf("\n");

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    //sptIndex* Y_fmode_inds = (sptIndex*) numa_alloc_onnode((Y_num_fmodes + 1) * sizeof(sptIndex), numa_node);
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }
    //for(sptIndex i = 0; i <= Y_num_fmodes;i++)
    //    printf("%d ", Y_fmode_inds[i]);
    //printf("\n");

    sptNnzIndex Y_nnz = Y->nnz;
    unsigned int Y_free_upper = 0;
    
#pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
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
        unsigned int Y_len = Y_val.len;
        if(Y_len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            if (Y_len >= Y_free_upper) Y_free_upper = Y_len + 1;
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < Y_ht_size; i++) {
        omp_destroy_lock(&locks[i]);
    }  

    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);  
    printf("[Input Processing]: %.2f s\n", sptElapsedTime(timer) + X_time );


    sptStartTimer(timer);

    //printf("Sorted X:\n");
    //sptSparseTensorStatus(X, stdout);
    //sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    //printf("Sorted Y:\n");
    //sptSparseTensorStatus(Y, stdout);
    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);

    /// Set fidx_X: indexing the combined free indices;
    sptNnzIndexVector fidx_X;
    //sptStartTimer(timer);
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);

    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[m];
    }

    /// For sorted Y 
    //for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
    //    ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[m];
    //}
    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }
    free(mode_order_X);
    free(mode_order_Y);

    // sptSparseTensor *Z_tmp = malloc(tk * sizeof (sptSparseTensor));
    sptSparseTensor *Z_tmp_dram = numa_alloc_onnode(tk * sizeof (sptSparseTensor), dram_node);
    sptSparseTensor *Z_tmp_optane = numa_alloc_onnode(tk * sizeof (sptSparseTensor), optane_node);

    for (int i = 0; i < tk; i++){
        //result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
        result = sptNewSparseTensorNuma(&(Z_tmp_dram[i]), nmodes_Z, ndims_buf, dram_node);
        result = sptNewSparseTensorNuma(&(Z_tmp_optane[i]), nmodes_Z, ndims_buf, optane_node);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    unsigned long long dram_cur = 0;
    unsigned long long dram_cap = 0; 
    unsigned long long Z_mem = 0;

    fp = popen("numactl -H", "r");  // Open the command for reading 
    while (fgets(path, sizeof(path), fp) != NULL) {  // Read the output a line at a time - output it. 
        s = strstr(path, "node 0 free:");      // Search for string "hassasin" in buff
        if (s != NULL)                     // If successful then s now points at "hassasin"
            if (2 == sscanf(s, "%*[^0123456789]%llu%*[^0123456789]%llu", &i1, &i2)){
                //printf("System DRAM memory: %lu MB\n", i2);
                dram_cap = i2 * 1024 * 1024 / 1.1; // Should be changed into: memory of the current system - X - Y_ht
                //printf("test: %llu B\n", dram_cap);
                break;
            }
    }
    pclose(fp);

    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);

    // For the progress
    int fx_counter = fidx_X.len;

#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Z_tmp_dram, Z_tmp_optane, Y_fmode_inds, Y_ht, Y_cmode_inds, dram_cap, dram_cur, Z_mem, fx_counter) 
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
        int tid = omp_get_thread_num();
        fx_counter--;
        //if (fx_counter % 1000 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
        if (tid == 0){
            sptStartTimer(timer_SPA);
        }
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];      

        /// The total number and memory of SPA for one x fiber.
        unsigned long long num_SPA_upper = 0;
        unsigned long long mem_SPA_upper = 0;
        unsigned long long mem_SPA_cur = 0;
        bool SPA_in_dram = false;
        /// The total memory of Z_tmp
        unsigned long long Z_tmp_mem = 0;
        /// hashtable size
        const unsigned int ht_size = 10000;
        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
        long int nnz_counter = 0;
        sptIndex current_idx = 0;

        /*for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
            sptValue valX = X->values.data[zX];
            //printf("valX: %f\n", valX);
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
            if(my_len == 0) continue;
            num_SPA_upper += my_len;
        }*/

        mem_SPA_upper = (Y_free_upper + fx_end - fx_begin) * sizeof(node_t) + sizeof(node_t*) * ht_size + sizeof(table_t);
        if(mem_SPA_upper + dram_cur <= dram_cap) { // spa in dram
            dram_cur += mem_SPA_upper;
            SPA_in_dram = true;
        }

        table_t *ht;
        ht = htCreate(ht_size);
        mem_SPA_cur = sizeof( node_t*)*ht_size + sizeof( table_t);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_prep += sptElapsedTime(timer_SPA);
        }

        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
            if (tid == 0){
                sptStartTimer(timer_SPA);       
            }
            sptValue valX = X->values.data[zX];
            //printf("valX: %f\n", valX);
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

            if (tid == 0){
                sptStartTimer(timer_SPA);               
            }
            for(int i = 0; i < my_len; i++){
                unsigned long long fmode =  Y_val.key_FM[i];
                //printf("i: %d, Y_val.key_FM[i]: %lu, Y_val.val[i]: %f\n", i, Y_val.key_FM[i], Y_val.val[i]);
                sptValue spa_val = htGet(ht, fmode);
                float result = Y_val.val[i] * valX;
                if(spa_val == LONG_MIN) {
                    htInsert(ht, fmode, result);
                    mem_SPA_cur += sizeof(node_t);
                    nnz_counter++;
                }
                else    
                    htUpdate(ht, fmode, spa_val + result);
            }

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_spa += sptElapsedTime(timer_SPA);
            }
            
        }  

        if (tid == 0){
            sptStartTimer(timer_SPA);     
        }

        if(SPA_in_dram) dram_cur = dram_cur - mem_SPA_upper + mem_SPA_cur;
        Z_tmp_mem = nnz_counter * (nmodes_Z * sizeof(sptIndex) + sizeof(sptValue));
        Z_mem += Z_tmp_mem;

        if(Z_tmp_mem + dram_cur <= dram_cap && (tid % 7 != 0)){ 
            dram_cur += Z_tmp_mem;
            for(int i = 0; i < ht->size; i++){
                node_t *temp = ht->list[i];
                while(temp){
                    unsigned long long idx_tmp = temp->key;
                    //nnz_counter++;
                    for(sptIndex m = 0; m < nmodes_spa; ++m) {                        
                        //sptAppendIndexVector(&Z_tmp_dram[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                        sptAppendIndexVectorNuma(&Z_tmp_dram[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                    }
                    //printf("val: %f\n", temp->val);
                    //sptAppendValueVector(&Z_tmp_dram[tid].values, temp->val);
                    sptAppendValueVectorNuma(&Z_tmp_dram[tid].values, temp->val);
                    node_t* pre = temp;
                    temp = temp->next;
                    free(pre);
                    //numa_free(pre, sizeof(node_t));
                }
            }
            Z_tmp_dram[tid].nnz += nnz_counter;
            for(sptIndex i = 0; i < nnz_counter; ++i) {
                for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {               
                    //sptAppendIndexVector(&Z_tmp_dram[tid].inds[m], X->inds[m].data[fx_begin]);
                    sptAppendIndexVectorNuma(&Z_tmp_dram[tid].inds[m], X->inds[m].data[fx_begin]);
                }
            }
        }
        else{ // append elements to Z_tmp_optane in Optane
            for(int i = 0; i < ht->size; i++){
                node_t *temp = ht->list[i];
                while(temp){
                    unsigned long long idx_tmp = temp->key;
                    //nnz_counter++;
                    for(sptIndex m = 0; m < nmodes_spa; ++m) {                        
                        //sptAppendIndexVector(&Z_tmp_optane[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                        sptAppendIndexVectorNuma(&Z_tmp_optane[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                    }
                    //printf("val: %f\n", temp->val);
                    //sptAppendValueVector(&Z_tmp_optane[tid].values, temp->val);
                    sptAppendValueVectorNuma(&Z_tmp_optane[tid].values, temp->val);
                    node_t* pre = temp;
                    temp = temp->next;
                    free(pre);
                    //numa_free(pre, sizeof(node_t));
                }
            }
            Z_tmp_optane[tid].nnz += nnz_counter;
            for(sptIndex i = 0; i < nnz_counter; ++i) {
                for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {               
                    //sptAppendIndexVector(&Z_tmp_optane[tid].inds[m], X->inds[m].data[fx_begin]);
                    sptAppendIndexVectorNuma(&Z_tmp_optane[tid].inds[m], X->inds[m].data[fx_begin]);
                }
            }
        }
        htFree(ht);
        if(SPA_in_dram) dram_cur -= mem_SPA_cur;

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_accumulate_z += sptElapsedTime(timer_SPA);
        }
        //printf("Z:\n");
        //sptDumpSparseTensor(Z, 0, stdout);
    }   // End Loop fiber pointers of X

    //sptAssert(sptDumpSparseTensor(Z, 0, stdout) == 0);

    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[Index Search]: %.2f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.2f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

    sptStartTimer(timer);
    if(Z_mem + dram_cur < dram_cap) numa_node = dram_node;

    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp_dram[i].nnz + Z_tmp_optane[i].nnz +  Z_tmp_start[i];
        Z_total_size +=  Z_tmp_dram[i].nnz + Z_tmp_optane[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }

    result = sptNewSparseTensorWithSizeNuma(Z, nmodes_Z, ndims_buf, numa_node, Z_total_size); 
    //result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

#pragma omp parallel for schedule(static) num_threads(tk) shared(Z_tmp_dram, Z_tmp_optane, Z, nmodes_Z, Z_tmp_start)
    for(int i = 0; i < tk; i++){
        int tid = omp_get_thread_num();
        if(Z_tmp_dram[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m) 
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp_dram[tid].inds[m], Z_tmp_start[tid]);        
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp_dram[tid].values, Z_tmp_start[tid]);  
        }
        if(Z_tmp_optane[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m)  
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp_optane[tid].inds[m], Z_tmp_start[tid] + Z_tmp_dram[tid].nnz);       
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp_optane[tid].values, Z_tmp_start[tid] + Z_tmp_dram[tid].nnz);   
        }
    }    
 
    sptStopTimer(timer);

    total_time += sptPrintElapsedTime(timer, "Writeback");
    sptStartTimer(timer);

    sptSparseTensorSortIndex(Z, 1, tk);

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Output Sorting");
    printf("[Total time]: %.2f s\n", total_time);
    //system("numactl -H");
    printf("\n");
    }

if(experiment_modes == 5){
    int result;
    int dram_node;
    int optane_node;
    sscanf(getenv("DRAM_NODE"), "%d", &dram_node);
    sscanf(getenv("OPTANE_NODE"), "%d", &optane_node);
    int numa_node = dram_node;

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

    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    sptSparseTensorSortIndex(X, 1, tk);
    
    sptStopTimer(timer);
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    unsigned long long tmp_dram_size = 0;
    FILE *fp;
    char *s;
    char path[1035];
    unsigned long long  i1, i2, i3, i4, i5, i6, i7, i8;
    fp = popen("numactl -H", "r");  
    while (fgets(path, sizeof(path), fp) != NULL) {  
        s = strstr(path, "node 0 free:");      
        if (s != NULL)                     
            if (2 == sscanf(s, "%*[^0123456789]%llu%*[^0123456789]%llu", &i1, &i2)){
                tmp_dram_size = i2 * 1024 * 1024; 
                //printf("test: %llu B\n", dram_cap);
                break;
            }
    }
    pclose(fp);

    unsigned int node_size = sizeof(unsigned long long) + sizeof(unsigned int) + sizeof(unsigned int) + sizeof(unsigned long long*) + sizeof(sptValue*) + sizeof(tensor_node_t*);
    unsigned long long Y_upper_size = node_size * (Y->nnz + Y->nnz);
    //printf("%lu\n", Y_upper_size);
    if (Y_upper_size < tmp_dram_size) numa_set_preferred(dram_node);
    else numa_set_preferred(numa_node);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { 
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_Y);

    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }
    sptAssert(ci == num_cmodes);
    //for(sptIndex m = 0; m < nmodes_Y; ++m) 
    //    printf ("mode_order_Y[m]: %d\n", mode_order_Y[m]);

    table_t *Y_ht;
    unsigned int Y_ht_size = Y->nnz;
    Y_ht = tensor_htCreate(Y_ht_size);    
    
    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) {
        omp_init_lock(&locks[i]);
    }

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }
    //for(sptIndex i = 0; i <= num_cmodes;i++)
    //    printf("%d ", Y_cmode_inds[i]);
    //printf("\n");

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    //sptIndex* Y_fmode_inds = (sptIndex*) numa_alloc_onnode((Y_num_fmodes + 1) * sizeof(sptIndex), numa_node);
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }
    //for(sptIndex i = 0; i <= Y_num_fmodes;i++)
    //    printf("%d ", Y_fmode_inds[i]);
    //printf("\n");

    sptNnzIndex Y_nnz = Y->nnz;
    unsigned int Y_free_upper = 0;
    
#pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
        if(placement == 3) numa_set_preferred(optane_node);
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
        unsigned int Y_len = Y_val.len;
        if(Y_len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            if (Y_len >= Y_free_upper) Y_free_upper = Y_len + 1;
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < Y_ht_size; i++) {
        omp_destroy_lock(&locks[i]);
    }  

    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);  
    printf("[Input Processing]: %.2f s\n", sptElapsedTime(timer) + X_time );


    sptStartTimer(timer);

    //printf("Sorted X:\n");
    //sptSparseTensorStatus(X, stdout);
    //sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    //printf("Sorted Y:\n");
    //sptSparseTensorStatus(Y, stdout);
    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);

    /// Set fidx_X: indexing the combined free indices;
    sptNnzIndexVector fidx_X;
    //sptStartTimer(timer);
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);

    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[m];
    }

    /// For sorted Y 
    //for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
    //    ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[m];
    //}
    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }
    free(mode_order_X);
    free(mode_order_Y);

    // sptSparseTensor *Z_tmp = malloc(tk * sizeof (sptSparseTensor));
    sptSparseTensor *Z_tmp_dram, *Z_tmp_optane;
    if(placement == 5) {
        Z_tmp_dram = numa_alloc_onnode(tk * sizeof (sptSparseTensor), optane_node);
        Z_tmp_optane = numa_alloc_onnode(tk * sizeof (sptSparseTensor), optane_node);
    }
    else{
        Z_tmp_dram = numa_alloc_onnode(tk * sizeof (sptSparseTensor), dram_node);
        Z_tmp_optane = numa_alloc_onnode(tk * sizeof (sptSparseTensor), optane_node);
    }

    for (int i = 0; i < tk; i++){
        //result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
        result = sptNewSparseTensorNuma(&(Z_tmp_dram[i]), nmodes_Z, ndims_buf, dram_node);
        result = sptNewSparseTensorNuma(&(Z_tmp_optane[i]), nmodes_Z, ndims_buf, optane_node);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    unsigned long long dram_cur = 0;
    unsigned long long dram_cap = 0; 
    unsigned long long Z_mem = 0;

    fp = popen("numactl -H", "r");  // Open the command for reading 
    while (fgets(path, sizeof(path), fp) != NULL) {  // Read the output a line at a time - output it. 
        s = strstr(path, "node 0 free:");      // Search for string "hassasin" in buff
        if (s != NULL)                     // If successful then s now points at "hassasin"
            if (2 == sscanf(s, "%*[^0123456789]%llu%*[^0123456789]%llu", &i1, &i2)){
                //printf("System DRAM memory: %lu MB\n", i2);
                dram_cap = i2 * 1024 * 1024 / 1.1; // Should be changed into: memory of the current system - X - Y_ht
                //printf("test: %llu B\n", dram_cap);
                break;
            }
    }
    pclose(fp);

    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);

    // For the progress
    int fx_counter = fidx_X.len;

#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Z_tmp_dram, Z_tmp_optane, Y_fmode_inds, Y_ht, Y_cmode_inds, dram_cap, dram_cur, Z_mem, fx_counter) 
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
        int tid = omp_get_thread_num();
        if(placement == 4) numa_set_preferred(optane_node);
        fx_counter--;
        //if (fx_counter % 1000 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
        if (tid == 0){
            sptStartTimer(timer_SPA);
        }
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];      

        /// The total number and memory of SPA for one x fiber.
        unsigned long long num_SPA_upper = 0;
        unsigned long long mem_SPA_upper = 0;
        unsigned long long mem_SPA_cur = 0;
        bool SPA_in_dram = false;
        /// The total memory of Z_tmp
        unsigned long long Z_tmp_mem = 0;
        /// hashtable size
        const unsigned int ht_size = 10000;
        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
        long int nnz_counter = 0;
        sptIndex current_idx = 0;

        /*for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
            sptValue valX = X->values.data[zX];
            //printf("valX: %f\n", valX);
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
            if(my_len == 0) continue;
            num_SPA_upper += my_len;
        }*/

        mem_SPA_upper = (Y_free_upper + fx_end - fx_begin) * sizeof(node_t) + sizeof(node_t*) * ht_size + sizeof(table_t);
        if(mem_SPA_upper + dram_cur <= dram_cap) { // spa in dram
            dram_cur += mem_SPA_upper;
            SPA_in_dram = true;
        }

        table_t *ht;
        ht = htCreate(ht_size);
        mem_SPA_cur = sizeof( node_t*)*ht_size + sizeof( table_t);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_prep += sptElapsedTime(timer_SPA);
        }

        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
            if (tid == 0){
                sptStartTimer(timer_SPA);       
            }
            sptValue valX = X->values.data[zX];
            //printf("valX: %f\n", valX);
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

            if (tid == 0){
                sptStartTimer(timer_SPA);               
            }
            if(placement == 4) numa_set_preferred(optane_node);
            for(int i = 0; i < my_len; i++){
                unsigned long long fmode =  Y_val.key_FM[i];
                //printf("i: %d, Y_val.key_FM[i]: %lu, Y_val.val[i]: %f\n", i, Y_val.key_FM[i], Y_val.val[i]);
                sptValue spa_val = htGet(ht, fmode);
                float result = Y_val.val[i] * valX;
                if(spa_val == LONG_MIN) {
                    htInsert(ht, fmode, result);
                    mem_SPA_cur += sizeof(node_t);
                    nnz_counter++;
                }
                else    
                    htUpdate(ht, fmode, spa_val + result);
            }

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_spa += sptElapsedTime(timer_SPA);
            }
            
        }  

        if (tid == 0){
            sptStartTimer(timer_SPA);     
        }

        if(SPA_in_dram) dram_cur = dram_cur - mem_SPA_upper + mem_SPA_cur;
        Z_tmp_mem = nnz_counter * (nmodes_Z * sizeof(sptIndex) + sizeof(sptValue));
        Z_mem += Z_tmp_mem;

       
        if(Z_tmp_mem + dram_cur <= dram_cap && (tid % 7 != 0)){ 
            dram_cur += Z_tmp_mem;
            for(int i = 0; i < ht->size; i++){
                if (placement == 5 && fx_ptr%(ht_size/10) == 0) numa_set_preferred(optane_node);
                node_t *temp = ht->list[i];
                while(temp){
                    unsigned long long idx_tmp = temp->key;
                    //nnz_counter++;
                    for(sptIndex m = 0; m < nmodes_spa; ++m) {                        
                        //sptAppendIndexVector(&Z_tmp_dram[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                        sptAppendIndexVectorNuma(&Z_tmp_dram[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                    }
                    //printf("val: %f\n", temp->val);
                    //sptAppendValueVector(&Z_tmp_dram[tid].values, temp->val);
                    sptAppendValueVectorNuma(&Z_tmp_dram[tid].values, temp->val);
                    node_t* pre = temp;
                    temp = temp->next;
                    free(pre);
                    //numa_free(pre, sizeof(node_t));
                }
            }
            Z_tmp_dram[tid].nnz += nnz_counter;
            for(sptIndex i = 0; i < nnz_counter; ++i) {
                for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {               
                    //sptAppendIndexVector(&Z_tmp_dram[tid].inds[m], X->inds[m].data[fx_begin]);
                    sptAppendIndexVectorNuma(&Z_tmp_dram[tid].inds[m], X->inds[m].data[fx_begin]);
                }
            }
        }
        else{
            for(int i = 0; i < ht->size; i++){
                if (placement == 5 && fx_ptr%(ht_size/10) == 0) numa_set_preferred(optane_node);
                node_t *temp = ht->list[i];
                while(temp){
                    unsigned long long idx_tmp = temp->key;
                    //nnz_counter++;
                    for(sptIndex m = 0; m < nmodes_spa; ++m) {                        
                        //sptAppendIndexVector(&Z_tmp_optane[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                        sptAppendIndexVectorNuma(&Z_tmp_optane[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                    }
                    //printf("val: %f\n", temp->val);
                    //sptAppendValueVector(&Z_tmp_optane[tid].values, temp->val);
                    sptAppendValueVectorNuma(&Z_tmp_optane[tid].values, temp->val);
                    node_t* pre = temp;
                    temp = temp->next;
                    free(pre);
                    //numa_free(pre, sizeof(node_t));
                }
            }
            Z_tmp_optane[tid].nnz += nnz_counter;
            for(sptIndex i = 0; i < nnz_counter; ++i) {
                for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {               
                    //sptAppendIndexVector(&Z_tmp_optane[tid].inds[m], X->inds[m].data[fx_begin]);
                    sptAppendIndexVectorNuma(&Z_tmp_optane[tid].inds[m], X->inds[m].data[fx_begin]);
                }
            }
        }
        htFree(ht);
        if(SPA_in_dram) dram_cur -= mem_SPA_cur;

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_accumulate_z += sptElapsedTime(timer_SPA);
        }
        //printf("Z:\n");
        //sptDumpSparseTensor(Z, 0, stdout);
    }   // End Loop fiber pointers of X

    //sptAssert(sptDumpSparseTensor(Z, 0, stdout) == 0);

    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[Index Search]: %.2f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.2f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

    sptStartTimer(timer);
    if(Z_mem + dram_cur < dram_cap) numa_node = dram_node;

    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp_dram[i].nnz + Z_tmp_optane[i].nnz +  Z_tmp_start[i];
        Z_total_size +=  Z_tmp_dram[i].nnz + Z_tmp_optane[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }

    if(placement == 6) {
        result = sptNewSparseTensorWithSizeNuma(Z, nmodes_Z, ndims_buf, optane_node, Z_total_size); 
    }
    else{
        result = sptNewSparseTensorWithSizeNuma(Z, nmodes_Z, ndims_buf, numa_node, Z_total_size); 
    }
    //result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

#pragma omp parallel for schedule(static) num_threads(tk) shared(Z_tmp_dram, Z_tmp_optane, Z, nmodes_Z, Z_tmp_start)
    for(int i = 0; i < tk; i++){
        int tid = omp_get_thread_num();
        if(Z_tmp_dram[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m) 
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp_dram[tid].inds[m], Z_tmp_start[tid]);        
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp_dram[tid].values, Z_tmp_start[tid]);  
        }
        if(Z_tmp_optane[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m)  
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp_optane[tid].inds[m], Z_tmp_start[tid] + Z_tmp_dram[tid].nnz);       
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp_optane[tid].values, Z_tmp_start[tid] + Z_tmp_dram[tid].nnz);   
        }
    }    
 
    sptStopTimer(timer);

    total_time += sptPrintElapsedTime(timer, "Writeback");
    sptStartTimer(timer);

    sptSparseTensorSortIndex(Z, 1, tk);

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Output Sorting");
    printf("[Total time]: %.2f s\n", total_time);
    //system("numactl -H");
    printf("\n");
    }

    return 0;
}