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

#include <HiParTI.h>
#include "sptensor.h"
#include "../cudawrap.h"
#include "mttkrp_cuda_kernels.h"



/**
 * CUDA parallelized Matriced sparse tensor times a sequence of dense matrix Khatri-Rao products (MTTKRP) on a specified mode
 * @param[out] mats[nmodes]    the result of MTTKRP, a dense matrix, with size
 * ndims[mode] * R
 * @param[in]  X    the sparse tensor input X
 * @param[in]  mats    (N+1) dense matrices, with mats[nmodes] as temporary
 * @param[in]  mats_order    the order of the Khatri-Rao products
 * @param[in]  mode   the mode on which the MTTKRP is performed
 * @param[in]  scratch an temporary array to store intermediate results, space assigned before this function
 *
 * This function uses support arbitrary-order sparse tensors with Khatri-Rao
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 * In this version, atomic function to lock the global reduction and a large
 * scratch is used to maximize parallelism. (To be optimized)
 */
int ptiCudaMTTKRPOneKernel(
    ptiSparseTensor const * const X,
    ptiMatrix ** const mats,     // mats[nmodes] as temporary space.
    ptiIndex * const mats_order,    // Correspond to the mode order of X.
    ptiIndex const mode,
    ptiIndex const impl_num)
{
    ptiIndex const nmodes = X->nmodes;
    ptiNnzIndex const nnz = X->nnz;
    ptiIndex const * const ndims = X->ndims;
    ptiIndex const R = mats[mode]->ncols;
    ptiIndex const stride = mats[mode]->stride;
    int result;

    double time_h2d, time_exe, time_d2h;
    double gbw_h2d, gflops_exe, gbytes_exe, gbw_d2h;
    ptiTimer timer;
    ptiNewTimer(&timer, 0);

    /* Check the mats. */
    for(ptiIndex i=0; i<nmodes; ++i) {
        if(mats[i]->ncols != mats[nmodes]->ncols) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
        }
        if(mats[i]->nrows != ndims[i]) {
            pti_CheckError(PTIERR_SHAPE_MISMATCH, "CUDA SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
        }
    }


    /* Transfer tensor and matrices */
    /* dev_mats_order: 1st gpu. */
    ptiIndex * dev_mats_order;
    /* dev_Xndims: 1st gpu. */
    ptiIndex * dev_Xndims;
    /* dev_Xvals: 1st gpu. */
    ptiValue * dev_Xvals;
    /* Xinds_header: 1st cpu, 2nd cpu (ghost pointers) */
    ptiIndex ** Xinds_header = new ptiIndex *[nmodes];
    /* dev_Xinds: 1st gpu, 2nd gpu. */
    ptiIndex ** dev_Xinds;
    /* mats_header: 1st cpu, 2nd cpu (ghost pointers) */
    ptiValue ** mats_header = new ptiValue *[nmodes+1];
    /* lengths: 1st cpu, store the lengths of mats */
    ptiNnzIndex * const lengths = new ptiNnzIndex[nmodes+1];
    /* dev_mats: 1st gpu, 2nd gpu. */
    ptiValue ** dev_mats;
    /* dev_scratch: 1st gpu. */
    ptiValue * dev_scratch;
    /* the pointer to dev_mats[nmodes] */
    ptiValue *dev_part_prod;
    ptiNnzIndex dev_mem_size = 0;
    uint64_t dev_flops = 2 * nnz * R + (nmodes - 1) * R;
    uint64_t dev_bytes = ( nmodes * sizeof(ptiIndex) + sizeof(ptiValue) ) * nnz;
    for (ptiIndex m=0; m<nmodes; ++m) {
        dev_bytes += ndims[m] * R * sizeof(ptiValue);
    }


    ptiStartTimer(timer);

    /* dev_mats_order */
    result = ptiCudaDuplicateMemory(&dev_mats_order, mats_order, nmodes * sizeof (ptiIndex), cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += nmodes * sizeof (ptiIndex);

    /* dev_Xndims */
    result = ptiCudaDuplicateMemory(&dev_Xndims, ndims, nmodes * sizeof (ptiIndex), cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += nmodes * sizeof (ptiIndex);

    /* dev_Xvals */
    result = ptiCudaDuplicateMemory(&dev_Xvals, X->values.data, nnz * sizeof (ptiValue), cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += nnz * sizeof (ptiValue);

    /* Xinds_header */
    for(ptiIndex m = 0; m < nmodes; ++m) {
        Xinds_header[m] = X->inds[m].data;
    }
    /* dev_Xinds */
    result = ptiCudaDuplicateMemoryIndirect(&dev_Xinds, Xinds_header, nmodes, nnz, cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += nmodes * nnz * sizeof(ptiIndex);

    /* mats_header and lengths */
    memset(mats[nmodes]->values, 0, mats[mode]->nrows * stride * sizeof(ptiValue));
    ptiNnzIndex sum_mat_length = 0;
    for(ptiIndex m = 0; m < nmodes; ++m) {
        mats_header[m] = mats[m]->values;
        lengths[m] = mats[m]->nrows * stride;
        sum_mat_length += mats[m]->nrows * stride;
    }
    mats_header[nmodes] = mats[nmodes]->values;
    lengths[nmodes] = mats[mode]->nrows * stride;
    sum_mat_length += mats[mode]->nrows * stride;
    /* dev_mats */
    result = ptiCudaDuplicateMemoryIndirect(&dev_mats, mats_header, nmodes+1, lengths, cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += sum_mat_length * sizeof(ptiValue);

    if(nmodes > 4) {
        /* dev_scratch */
        result = cudaMalloc((void **) &dev_scratch, nnz * stride * sizeof (ptiValue));
        pti_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
        result = cudaMemset(dev_scratch, 0, nnz * stride * sizeof (ptiValue));
        pti_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
        dev_mem_size +=  nnz * stride * sizeof (ptiValue);
    }

    ptiStopTimer(timer);
    time_h2d = ptiElapsedTime(timer);
    gbw_h2d = dev_mem_size / time_h2d /1e9;
    ptiPrintElapsedTime(timer, "CUDA SpTns MTTKRP H2D");
    printf("[Bandwidth H2D]: %lf GBytes/sec\n", gbw_h2d);


    // ptiNnzIndex max_nthreads_per_block = 512;    // old run
    ptiNnzIndex max_nthreads_per_block = 256;
    ptiNnzIndex max_nblocks = 32768;
    ptiNnzIndex max_nthreadsy = 16;

    ptiNnzIndex nthreadsx = 0;
    ptiNnzIndex nthreadsy = 0;
    ptiNnzIndex all_nblocks = 0;
    ptiNnzIndex nblocks = 0;
    switch(impl_num) {
    // case 1:
    case 11: // Naive, 1D
        if(nnz < max_nthreads_per_block) {
            nthreadsx = nnz;
            nblocks = 1;
        } else {
            nthreadsx = max_nthreads_per_block;
            all_nblocks = (nnz + nthreadsx -1) / nthreadsx;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }
        }
        break;
    // case 2: // 2D
    case 12:
        if(R <= max_nthreadsy)
            nthreadsy = R;
        else
            nthreadsy = max_nthreadsy;
        nthreadsx = max_nthreads_per_block / nthreadsy;

        if(nnz < nthreadsx) {
            nthreadsx = nnz;
            nblocks = 1;
        } else {
            all_nblocks = (nnz + nthreadsx -1) / nthreadsx;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }   
        }
        break;
    // case 3: // 2D, rank split
    //     if(R <= max_nthreadsy)
    //         nthreadsy = R;
    //     else
    //         nthreadsy = max_nthreadsy;
    //     nthreadsx = max_nthreads_per_block / nthreadsy;
    //     all_nblocks = (nnz + nthreadsx -1) / nthreadsx;
    //     break;
    // case 4: // 2D, exchange x and y
    //     nthreadsx = R;
    //     nthreadsy = max_nthreads_per_block / nthreadsx;
    //     all_nblocks = (nnz + nthreadsy -1) / nthreadsy;
    //     break;
    // case 5:
    case 15: // 2D, exchange x and y, rank split. Best performance
    case 16:
        if(R <= max_nthreadsy)
            nthreadsx = R;
        else
            nthreadsx = max_nthreadsy;
        nthreadsy = max_nthreads_per_block / nthreadsx;

        if(nnz < nthreadsy) {
            nthreadsy = nnz;
            nblocks = 1;
        } else {
            all_nblocks = (nnz + nthreadsy -1) / nthreadsy;
            if(all_nblocks < max_nblocks) {
                nblocks = all_nblocks;
            } else {
                nblocks = max_nblocks;
            }   
        }
        break;
    }
    dim3 dimBlock(nthreadsx, nthreadsy);
    printf("all_nblocks: %lu, nthreadsx: %lu, nthreadsy: %lu\n", all_nblocks, nthreadsx, nthreadsy);


    ptiStartTimer(timer);

    switch(nmodes) {
    case 3:
        switch(impl_num) {
        // case 1:
        case 11: // Naive
            printf("Execute pti_MTTKRPKernelNnz3DOneKernel (%lu, %lu)\n", nblocks, nthreadsx);
            pti_MTTKRPKernelNnz3DOneKernel<<<nblocks, nthreadsx>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims,
                dev_Xinds,
                dev_Xvals,
                dev_mats_order,
                dev_mats);
            break;
        // case 2:
        case 12:
            printf("Execute pti_MTTKRPKernelRankNnz3DOneKernel (%lu, (%u, %u))\n", nblocks, dimBlock.x, dimBlock.y);
            pti_MTTKRPKernelRankNnz3DOneKernel<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims,
                dev_Xinds,
                dev_Xvals,
                dev_mats_order,
                dev_mats);
            break;
        case 3:
            printf("Execute pti_MTTKRPKernelNnzRankSplit3D (%lu, (%u, %u))\n", nblocks, dimBlock.x, dimBlock.y);
            // pti_MTTKRPKernelNnzRankSplit3D<<<nblocks, dimBlock>>>(
            //     mode,
            //     nmodes,
            //     nnz,
            //     R,
            //     stride,
            //     dev_Xndims,
            //     dev_Xinds,
            //     dev_Xvals,
            //     dev_mats_order,
            //     dev_mats,
            //     block_offset);
            break;
        case 4:
            printf("Execute pti_MTTKRPKernelRankNnz3D (%lu, (%u, %u))\n", nblocks, dimBlock.x, dimBlock.y);
            // pti_MTTKRPKernelRankNnz3D<<<nblocks, dimBlock>>>(
            //     mode,
            //     nmodes,
            //     nnz,
            //     R,
            //     stride,
            //     dev_Xndims,
            //     dev_Xinds,
            //     dev_Xvals,
            //     dev_mats_order,
            //     dev_mats,
            //     block_offset);
            break;
        // case 5:
        case 15:
            printf("Execute pti_MTTKRPKernelRankSplitNnz3DOneKernel (%lu, (%u, %u))\n", nblocks, dimBlock.x, dimBlock.y);
            pti_MTTKRPKernelRankSplitNnz3DOneKernel<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims,
                dev_Xinds,
                dev_Xvals,
                dev_mats_order,
                dev_mats);
            break;
        case 16:
            printf("Execute pti_MTTKRPKernelRankSplitNnzRB3DOneKernel (%lu, (%u, %u))\n", nblocks, dimBlock.x, dimBlock.y);
            pti_MTTKRPKernelRankSplitNnzRB3DOneKernel<<<nblocks, dimBlock>>>(
                mode,
                nmodes,
                nnz,
                R,
                stride,
                dev_Xndims,
                dev_Xinds,
                dev_Xvals,
                dev_mats_order,
                dev_mats);
            break;
        }   // End switch impl_num
        break;

    case 4: 
        switch(impl_num) {
        default:
            printf("Not support: Execute pti_MTTKRPKernelScratch (%lu, %lu)\n", nblocks, nthreadsx);
            // pti_MTTKRPKernelScratch<<<nblocks, nthreadsx>>>(
            //     mode,
            //     nmodes,
            //     nnz,
            //     R,
            //     stride,
            //     dev_Xndims,
            //     dev_Xinds,
            //     dev_Xvals,
            //     dev_mats_order,
            //     dev_mats,
            //     dev_scratch,
            //     block_offset);
        }   // End switch impl_num
        break;

    default:
        printf("Execute pti_MTTKRPKernelScratch (%lu, %lu)\n", nblocks, nthreadsx);
        // pti_MTTKRPKernelScratch<<<nblocks, nthreadsx>>>(
        //     mode,
        //     nmodes,
        //     nnz,
        //     R,
        //     stride,
        //     dev_Xndims,
        //     dev_Xinds,
        //     dev_Xvals,
        //     dev_mats_order,
        //     dev_mats,
        //     dev_scratch,
        //     block_offset);
    }   // End switch nmodes
    result = cudaThreadSynchronize();
    pti_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");



    ptiStopTimer(timer);
    time_exe = ptiElapsedTime(timer);
    gflops_exe = (double)dev_flops / time_exe / 1e9;
    gbytes_exe = (double)dev_bytes / time_exe / 1e9;
    ptiPrintElapsedTime(timer, "CUDA SpTns MTTKRP");
    printf("[GFLOPS]: %.2lf GFlops, [Bandwidth]: %.2lf GB/s\n", gflops_exe, gbytes_exe);

    ptiStartTimer(timer);

    dev_mem_size = 0;
    /* Copy back the pointer to dev_mats[nmodes] to the result */
    result = cudaMemcpy(&dev_part_prod, dev_mats + nmodes, sizeof dev_part_prod, cudaMemcpyDeviceToHost);
    pti_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += sizeof dev_part_prod;

    result = cudaMemcpy(mats[nmodes]->values, dev_part_prod, mats[mode]->nrows * stride * sizeof (ptiValue), cudaMemcpyDeviceToHost);
    pti_CheckCudaError(result != 0, "CUDA SpTns SpltMTTKRP");
    dev_mem_size += mats[mode]->nrows * stride * sizeof (ptiValue);

    ptiStopTimer(timer);
    time_d2h = ptiElapsedTime(timer);
    gbw_d2h = dev_mem_size / time_d2h /1e9;
    ptiPrintElapsedTime(timer, "CUDA SpTns MTTKRP D2H");
    printf("[Bandwidth D2H]: %lf GBytes/sec\n", gbw_d2h);
    printf("\n");
    ptiFreeTimer(timer);

    result = cudaFree(dev_mats_order);
    pti_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaFree(dev_Xndims);
    pti_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaFree(dev_Xvals);
    pti_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaFree(dev_Xinds);
    pti_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    result = cudaFree(dev_mats);
    pti_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    if(nmodes > 4) {
        result = cudaFree(dev_scratch);
        pti_CheckCudaError(result != 0, "CUDA SpTns MTTKRP");
    }
    delete[] Xinds_header;
    delete[] mats_header;
    delete[] lengths;

  return 0;
}


