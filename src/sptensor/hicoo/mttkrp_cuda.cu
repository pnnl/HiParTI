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
#include "hicoo.h"
#include "../../cudawrap.h"
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
int ptiCudaMTTKRPHiCOO(
    ptiSparseTensorHiCOO const * const hitsr,
    ptiMatrix ** const mats,     // mats[nmodes] as temporary space.
    ptiIndex * const mats_order,    // Correspond to the mode order of X.
    ptiIndex const mode,
    ptiNnzIndex const max_nnzb,
    int const impl_num)
{
    ptiIndex const nmodes = hitsr->nmodes;
    ptiNnzIndex const nnz = hitsr->nnz;
    ptiIndex const * const ndims = hitsr->ndims;
    ptiIndex const R = mats[mode]->ncols;
    ptiIndex const stride = mats[mode]->stride;
    int result;

    double time_h2d, time_exe, time_d2h;
    double gbw_h2d, gflops_exe, gbw_d2h;
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


    /* Copy tensor */
    /* dev_ndims: 1st gpu. */
    ptiIndex * dev_ndims;
    /* dev_cptr: 1st gpu. */
    ptiNnzIndex * dev_cptr;
    /* dev_bptr: 1st gpu. */
    ptiNnzIndex * dev_bptr;
    /* binds_header: 1st cpu, 2nd cpu (ghost pointers) */
    ptiBlockIndex ** binds_header = new ptiBlockIndex *[nmodes];
    /* dev_binds: 1st gpu, 2nd gpu. */
    ptiBlockIndex ** dev_binds;
    /* einds_header: 1st cpu, 2nd cpu (ghost pointers) */
    ptiElementIndex ** einds_header = new ptiElementIndex *[nmodes];
    /* dev_einds: 1st gpu, 2nd gpu. */
    ptiElementIndex ** dev_einds;
    /* dev_values: 1st gpu. */
    ptiValue * dev_values;


    /* Copy matrices */
    /* dev_mats_order: 1st gpu. */
    ptiIndex * dev_mats_order;
    /* mats_header: 1st cpu, 2nd cpu (ghost pointers) */
    ptiValue ** mats_header = new ptiValue *[nmodes+1];
    /* lengths: 1st cpu, store the lengths of mats */
    ptiIndex * const lengths = new ptiIndex[nmodes+1];
    /* dev_mats: 1st gpu, 2nd gpu. */
    ptiValue ** dev_mats;
    /* dev_scratch: 1st gpu. */
    ptiValue * dev_scratch;
    /* the pointer to dev_mats[nmodes] */
    ptiValue *dev_part_prod;
    ptiNnzIndex dev_mem_size = 0;
    ptiNnzIndex dev_flops = 2 * nnz * R + (nmodes - 1) * R;


    ptiStartTimer(timer);

    /* dev_ndims */
    result = ptiCudaDuplicateMemory(&dev_ndims, ndims, nmodes * sizeof (*dev_ndims), cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns SpltMTTKRP");
    dev_mem_size += nmodes * sizeof (*dev_ndims);
    /* dev_cptr */
    result = ptiCudaDuplicateMemory(&dev_cptr, hitsr->cptr.data, hitsr->cptr.len * sizeof (*dev_cptr), cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns SpltMTTKRP");
    dev_mem_size += hitsr->cptr.len * sizeof (*dev_cptr);
    /* dev_bptr */
    result = ptiCudaDuplicateMemory(&dev_bptr, hitsr->bptr.data, hitsr->bptr.len * sizeof (*dev_bptr), cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns SpltMTTKRP");
    dev_mem_size += hitsr->bptr.len * sizeof (*dev_bptr);
    /* binds_header */
    for(ptiIndex m = 0; m < nmodes; ++m) {
        binds_header[m] = hitsr->binds[m].data;
    }
    /* dev_binds */
    result = ptiCudaDuplicateMemoryIndirect(&dev_binds, binds_header, nmodes, hitsr->binds[0].len, cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns SpltMTTKRP");
    dev_mem_size += nmodes * hitsr->binds[0].len * sizeof(**dev_binds);
    /* einds_header */
    for(ptiIndex m = 0; m < nmodes; ++m) {
        einds_header[m] = hitsr->einds[m].data;
    }
    /* dev_einds */
    result = ptiCudaDuplicateMemoryIndirect(&dev_einds, einds_header, nmodes, nnz, cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns SpltMTTKRP");
    dev_mem_size += nmodes * nnz * sizeof(**dev_einds);
    /* dev_values */
    result = ptiCudaDuplicateMemory(&dev_values, hitsr->values.data, nnz * sizeof (*dev_values), cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns SpltMTTKRP");
    dev_mem_size += nnz * sizeof (*dev_values);


    /* dev_mats_order */
    result = ptiCudaDuplicateMemory(&dev_mats_order, mats_order, nmodes * sizeof (*dev_mats_order), cudaMemcpyHostToDevice);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns SpltMTTKRP");
    dev_mem_size += nmodes * sizeof (*dev_mats_order);

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
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns SpltMTTKRP");
    dev_mem_size += sum_mat_length * sizeof(**dev_mats);

    if(nmodes > 4) {
        /* dev_scratch */
        result = cudaMalloc((void **) &dev_scratch, nnz * stride * sizeof (*dev_scratch));
        pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");
        result = cudaMemset(dev_scratch, 0, nnz * stride * sizeof (*dev_scratch));
        pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");
        dev_mem_size +=  nnz * stride * sizeof (*dev_scratch);
    }

    ptiStopTimer(timer);
    time_h2d = ptiElapsedTime(timer);
    gbw_h2d = dev_mem_size / time_h2d /1e9;
    ptiPrintElapsedTime(timer, "CUDA HiCOO SpTns MTTKRP H2D");
    printf("[Bandwidth H2D]: %lf GBytes/sec\n", gbw_h2d);

    ptiStartTimer(timer);

    /* Loop kernels */
    // for(ptiIndex k=0; k<hitsr->kptr.len - 1; ++k) {
    //     ptiNnzIndex kptr_begin = hitsr->kptr.data[k];
    //     ptiNnzIndex kptr_end = hitsr->kptr.data[k+1];
        ptiNnzIndex kptr_begin = hitsr->kptr.data[0];   //useless
        ptiNnzIndex kptr_end = hitsr->kptr.data[1];   //useless

        ptiAssert( ptiMTTKRPKernelHiCOO(
            mode,
            nmodes,
            nnz,
            max_nnzb,
            R,
            stride,
            hitsr->sb_bits,
            hitsr->sc_bits,
            hitsr->bptr.len-1,
            impl_num,
            kptr_begin,
            kptr_end,
            dev_ndims,
            dev_cptr,
            dev_bptr,
            dev_binds,
            dev_einds,
            dev_values,
            dev_mats_order,
            dev_mats) == 0 );

    // }   // End loop kernels
    ptiStopTimer(timer);
    time_exe = ptiElapsedTime(timer);
    gflops_exe = dev_flops / time_exe / 1e9;
    ptiPrintElapsedTime(timer, "CUDA HiCOO SpTns MTTKRP");
    printf("[GFLOPS]: %lf GFlops \n", gflops_exe);

    ptiStartTimer(timer);

    dev_mem_size = 0;
    /* Copy back the pointer to dev_mats[nmodes] to the result */
    result = cudaMemcpy(&dev_part_prod, dev_mats + nmodes, sizeof dev_part_prod, cudaMemcpyDeviceToHost);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns SpltMTTKRP");
    dev_mem_size += sizeof dev_part_prod;

    result = cudaMemcpy(mats[nmodes]->values, dev_part_prod, mats[mode]->nrows * stride * sizeof (*dev_part_prod), cudaMemcpyDeviceToHost);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns SpltMTTKRP");
    dev_mem_size += mats[mode]->nrows * stride * sizeof (*dev_part_prod);

    ptiStopTimer(timer);
    time_d2h = ptiElapsedTime(timer);
    gbw_d2h = dev_mem_size / time_d2h /1e9;
    ptiPrintElapsedTime(timer, "CUDA HiCOO SpTns MTTKRP D2H");
    printf("[Bandwidth D2H]: %lf GBytes/sec\n", gbw_d2h);
    ptiFreeTimer(timer);


    result = cudaFree(dev_ndims);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");
    result = cudaFree(dev_cptr);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");
    result = cudaFree(dev_bptr);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");
    result = cudaFree(dev_binds);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");
    result = cudaFree(dev_einds);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");
    result = cudaFree(dev_values);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");

    result = cudaFree(dev_mats_order);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");
    result = cudaFree(dev_mats);
    pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");
    if(nmodes > 4) {
        result = cudaFree(dev_scratch);
        pti_CheckCudaError(result != 0, "CUDA HiCOO SpTns MTTKRP");
    }
    delete[] binds_header;
    delete[] einds_header;
    delete[] mats_header;
    delete[] lengths;

  return 0;
}


