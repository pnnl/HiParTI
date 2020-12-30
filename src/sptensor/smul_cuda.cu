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


static __global__ void pti_sMulKernel(
    ptiValue * __restrict__ X_val, 
    ptiNnzIndex X_nnz,
    ptiValue a)
{
    ptiNnzIndex num_loops_nnz = 1;
    ptiNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(X_nnz > nnz_per_loop) {
        num_loops_nnz = (X_nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const ptiNnzIndex tidx = threadIdx.x;
    ptiNnzIndex x;

    for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < X_nnz) {
            X_val[x] = a * X_val[x];
        }
    }

}


/**
 * Multiply a sparse tensors with a scalar.
 * @param[in]  a the input scalar
 * @param[in/out]  X the input/output tensor
 */
int ptiCudaSparseTensorMulScalar(ptiSparseTensor *X, ptiValue a)
{
    ptiAssert(a != 0.0);
    int result;

    ptiTimer timer;
    ptiNewTimer(&timer, 0);
    double comp_time;

    /* Device memory allocation */
    ptiValue *X_val = NULL;
    result = cudaMalloc((void **) &X_val, X->nnz * sizeof (ptiValue));
    pti_CheckCudaError(result != 0, "Cuda ptins MulScalar");

    /* Device memory copy */
    cudaMemcpy(X_val, X->values.data, X->nnz * sizeof (ptiValue), cudaMemcpyHostToDevice);

    ptiStartTimer(timer);

    const ptiNnzIndex max_nblocks = 32768;
    const ptiNnzIndex max_nthreads_per_block = 256;

    ptiNnzIndex nthreadsx = 1;
    ptiNnzIndex all_nblocks = 0;
    ptiNnzIndex nblocks = 0;

    if(X->nnz < max_nthreads_per_block) {
        nthreadsx = X->nnz;
        nblocks = 1;
    } else {
        nthreadsx = max_nthreads_per_block;
        all_nblocks = (X->nnz + nthreadsx -1) / nthreadsx;
        if(all_nblocks < max_nblocks) {
            nblocks = all_nblocks;
        } else {
            nblocks = max_nblocks;
        }
    }
    dim3 dimBlock(nthreadsx);
    printf("all_nblocks: %lu, nthreadsx: %lu\n", all_nblocks, nthreadsx);

    printf("[Cuda ptins MulScalar] pti_sMulKernel<<<%lu, (%lu)>>>\n", nblocks, nthreadsx);
    pti_sMulKernel<<<nblocks, dimBlock>>>(X_val, X->nnz, a);
    result = cudaThreadSynchronize();
    pti_CheckCudaError(result != 0, "Cuda ptins MulScalar kernel");

    ptiStopTimer(timer);
    comp_time = ptiPrintElapsedTime(timer, "Cuda ptins MulScalar");

    cudaMemcpy(X->values.data, X_val, X->nnz * sizeof (ptiValue), cudaMemcpyDeviceToHost);
    ptiFreeTimer(timer);

    result = cudaFree(X_val);
    pti_CheckCudaError(result != 0, "Cuda ptins MulScalar");
    
    printf("[GPU CooMulScalar]: %lf\n", comp_time);
    printf("\n");

    return 0;
}
