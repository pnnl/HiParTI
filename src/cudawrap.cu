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
#include <stdlib.h>
#include "cudawrap.h"
#include <cusparse.h>
#include <cusolverSp.h>

int ptiCudaSetDevice(int device) {
    return (int) cudaSetDevice(device);
}

int ptiCudaGetLastError(void) {
    return (int) cudaGetLastError();
}

int pti_cusparseCreate(cusparseHandle_t *handle) {
    static cusparseHandle_t h = NULL;
    int result = 0;
    if(!h) {
        result = cusparseCreate(&h);
    }
    *handle = h;
    return result;
}

int pti_cusolverSpCreate(cusolverSpHandle_t *handle) {
    static cusolverSpHandle_t h = NULL;
    int result = 0;
    if(!h) {
        result = cusolverSpCreate(&h);
    }
    *handle = h;
    return result;
}

int pti_CudaDuplicateMemoryGenerics(void **dest, const void *src, size_t size, int direction) {
    int result;
    switch(direction) {
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDeviceToDevice:
        result = cudaMalloc(dest, size);
        pti_CheckCudaError(result != 0, "ptiCudaDuplicateMemory");
        break;
    case cudaMemcpyDeviceToHost:
        *dest = malloc(size);
        pti_CheckOSError(*dest == NULL, "ptiCudaDuplicateMemory");
        break;
    default:
        pti_CheckError(PTIERR_UNKNOWN, "ptiCudaDuplicateMemory", "Unknown memory copy kind")
    }
    result = cudaMemcpy(*dest, src, size, (cudaMemcpyKind) direction);
    pti_CheckCudaError(result != 0, "ptiCudaDuplicateMemory");
    return 0;
}


int pti_CudaDuplicateMemoryGenericsAsync(void **dest, const void *src, size_t size, int direction, cudaStream_t stream) {
    int result;
    switch(direction) {
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDeviceToDevice:
        result = cudaMalloc(dest, size);
        pti_CheckCudaError(result != 0, "ptiCudaDuplicateMemory");
        break;
    case cudaMemcpyDeviceToHost:
        *dest = malloc(size);
        pti_CheckOSError(*dest == NULL, "ptiCudaDuplicateMemory");
        break;
    default:
        pti_CheckError(PTIERR_UNKNOWN, "ptiCudaDuplicateMemory", "Unknown memory copy kind")
    }
    result = cudaMemcpyAsync(*dest, src, size, (cudaMemcpyKind) direction, stream);
    pti_CheckCudaError(result != 0, "ptiCudaDuplicateMemory");
    return 0;
}
