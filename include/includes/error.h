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

#ifndef HIPARTI_ERROR_H_INCLUDED
#define HIPARTI_ERROR_H_INCLUDED

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

/**
 * Check if a value is not zero, print error message and return.
 * @param errcode the value to be checked
 * @param module  the module name of current procedure
 * @param reason  human readable error explanation
 */
#ifndef NDEBUG
#define pti_CheckError(errcode, module, reason) \
    if(unlikely((errcode) != 0)) { \
        pti_ComplainError(module, (errcode), __FILE__, __LINE__, (reason)); \
        return (errcode); \
    }
#else
#define pti_CheckError(errcode, module, reason) \
    if(unlikely((errcode) != 0)) { \
        return (errcode); \
    }
#endif

#ifndef NDEBUG
#define pti_CheckOmpError(errcode, module, reason) \
    if(unlikely((errcode) != 0)) { \
        pti_ComplainError(module, (errcode), __FILE__, __LINE__, (reason)); \
        exit(errcode); \
    }
#else
#define pti_CheckOmpError(errcode, module, reason) \
    if(unlikely((errcode) != 0)) { \
        exit(errcode); \
    }
#endif

/**
 * Check if a condition is true, set the error information as the system error, print error message and return.
 * @param cond   the condition to be checked
 * @param module the module name of current procedure
 */
#define pti_CheckOSError(cond, module) \
    if(unlikely((cond))) { \
        pti_CheckError(errno + PTIERR_OS_ERROR, (module), strerror(errno)); \
    }

/**
 * Check if a condition is true, set the error information as the last CUDA error, print error message and return.
 * @param cond   the condition to be checked
 * @param module the module name of current procedure
 */
#define pti_CheckCudaError(cond, module) \
    if(unlikely((cond))) { \
        cudaError_t _cuda_error = cudaGetLastError(); \
        pti_CheckError(_cuda_error + PTIERR_CUDA_ERROR, (module), cudaGetErrorString(_cuda_error)); \
    }

void pti_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason);

#ifdef __cplusplus
}
#endif

#endif
