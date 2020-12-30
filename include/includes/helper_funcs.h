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

#ifndef HIPARTI_HELPER_FUNCS_H
#define HIPARTI_HELPER_FUNCS_H

#include <stdlib.h>

#define max(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a > _b ? _a : _b; })

int ptiGetLastError(const char **module, const char **file, unsigned *line, const char **reason);
void ptiClearLastError(void);
void pti_Panic(const char *file, unsigned line, const char *expr);
/**
 * The assert function that always execute even when `NDEBUG` is set
 *
 * Quick & dirty error checking. Useful when writing small programs.
 */
#define ptiAssert(expr) ((expr) ? (void) 0 : pti_Panic(__FILE__, __LINE__, #expr))

/* Helper function for pure C module */
int ptiCudaSetDevice(int device);
int ptiCudaGetLastError(void);

/* Timer functions, using either CPU or GPU timer */
int ptiNewTimer(ptiTimer *timer, int use_cuda);
int ptiStartTimer(ptiTimer timer);
int ptiStopTimer(ptiTimer timer);
double ptiElapsedTime(const ptiTimer timer);
double ptiPrintElapsedTime(const ptiTimer timer, const char *name);
double ptiPrintAverageElapsedTime(const ptiTimer timer, const int niters, const char *name);
int ptiFreeTimer(ptiTimer timer);
double ptiPrintGFLOPS(const double elapsed_time, const ptiNnzIndex flops, const char *name);

/* Base functions */
char * ptiBytesString(uint64_t const bytes);
ptiValue ptiRandomValue(void);


/**
 * OMP Lock functions
 */
ptiMutexPool * ptiMutexAlloc();
ptiMutexPool * SptMutexAllocCustom(
    ptiIndex const num_locks,
    ptiIndex const pad_size);
void ptiMutexFree(ptiMutexPool * pool);

static inline ptiIndex ptiMutexTranslateId(
    ptiIndex const id,
    ptiIndex const num_locks,
    ptiIndex const pad_size)
{
  return (id % num_locks) * pad_size;
}

static inline void ptiMutexSetLock(
    ptiMutexPool * const pool,
    ptiIndex const id)
{
#ifdef HIPARTI_USE_OPENMP
  ptiIndex const lock_id = ptiMutexTranslateId(id, pool->nlocks, pool->padsize);
  omp_set_lock(pool->locks + lock_id);
#else
  fprintf(stderr, "OpenMP support is not enabled.\n");
  abort();
#endif
}

static inline void ptiMutexUnsetLock(
    ptiMutexPool * const pool,
    ptiIndex const id)
{
#ifdef HIPARTI_USE_OPENMP
  ptiIndex const lock_id = ptiMutexTranslateId(id, pool->nlocks, pool->padsize);
  omp_unset_lock(pool->locks + lock_id);
#else
  fprintf(stderr, "OpenMP support is not enabled.\n");
  abort();
#endif
}


#endif
