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

sptMutexPool * SptMutexAllocCustom(
    sptIndex const num_locks,
    sptIndex const pad_size)
{
#ifdef PARTI_USE_OPENMP
  sptMutexPool * pool = (sptMutexPool*)malloc(sizeof(*pool));

  pool->nlocks = num_locks;
  pool->padsize = pad_size;

  pool->locks = (omp_lock_t*)malloc(num_locks * pad_size * sizeof(*pool->locks));
  for(sptIndex l=0; l < num_locks; ++l) {
    sptIndex const lock = sptMutexTranslateId(l, num_locks, pad_size);
    omp_init_lock(pool->locks + lock);
  }

  return pool;
#else
  fprintf(stderr, "OpenMP support is not enabled.\n");
  abort();
#endif
}


sptMutexPool * sptMutexAlloc()
{
#ifdef PARTI_USE_OPENMP
  return SptMutexAllocCustom(PARTI_DEFAULT_NLOCKS, PARTI_DEFAULT_LOCK_PAD_SIZE);
#else
  fprintf(stderr, "OpenMP support is not enabled.\n");
  abort();
#endif
}


void sptMutexFree(
    sptMutexPool * pool)
{
#ifdef PARTI_USE_OPENMP
  for(sptIndex l=0; l < pool->nlocks; ++l) {
    sptIndex const lock = sptMutexTranslateId(l, pool->nlocks, pool->padsize);
    omp_destroy_lock(pool->locks + lock);
  }

  free(pool->locks);
  free(pool);
#else
  fprintf(stderr, "OpenMP support is not enabled.\n");
  abort();
#endif
}
