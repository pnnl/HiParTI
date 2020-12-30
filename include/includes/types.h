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

#ifndef HIPARTI_TYPES_H
#define HIPARTI_TYPES_H

#include <stdint.h>

/**
 * Define types, TODO: check the bit size of them, add branch for different settings
 */
#define HIPARTI_INDEX_TYPEWIDTH 32
#define HIPARTI_VALUE_TYPEWIDTH 32
#define HIPARTI_ELEMENT_INDEX_TYPEWIDTH 8

#if HIPARTI_INDEX_TYPEWIDTH == 32
  typedef uint32_t ptiIndex;
  typedef uint32_t ptiBlockIndex;
  #define HIPARTI_INDEX_MAX UINT32_MAX
  #define HIPARTI_PRI_INDEX PRIu32
  #define HIPARTI_SCN_INDEX SCNu32
  #define HIPARTI_PRI_BLOCK_INDEX PRIu32
  #define HIPARTI_SCN_BLOCK_INDEX SCNu32
#elif HIPARTI_INDEX_TYPEWIDTH == 64
  typedef uint64_t ptiIndex;
  typedef uint64_t ptiBlockIndex;
  #define HIPARTI_INDEX_MAX UINT64_MAX
  #define HIPARTI_PFI_INDEX PRIu64
  #define HIPARTI_SCN_INDEX SCNu64
  #define HIPARTI_PRI_BLOCK_INDEX PRIu64
  #define HIPARTI_SCN_BLOCK_INDEX SCNu64
#else
  #error "Unrecognized HIPARTI_INDEX_TYPEWIDTH."
#endif

#if HIPARTI_VALUE_TYPEWIDTH == 32
  typedef float ptiValue;
  #define HIPARTI_PRI_VALUE "f"
  #define HIPARTI_SCN_VALUE "f"
#elif HIPARTI_VALUE_TYPEWIDTH == 64
  typedef double ptiValue;
  #define HIPARTI_PRI_VALUE "lf"
  #define HIPARTI_SCN_VALUE "lf"
#else
  #error "Unrecognized HIPARTI_VALUE_TYPEWIDTH."
#endif

#if HIPARTI_ELEMENT_INDEX_TYPEWIDTH == 8
  // typedef uint_fast8_t ptiElementIndex;
  // typedef uint_fast16_t ptiBlockMatrixIndex;  // R < 256
  // #define HIPARTI_PRI_ELEMENT_INDEX PRIuFAST8
  // #define HIPARTI_SCN_ELEMENT_INDEX SCNuFAST8
  // #define HIPARTI_PRI_BLOCKMATRIX_INDEX PRIuFAST16
  // #define HIPARTI_SCN_BLOCKMATRIX_INDEX SCNuFAST16
  typedef uint8_t ptiElementIndex;
  typedef uint16_t ptiBlockMatrixIndex;  // R < 256
  #define HIPARTI_PRI_ELEMENT_INDEX PRIu8
  #define HIPARTI_SCN_ELEMENT_INDEX SCNu8
  #define HIPARTI_PRI_BLOCKMATRIX_INDEX PRIu16
  #define HIPARTI_SCN_BLOCKMATRIX_INDEX SCNu16
#elif HIPARTI_ELEMENT_INDEX_TYPEWIDTH == 16
  typedef uint16_t ptiElementIndex;
  typedef uint32_t ptiBlockMatrixIndex;
  #define HIPARTI_PFI_ELEMENT_INDEX PRIu16
  #define HIPARTI_SCN_ELEMENT_INDEX SCNu16
  #define HIPARTI_PRI_BLOCKMATRIX_INDEX PRIu32
  #define HIPARTI_SCN_BLOCKMATRIX_INDEX SCNu32
#elif HIPARTI_ELEMENT_INDEX_TYPEWIDTH == 32
  typedef uint32_t ptiElementIndex;
  typedef uint32_t ptiBlockMatrixIndex;
  #define HIPARTI_PFI_ELEMENT_INDEX PRIu32
  #define HIPARTI_SCN_ELEMENT_INDEX SCNu32
  #define HIPARTI_PRI_BLOCKMATRIX_INDEX PRIu32
  #define HIPARTI_SCN_BLOCKMATRIX_INDEX SCNu32
#else
  #error "Unrecognized HIPARTI_ELEMENT_INDEX_TYPEWIDTH."
#endif

typedef ptiBlockIndex ptiBlockNnzIndex;
#define HIPARTI_PRI_BLOCKNNZ_INDEX HIPARTI_PRI_BLOCK_INDEX
#define HIPARTI_SCN_BLOCKNNZ_INDEX HIPARTI_SCN_BLOCK_INDEX

typedef uint64_t ptiNnzIndex;
#define HIPARTI_NNZ_INDEX_MAX UINT64_MAX
#define HIPARTI_PRI_NNZ_INDEX PRIu64
#define HIPARTI_SCN_NNZ_INDEX PRIu64

typedef unsigned __int128 ptiMortonIndex;
// typedef __uint128_t ptiMortonIndex;


#endif