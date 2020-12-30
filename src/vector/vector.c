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
#include <string.h>


/**
 * Initialize a new value vector
 *
 * @param vec a valid pointer to an uninitialized ptiValueVector variable,
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int ptiNewValueVector(ptiValueVector *vec, ptiNnzIndex len, ptiNnzIndex cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    pti_CheckOSError(!vec->data, "ValVec New");
    memset(vec->data, 0, cap * sizeof *vec->data);
    return 0;
}

/**
 * Fill an existed dense value vector with a specified constant
 *
 * @param vec   a valid pointer to an existed ptiVector variable,
 * @param val   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int ptiConstantValueVector(ptiValueVector * const vec, ptiValue const val) {
    for(ptiNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = val;
    return 0;
}

/**
 * Fill an existed dense value vector with a randomized values
 *
 * @param vec   a valid pointer to an existed ptiVector variable,
 * @param val   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int ptiRandomValueVector(ptiValueVector * const vec) {
    // srand(time(NULL));
    for(ptiNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = rand() % 10 + 1;
    return 0;
}

/**
 * Copy a value vector to an uninitialized value vector
 *
 * @param dest a pointer to an uninitialized value vector
 * @param src  a pointer to an existing valid value vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int ptiCopyValueVector(ptiValueVector *dest, const ptiValueVector *src, int const nt) {
    int result = ptiNewValueVector(dest, src->len, src->len);
    pti_CheckError(result, "ValVec Copy", NULL);
#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for num_threads(nt)
    for (ptiNnzIndex i=0; i<src->len; ++i) {
        dest->data[i] = src->data[i];
    }
#else
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
#endif
    return 0;
}

/**
 * Add a value to the end of a value vector
 *
 * @param vec   a pointer to a valid value vector
 * @param value the value to be appended
 *
 * The length of the value vector will be changed to contain the new value.
 */
int ptiAppendValueVector(ptiValueVector *vec, ptiValue const value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        ptiNnzIndex newcap = vec->cap + vec->cap/2;
#else
        ptiNnzIndex newcap = vec->len+1;
#endif
        ptiValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "ValVec Append");
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

/**
 * Add a value to the end of a value vector
 *
 * @param vec        a pointer to a valid value vector
 * @param append_vec a pointer to another value vector, containing the values to be appended
 *
 * The values from `append_vec` will be appended to `vec`.
 */
int ptiAppendValueVectorWithVector(ptiValueVector *vec, const ptiValueVector *append_vec) {
    ptiNnzIndex newlen = vec->len + append_vec->len;
    if(vec->cap <= newlen) {
        ptiNnzIndex newcap = vec->cap + append_vec->cap;
        ptiValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "ValVec Append ValVec");
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(ptiNnzIndex i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
    }
    vec->len = newlen;

    return 0;
}

/**
 * Resize a value vector
 *
 * @param vec  the value vector to resize
 * @param size the new size of the value vector
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int ptiResizeValueVector(ptiValueVector *vec, ptiNnzIndex const size) {
    ptiNnzIndex newcap = size < 2 ? 2 : size;
    if(newcap != vec->cap) {
        ptiValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "ValVec Resize");
        vec->len = size;
        vec->cap = newcap;
        vec->data = newdata;
    } else {
        vec->len = size;
    }
    return 0;
}

/**
 * Release the memory buffer a value vector is holding
 *
 * @param vec a pointer to a valid value vector
 *
 */
void ptiFreeValueVector(ptiValueVector *vec) {
    vec->len = 0;
    vec->cap = 0;
    free(vec->data);
}


/*
 * Initialize a new ptiIndex vector
 *
 * @param vec a valid pointer to an uninitialized ptiIndex variable,
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int ptiNewIndexVector(ptiIndexVector *vec, ptiNnzIndex len, ptiNnzIndex cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    pti_CheckOSError(!vec->data, "IdxVec New");
    memset(vec->data, 0, cap * sizeof *vec->data);
    return 0;
}

/**
 * Fill an existed dense index vector with a specified constant
 *
 * @param vec   a valid pointer to an existed ptiIndexVector variable,
 * @param num   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int ptiConstantIndexVector(ptiIndexVector * const vec, ptiIndex const num) {
    for(ptiNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = num;
    return 0;
}

/**
 * Copy an index vector to an uninitialized index vector
 *
 * @param dest a pointer to an uninitialized index vector
 * @param src  a pointer to an existing valid index vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int ptiCopyIndexVector(ptiIndexVector *dest, const ptiIndexVector *src, int const nt) {
    int result = ptiNewIndexVector(dest, src->len, src->len);
    pti_CheckError(result, "IdxVec Copy", NULL);
#ifdef HIPARTI_USE_OPENMP
    #pragma omp parallel for num_threads(nt)
    for (ptiNnzIndex i=0; i<src->len; ++i) {
        dest->data[i] = src->data[i];
    }
#else
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
#endif
    return 0;
}


/**
 * Add a value to the end of a ptiIndexVector
 *
 * @param vec   a pointer to a valid index vector
 * @param value the value to be appended
 *
 * The length of the size vector will be changed to contain the new value.
 */
int ptiAppendIndexVector(ptiIndexVector *vec, ptiIndex const value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        ptiNnzIndex newcap = vec->cap + vec->cap/2;
#else
        ptiNnzIndex newcap = vec->len+1;
#endif
        ptiIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "IdxVec Append");
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

/**
 * Add a value to the end of an index vector
 *
 * @param vec        a pointer to a valid index vector
 * @param append_vec a pointer to another index vector, containing the values to be appended
 *
 * The values from `append_vec` will be appended to `vec`.
 */
int ptiAppendIndexVectorWithVector(ptiIndexVector *vec, const ptiIndexVector *append_vec) {
    ptiNnzIndex newlen = vec->len + append_vec->len;
    if(vec->cap <= newlen) {
        ptiNnzIndex newcap = vec->cap + append_vec->cap;
        ptiIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "IdxVec Append IdxVec");
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(ptiNnzIndex i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
    }
    vec->len = newlen;

    return 0;
}

/**
 * Resize an index vector
 *
 * @param vec  the index vector to resize
 * @param size the new size of the index vector
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int ptiResizeIndexVector(ptiIndexVector *vec, ptiNnzIndex const size) {
    ptiNnzIndex newcap = size < 2 ? 2 : size;
    if(newcap != vec->cap) {
        ptiIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "IdxVec Resize");
        vec->len = size;
        vec->cap = newcap;
        vec->data = newdata;
    } else {
        vec->len = size;
    }
    return 0;
}

/**
 * Release the memory buffer a ptiIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void ptiFreeIndexVector(ptiIndexVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}


/*
 * Initialize a new ptiElementIndexVector vector
 *
 * @param vec a valid pointer to an uninitialized ptiElementIndex variable,
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int ptiNewElementIndexVector(ptiElementIndexVector *vec, ptiNnzIndex len, ptiNnzIndex cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    pti_CheckOSError(!vec->data, "EleIdxVec New");
    memset(vec->data, 0, cap * sizeof *vec->data);
    return 0;
}

/**
 * Fill an existed dense element index vector with a specified constant
 *
 * @param vec   a valid pointer to an existed ptiElementIndexVector variable,
 * @param num   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int ptiConstantElementIndexVector(ptiElementIndexVector * const vec, ptiElementIndex const num) {
    for(ptiNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = num;
    return 0;
}

/**
 * Copy an element index vector to an uninitialized element index vector
 *
 * @param dest a pointer to an uninitialized element index vector
 * @param src  a pointer to an existing valid element index vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int ptiCopyElementIndexVector(ptiElementIndexVector *dest, const ptiElementIndexVector *src) {
    int result = ptiNewElementIndexVector(dest, src->len, src->len);
    pti_CheckError(result, "EleIdxVec Copy", NULL);
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
    return 0;
}

/**
 * Add a value to the end of a ptiElementIndexVector
 *
 * @param vec   a pointer to a valid element index vector
 * @param value the value to be appended
 *
 * The length of the element index vector will be changed to contain the new value.
 */
int ptiAppendElementIndexVector(ptiElementIndexVector *vec, ptiElementIndex const value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        ptiNnzIndex newcap = vec->cap + vec->cap/2;
#else
        ptiNnzIndex newcap = vec->len+1;
#endif
        ptiElementIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "EleIdxVec Append");
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

/**
 * Add a value to the end of an element index vector
 *
 * @param vec        a pointer to a valid element index vector
 * @param append_vec a pointer to another element index vector, containing the values to be appended
 *
 * The values from `append_vec` will be appended to `vec`.
 */
int ptiAppendElementIndexVectorWithVector(ptiElementIndexVector *vec, const ptiElementIndexVector *append_vec) {
    ptiNnzIndex newlen = vec->len + append_vec->len;
    if(vec->cap <= newlen) {
        ptiNnzIndex newcap = vec->cap + append_vec->cap;
        ptiElementIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "EleIdxVec Append EleIdxVec");
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(ptiNnzIndex i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
    }
    vec->len = newlen;

    return 0;
}

/**
 * Resize a element index vector
 *
 * @param vec  the element index vector to resize
 * @param size the new size of the element index vector
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int ptiResizeElementIndexVector(ptiElementIndexVector *vec, ptiNnzIndex const size) {
    ptiNnzIndex newcap = size < 2 ? 2 : size;
    if(newcap != vec->cap) {
        ptiElementIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "EleIdxVec Resize");
        vec->len = size;
        vec->cap = newcap;
        vec->data = newdata;
    } else {
        vec->len = size;
    }
    return 0;
}

/**
 * Release the memory buffer a ptiElementIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void ptiFreeElementIndexVector(ptiElementIndexVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}


/*
 * Initialize a new ptiBlockIndexVector vector
 *
 * @param vec a valid pointer to an uninitialized ptiBlockIndex variable,
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int ptiNewBlockIndexVector(ptiBlockIndexVector *vec, ptiNnzIndex len, ptiNnzIndex cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    pti_CheckOSError(!vec->data, "BlkIdxVec New");
    memset(vec->data, 0, cap * sizeof *vec->data);
    return 0;
}

/**
 * Fill an existed dense element index vector with a specified constant
 *
 * @param vec   a valid pointer to an existed ptiBlockIndexVector variable,
 * @param num   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int ptiConstantBlockIndexVector(ptiBlockIndexVector * const vec, ptiBlockIndex const num) {
    for(ptiNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = num;
    return 0;
}

/**
 * Copy a block index vector to an uninitialized block index vector
 *
 * @param dest a pointer to an uninitialized block index vector
 * @param src  a pointer to an existing valid block index vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int ptiCopyBlockIndexVector(ptiBlockIndexVector *dest, const ptiBlockIndexVector *src) {
    int result = ptiNewBlockIndexVector(dest, src->len, src->len);
    pti_CheckError(result, "BlkIdxVec Copy", NULL);
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
    return 0;
}

/**
 * Add a value to the end of a ptiBlockIndexVector
 *
 * @param vec   a pointer to a valid block index vector
 * @param value the value to be appended
 *
 * The length of the block index vector will be changed to contain the new value.
 */
int ptiAppendBlockIndexVector(ptiBlockIndexVector *vec, ptiBlockIndex const value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        ptiNnzIndex newcap = vec->cap + vec->cap/2;
#else
        ptiNnzIndex newcap = vec->len+1;
#endif
        ptiBlockIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "BlkIdxVec Append");
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

/**
 * Add a value to the end of a block index vector
 *
 * @param vec        a pointer to a valid block index vector
 * @param append_vec a pointer to another block index vector, containing the values to be appended
 *
 * The values from `append_vec` will be appended to `vec`.
 */
int ptiAppendBlockIndexVectorWithVector(ptiBlockIndexVector *vec, const ptiBlockIndexVector *append_vec) {
    ptiNnzIndex newlen = vec->len + append_vec->len;
    if(vec->cap <= newlen) {
        ptiNnzIndex newcap = vec->cap + append_vec->cap;
        ptiBlockIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "BlkIdxVec Append BlkIdxVec");
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(ptiNnzIndex i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
    }
    vec->len = newlen;

    return 0;
}

/**
 * Resize a block index vector
 *
 * @param vec  the block index vector to resize
 * @param size the new size of the block index vector
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int ptiResizeBlockIndexVector(ptiBlockIndexVector *vec, ptiNnzIndex const size) {
    ptiNnzIndex newcap = size < 2 ? 2 : size;
    if(newcap != vec->cap) {
        ptiBlockIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "BlkIdxVec Resize");
        vec->len = size;
        vec->cap = newcap;
        vec->data = newdata;
    } else {
        vec->len = size;
    }
    return 0;
}

/**
 * Release the memory buffer a ptiBlockIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void ptiFreeBlockIndexVector(ptiBlockIndexVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}


/*
 * Initialize a new ptiNnzIndexVector vector
 *
 * @param vec a valid pointer to an uninitialized ptiNnzIndex variable,
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int ptiNewNnzIndexVector(ptiNnzIndexVector *vec, ptiNnzIndex len, ptiNnzIndex cap) {
    if(cap < len) {
        cap = len;
    }
    if(cap < 2) {
        cap = 2;
    }
    vec->len = len;
    vec->cap = cap;
    vec->data = malloc(cap * sizeof *vec->data);
    pti_CheckOSError(!vec->data, "NnzIdxVec New");
    memset(vec->data, 0, cap * sizeof *vec->data);
    return 0;
}

/**
 * Fill an existed dense long nnz index vector with a specified constant
 *
 * @param vec   a valid pointer to an existed ptiNnzIndexVector variable,
 * @param num   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int ptiConstantNnzIndexVector(ptiNnzIndexVector * const vec, ptiNnzIndex const num) {
    for(ptiNnzIndex i=0; i<vec->len; ++i)
        vec->data[i] = num;
    return 0;
}

/**
 * Copy a long nnz index vector to an uninitialized long nnz index vector
 *
 * @param dest a pointer to an uninitialized long nnz index vector
 * @param src  a pointer to an existing valid long nnz index vector
 *
 * The contents of `src` will be copied to `dest`.
 */
int ptiCopyNnzIndexVector(ptiNnzIndexVector *dest, const ptiNnzIndexVector *src) {
    int result = ptiNewNnzIndexVector(dest, src->len, src->len);
    pti_CheckError(result, "NnzIdxVec Copy", NULL);
    memcpy(dest->data, src->data, src->len * sizeof *src->data);
    return 0;
}

/**
 * Add a value to the end of a ptiNnzIndexVector
 *
 * @param vec   a pointer to a valid long nnz index vector
 * @param value the value to be appended
 *
 * The length of the long nnz index vector will be changed to contain the new value.
 */
int ptiAppendNnzIndexVector(ptiNnzIndexVector *vec, ptiNnzIndex const value) {
    if(vec->cap <= vec->len) {
#ifndef MEMCHECK_MODE
        ptiNnzIndex newcap = vec->cap + vec->cap/2;
#else
        ptiNnzIndex newcap = vec->len+1;
#endif
        ptiNnzIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "NnzIdxVec Append");
        vec->cap = newcap;
        vec->data = newdata;
    }
    vec->data[vec->len] = value;
    ++vec->len;
    return 0;
}

/**
 * Add a value to the end of a long nnz index vector
 *
 * @param vec        a pointer to a valid long nnz index vector
 * @param append_vec a pointer to another long nnz index vector, containing the values to be appended
 *
 * The values from `append_vec` will be appended to `vec`.
 */
int ptiAppendNnzIndexVectorWithVector(ptiNnzIndexVector *vec, const ptiNnzIndexVector *append_vec) {
    ptiNnzIndex newlen = vec->len + append_vec->len;
    if(vec->cap <= newlen) {
        ptiNnzIndex newcap = vec->cap + append_vec->cap;
        ptiNnzIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "NnzIdxVec Append NnzIdxVec");
        vec->cap = newcap;
        vec->data = newdata;
    }
    for(ptiNnzIndex i=0; i<append_vec->len; ++i) {
        vec->data[vec->len + i] = append_vec->data[i];
    }
    vec->len = newlen;

    return 0;
}

/**
 * Resize a long nnz index vector
 *
 * @param vec  the long nnz index vector to resize
 * @param size the new size of the long nnz index vector
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int ptiResizeNnzIndexVector(ptiNnzIndexVector *vec, ptiNnzIndex const size) {
    ptiNnzIndex newcap = size < 2 ? 2 : size;
    if(newcap != vec->cap) {
        ptiNnzIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
        pti_CheckOSError(!newdata, "NnzIdxVec Resize");
        vec->len = size;
        vec->cap = newcap;
        vec->data = newdata;
    } else {
        vec->len = size;
    }
    return 0;
}

/**
 * Release the memory buffer a ptiNnzIndexVector is holding
 *
 * @param vec a pointer to a valid long nnz vector
 *
 */
void ptiFreeNnzIndexVector(ptiNnzIndexVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}

