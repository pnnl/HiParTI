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

#ifndef HIPARTI_STRUCTS_H
#define HIPARTI_STRUCTS_H



/**
 * Dense dynamic array of specified type of scalars
 */
typedef struct {
    ptiNnzIndex    len;   /// length
    ptiNnzIndex    cap;   /// capacity
    ptiValue    *data; /// data
} ptiValueVector;

/**
 * Dense dynamic array of different types of integers
 */
typedef struct {
    ptiNnzIndex len;   /// length
    ptiNnzIndex cap;   /// capacity
    ptiIndex *data; /// data
} ptiIndexVector;

typedef struct {
    ptiNnzIndex len;   /// length
    ptiNnzIndex cap;   /// capacity
    ptiElementIndex *data; /// data
} ptiElementIndexVector;

typedef struct {
    ptiNnzIndex len;   /// length
    ptiNnzIndex cap;   /// capacity
    ptiBlockIndex *data; /// data
} ptiBlockIndexVector;

typedef struct {
    ptiNnzIndex len;   /// length
    ptiNnzIndex cap;   /// capacity
    ptiNnzIndex *data; /// data
} ptiNnzIndexVector;


/**
 * Dense matrix type
 */
typedef struct {
    ptiIndex nrows;   /// # rows
    ptiIndex ncols;   /// # columns
    ptiIndex cap;     /// # of allocated rows
    ptiIndex stride;  /// ncols rounded up to 8
    ptiValue *values; /// values, length cap*stride
} ptiMatrix;


/**
 * Dense matrix type, ncols = small rank (<= 256)
 */
typedef struct {
    ptiIndex    nrows;   /// # rows
    ptiElementIndex    ncols;   /// # columns, <= 256
    ptiIndex    cap;     /// # of allocated rows
    ptiElementIndex    stride;  /// ncols rounded up to 8, <= 256
    ptiValue *values; /// values, length cap*stride
} ptiRankMatrix;

/**
 * Sparse matrix type, COO format
 */
typedef struct {
    ptiIndex nrows;  /// # rows
    ptiIndex ncols;  /// # colums
    ptiNnzIndex nnz;    /// # non-zeros
    ptiIndexVector rowind; /// row indices, length nnz
    ptiIndexVector colind; /// column indices, length nnz
    ptiValueVector values; /// non-zero values, length nnz
} ptiSparseMatrix;


/**
 * Sparse matrix type, CSR format
 */
typedef struct {
    ptiIndex nrows;  /// # rows
    ptiIndex ncols;  /// # colums
    ptiNnzIndex nnz;    /// # non-zeros
    ptiNnzIndexVector rowptr; /// row indices, length nnz
    ptiIndexVector colind; /// column indices, length nnz
    ptiValueVector values; /// non-zero values, length nnz
} ptiSparseMatrixCSR;


/**
 * Sparse tensor type, Hierarchical COO format (HiCOO)
 */
typedef struct {
    /* Basic information */
    ptiIndex            nrows;  /// # rows
    ptiIndex            ncols;  /// # columns
    ptiNnzIndex         nnz;         /// # non-zeros

    /* Parameters */
    ptiElementIndex       sb_bits;         /// block size by nnz
    ptiElementIndex       sk_bits;         /// superblock size by nnz

    /* Index data arrays */
    ptiNnzIndexVector         bptr;      /// Block pointers to all nonzeros, nb = bptr.length - 1
    ptiBlockIndexVector       bindI;    /// Block indices for rows, length nb
    ptiBlockIndexVector       bindJ;    /// Block indices for columns, length nb
    ptiElementIndexVector     eindI;    /// Element indices within each block for rows, length nnz
    ptiElementIndexVector     eindJ;    /// Element indices within each block for columns, length nnz
    ptiValueVector            values;      /// non-zero values, length nnz

    /* Scheduling information */    /// TODO: move scheduler out of HiCOO format
    ptiNnzIndexVector         kptr;      /// Nonzero kernel pointers in 1-D array, indexing blocks. ptiIndexVector may be enough
    ptiIndexVector            *kschr;    /// Kernel scheduler
    ptiIndex                  nkiters;     /// max-length of iterations
} ptiSparseMatrixHiCOO;



/**
 * Sparse tensor type, COO format
 */
typedef struct {
    ptiIndex nmodes;      /// # modes
    ptiIndex * sortorder;  /// the order in which the indices are sorted
    ptiIndex * ndims;      /// size of each mode, length nmodes
    ptiNnzIndex nnz;         /// # non-zeros
    ptiIndexVector * inds;       /// indices of each element, length [nmodes][nnz]
    ptiValueVector values;      /// non-zero values, length nnz
} ptiSparseTensor;


/**
 * Sparse tensor type, Hierarchical COO format (HiCOO)
 */
typedef struct {
    /* Basic information */
    ptiIndex            nmodes;      /// # modes
    ptiIndex            *sortorder;  /// the order in which the indices are sorted
    ptiIndex            *ndims;      /// size of each mode, length nmodes
    ptiNnzIndex         nnz;         /// # non-zeros

    /* Parameters */
    ptiElementIndex       sb_bits;         /// block size by nnz
    ptiElementIndex       sk_bits;         /// kernel size by nnz
    ptiElementIndex       sc_bits;         /// chunk size by blocks

    /* Scheduling information */
    ptiNnzIndexVector         kptr;      /// Nonzero kernel pointers in 1-D array, indexing blocks. ptiIndexVector may be enough
    ptiIndexVector            **kschr;    /// Kernel scheduler
    ptiIndex                  *nkiters;     /// max-length of iterations
    ptiNnzIndexVector         cptr;      /// Chunk pointers to evenly split or combine blocks in a group, indexing blocks. ptiIndexVector may be enough

    /* Balanced scheduler */
    ptiIndexVector            **kschr_balanced;    /// Balanced kernel scheduler, nmodes * ndims / sk * even_nks
    ptiIndexVector            **kschr_balanced_pos;     /// indicators of partitions
    ptiIndex                  *nkpars;     /// max-length of partitions
    ptiIndexVector            *kschr_rest;    /// The rest imbalanced kernels
    ptiNnzIndexVector         knnzs;        /// Record the nnzs of each kernel

    /* Index data arrays */
    ptiNnzIndexVector         bptr;      /// Block pointers to all nonzeros
    ptiBlockIndexVector       *binds;    /// Block indices within each group
    ptiElementIndexVector     *einds;    /// Element indices within each block
    ptiValueVector            values;      /// non-zero values, length nnz
} ptiSparseTensorHiCOO;


/**
 * Sparse tensor type, extended Hierarchical COO format (ExHiCOO)
 */
typedef struct {
    /* Basic information */
    ptiIndex            nmodes;      /// # modes
    ptiIndex            *sortorder;  /// the order in which the indices are sorted
    ptiIndex            *ndims;      /// size of each mode, length nmodes
    ptiNnzIndex         nnz;         /// # non-zeros

    /* Parameters */
    ptiIndex              block_lvl;     /// blocking levels, could be 0
    ptiIndex              bit_length[MAX_BLOCK_LVLS + 1];
    ptiIndex              block_sizes[MAX_BLOCK_LVLS];     /// blocking sizes for multiple levels, not require block_sizes[l] > block_sizes[l+1]
    ptiIndex              sk;         /// kernel size by nnz

    /* Scheduling information */
    ptiNnzIndexVector         kptr;      /// Nonzero kernel pointers in 1-D array, indexing blocks. ptiIndexVector may be enough
    ptiIndexVector            **kschr;    /// Kernel scheduler
    ptiIndex                  *nkiters;     /// max-length of iterations

    /* Index data arrays */
    ptiNnzIndex               num_blocks[MAX_BLOCK_LVLS];
    ptiNnzIndex               *ptrs[MAX_BLOCK_LVLS];    /// blocking pointers, size num_blocks[l] + 1
    void                      **inds[MAX_BLOCK_LVLS + 1];   /// blocking indices
    ptiValueVector            values;      /// non-zero values, length nnz

} ptiSparseTensorHiCOOExt;


/**
 * Semi-sparse tensor type
 * The chosen mode is dense, while other modes are sparse.
 * Can be considered as "sparse tensor of dense fiber".
 * The "fiber" here can be defined as a vector of elements that have indices
 * only different in the last mode.
 */
typedef struct {
    ptiIndex nmodes; /// # Modes, must >= 2
    ptiIndex *ndims; /// size of each mode, length nmodes
    ptiIndex mode;   /// the mode where data is stored in dense format
    ptiNnzIndex nnz;    /// # non-zero fibers
    ptiIndexVector *inds;  /// indices of each dense fiber, length [nmodes][nnz], the mode-th value is ignored
    ptiIndex stride; /// ndims[mode] rounded up to 8
    ptiMatrix     values; /// dense fibers, size nnz*ndims[mode]
} ptiSemiSparseTensor;


/**
 * General Semi-sparse tensor type
 */
typedef struct {
    ptiIndex nmodes; /// # Modes, must >= 2
    ptiIndex *ndims; /// size of each mode, length nmodes
    ptiIndex ndmodes;
    ptiIndex *dmodes;   /// the mode where data is stored in dense format, allocate nmodes sized space
    ptiNnzIndex nnz;    /// # non-zero fibers
    ptiIndexVector *inds;  /// indices of each dense fiber, length [nmodes][nnz], the mode-th value is ignored
    ptiIndex *strides; /// ndims[mode] rounded up to 8
    ptiMatrix     values; /// dense fibers, size nnz*ndims[mode]
} ptiSemiSparseTensorGeneral;


/**
 * Kruskal tensor type, for CP decomposition result
 */
typedef struct {
  ptiIndex nmodes;
  ptiIndex rank;
  ptiIndex * ndims;
  ptiValue * lambda;
  double fit;
  ptiMatrix ** factors;
} ptiKruskalTensor;


/**
 * Kruskal tensor type, for CP decomposition result. 
 * ncols = small rank (<= 256)
 */
typedef struct {
  ptiIndex nmodes;
  ptiElementIndex rank;
  ptiIndex * ndims;
  ptiValue * lambda;
  double fit;
  ptiRankMatrix ** factors;
} ptiRankKruskalTensor;

/**
 * Key-value pair structure
 */
typedef struct 
{
  ptiIndex key;
  ptiIndex value;
} ptiKeyValuePair;

#ifdef HIPARTI_USE_OPENMP
/**
 * OpenMP lock pool.
 */
typedef struct
{
  bool initialized;
  ptiIndex nlocks;
  ptiIndex padsize;
  omp_lock_t * locks;
} ptiMutexPool;
#else
typedef struct ptiMutexPool ptiMutexPool;
#endif

/**
* @brief This struct is written to the beginning of any binary tensor file
*        written by SPLATT.
*/
typedef struct
{
  int32_t magic;
  uint64_t idx_width;
  uint64_t val_width;
} bin_header;

#endif
