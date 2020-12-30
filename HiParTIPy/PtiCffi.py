from cffi import FFI

pti = FFI()
PTI = pti.dlopen('./build/libHiParTI.so')
print("Loaded lib {}".format(PTI))

pti.cdef("""
typedef struct 
{
 short level ;
 short token ;
 short bsize ;
 char fd ;
 unsigned flags ;
 unsigned char hold ;
 unsigned char *buffer ;
 unsigned char * curp ;
 unsigned istemp; 
}FILE ;
int printf(const char *format, ...);
FILE *fopen(const char *file_name, const char *mode_of_operation);
void *malloc(size_t size);

typedef uint32_t ptiIndex;
typedef uint32_t ptiBlockIndex;
typedef float ptiValue;
typedef uint8_t ptiElementIndex;
typedef uint16_t ptiBlockMatrixIndex;
typedef ptiBlockIndex ptiBlockNnzIndex;
typedef uint64_t ptiNnzIndex;

typedef struct {
    ptiNnzIndex    len;   /// length
    ptiNnzIndex    cap;   /// capacity
    ptiValue    *data; /// data
} ptiValueVector;
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

typedef struct {
    ptiIndex nrows;   /// # rows
    ptiIndex ncols;   /// # columns
    ptiIndex cap;     /// # of allocated rows
    ptiIndex stride;  /// ncols rounded up to 8
    ptiValue *values; /// values, length cap*stride
} ptiMatrix;
typedef struct {
    ptiIndex    nrows;   /// # rows
    ptiElementIndex    ncols;   /// # columns, <= 256
    ptiIndex    cap;     /// # of allocated rows
    ptiElementIndex    stride;  /// ncols rounded up to 8, <= 256
    ptiValue *values; /// values, length cap*stride
} ptiRankMatrix;
typedef struct {
    ptiIndex nrows;  /// # rows
    ptiIndex ncols;  /// # colums
    ptiNnzIndex nnz;    /// # non-zeros
    ptiIndexVector rowind; /// row indices, length nnz
    ptiIndexVector colind; /// column indices, length nnz
    ptiValueVector values; /// non-zero values, length nnz
} ptiSparseMatrix;
typedef struct {
    ptiIndex nrows;  /// # rows
    ptiIndex ncols;  /// # colums
    ptiNnzIndex nnz;    /// # non-zeros
    ptiNnzIndexVector rowptr; /// row indices, length nnz
    ptiIndexVector colind; /// column indices, length nnz
    ptiValueVector values; /// non-zero values, length nnz
} ptiSparseMatrixCSR;
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

typedef struct {
    ptiIndex nmodes;      /// # modes
    ptiIndex * sortorder;  /// the order in which the indices are sorted
    ptiIndex * ndims;      /// size of each mode, length nmodes
    ptiNnzIndex nnz;         /// # non-zeros
    ptiIndexVector * inds;       /// indices of each element, length [nmodes][nnz]
    ptiValueVector values;      /// non-zero values, length nnz
} ptiSparseTensor;

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
typedef struct {
    ptiIndex nmodes; /// # Modes, must >= 2
    ptiIndex *ndims; /// size of each mode, length nmodes
    ptiIndex mode;   /// the mode where data is stored in dense format
    ptiNnzIndex nnz;    /// # non-zero fibers
    ptiIndexVector *inds;  /// indices of each dense fiber, length [nmodes][nnz], the mode-th value is ignored
    ptiIndex stride; /// ndims[mode] rounded up to 8
    ptiMatrix     values; /// dense fibers, size nnz*ndims[mode]
} ptiSemiSparseTensor;
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
typedef struct {
  ptiIndex nmodes;
  ptiIndex rank;
  ptiIndex * ndims;
  ptiValue * lambda;
  double fit;
  ptiMatrix ** factors;
} ptiKruskalTensor;
typedef struct {
  ptiIndex nmodes;
  ptiElementIndex rank;
  ptiIndex * ndims;
  ptiValue * lambda;
  double fit;
  ptiRankMatrix ** factors;
} ptiRankKruskalTensor;


int ptiMakeVectorBuff(ptiValueVector * ybufs, ptiIndex nrows);
int ptiFreeVecBuff(ptiValueVector * ybufs);

int ptiLoadSparseMatrix(ptiSparseMatrix *mtx, ptiIndex start_index, FILE *fid);
int ptiDumpSparseMatrix(ptiSparseMatrix *mtx, ptiIndex start_index, FILE *fid);
void ptiSparseMatrixStatus(ptiSparseMatrix *mtx, FILE *fp);
int ptiSparseMatrixMulVector(ptiValueVector * y, const ptiSparseMatrix *mtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVector(ptiValueVector * y, const ptiSparseMatrix *mtx, ptiValueVector * x) ;
int ptiOmpSparseMatrixMulVectorReduce(ptiValueVector * y, const ptiSparseMatrix *mtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVector_Reduce(ptiValueVector * y, ptiValueVector * ybufs, const ptiSparseMatrix *mtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVectorHiCOO_Schedule_Reduce(ptiValueVector * y, ptiValueVector * ybufs, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x);
int ptiSparseMatrixMulMatrix(ptiMatrix * C, const ptiSparseMatrix *spA, ptiMatrix * B);
int ptiOmpSparseMatrixMulMatrix(ptiMatrix * C, const ptiSparseMatrix *spA, ptiMatrix * B);
int ptiOmpSparseMatrixMulMatrixReduce(ptiMatrix * C, const ptiSparseMatrix *spA, ptiMatrix * B);

void ptiSparseMatrixSortIndexMorton(ptiSparseMatrix *mtx, int force,ptiNnzIndex begin, ptiNnzIndex end, ptiElementIndex sb_bits);
void ptiSparseMatrixSortIndexSingleMode(ptiSparseMatrix *mtx, int force, ptiIndex mode, int tk) ;
void ptiSparseMatrixSortIndexRowBlock( ptiSparseMatrix *mtx,  int force, ptiNnzIndex begin, ptiNnzIndex end, ptiElementIndex sk_bits);
void ptiIndexRelabel(ptiSparseMatrix * mtx, ptiIndex ** newIndices, int renumber, ptiIndex iterations, int tk);
void ptiGetRandomShuffledIndicesMat(ptiSparseMatrix *mtx, ptiIndex ** map_inds);
void ptiSparseMatrixShuffleIndices(ptiSparseMatrix *mtx, ptiIndex ** map_inds);

int ptiDumpSparseMatrixHiCOO(ptiSparseMatrixHiCOO * const himtx, FILE *fp);
void ptiSparseMatrixStatusHiCOO(ptiSparseMatrixHiCOO *himtx, FILE *fp);
int ptiSparseMatrixToHiCOO(ptiSparseMatrixHiCOO *himtx, ptiNnzIndex *max_nnzb, ptiSparseMatrix *mtx,  const ptiElementIndex sb_bits, const ptiElementIndex sk_bits);
int ptiSparseMatrixMulVectorHiCOO(ptiValueVector * y, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x);
int ptiSparseMatrixMulMatrixHiCOO(ptiMatrix * C, const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B) ;
int ptiOmpSparseMatrixMulVectorHiCOO(ptiValueVector * y, const ptiSparseMatrixHiCOO *himtx, ptiValueVector * x) ;
int ptiOmpSparseMatrixMulVectorHiCOOReduce(ptiValueVector * y, const ptiSparseMatrixHiCOO *mtx, ptiValueVector * x) ;
int ptiOmpSparseMatrixMulMatrixHiCOO(ptiMatrix * C, const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B);
int ptiOmpSparseMatrixMulMatrixHiCOOReduce(ptiMatrix * C,  const ptiSparseMatrixHiCOO *himtx, ptiMatrix * B);


int ptiDumpSparseMatrixCSR(ptiSparseMatrixCSR * const csrmtx, FILE *fp);
void ptiSparseMatrixStatusCSR(ptiSparseMatrixCSR *csrmtx, FILE *fp);
int ptiSparseMatrixToCSR (ptiSparseMatrixCSR * csrmtx, ptiSparseMatrix * mtx);
int ptiSparseMatrixMulVectorCSR(ptiValueVector * y, ptiSparseMatrixCSR *csrmtx, ptiValueVector * x);
int ptiOmpSparseMatrixMulVectorCSR(ptiValueVector * y, const ptiSparseMatrixCSR *himtx, ptiValueVector * x) ;
int ptiOmpSparseMatrixMulVectorCSRReduce(ptiValueVector * y, const ptiSparseMatrixCSR *mtx, ptiValueVector * x) ;
int ptiOmpSparseMatrixMulVectorCSR_Reduce(ptiValueVector *y, ptiValueVector * ybufs, const ptiSparseMatrixCSR *csrmtx, ptiValueVector * x);
int ptiSparseMatrixMulMatrixCSR(ptiMatrix * C, ptiSparseMatrixCSR *csrmtx, ptiMatrix * B) ;
int ptiOmpSparseMatrixMulMatrixCSR(ptiMatrix * C, ptiSparseMatrixCSR *csrmtx, ptiMatrix * B);
int ptiOmpSparseMatrixMulMatrixCSRReduce(ptiMatrix * C,  const ptiSparseMatrixCSR *csrmtx, ptiMatrix * B);


int ptiNewValueVector(ptiValueVector *vec, ptiNnzIndex len, ptiNnzIndex cap);
int ptiDumpValueVector(ptiValueVector *vec, FILE *fp);
int ptiRandomValueVector(ptiValueVector * const vec);
void ptiFreeValueVector(ptiValueVector *vec);
int ptiConstantValueVector(ptiValueVector * const vec, ptiValue const val);

int ptiNewNnzIndexVector(ptiNnzIndexVector *vec, ptiNnzIndex len, ptiNnzIndex cap);
int ptiConstantNnzIndexVector(ptiNnzIndexVector * const vec, ptiNnzIndex const num);
void ptiQuickSortNnzIndexArray(ptiNnzIndex * array, ptiNnzIndex l, ptiNnzIndex r);

int ptiNewMatrix(ptiMatrix *mtx, ptiIndex const nrows, ptiIndex const ncols);
int ptiConstantMatrix(ptiMatrix *mtx, ptiValue const val);
int ptiDumpMatrix(ptiMatrix *mtx, FILE *fp);
int ptiResizeMatrix(ptiMatrix *mtx, ptiIndex const new_nrows);
int ptiRandomizeMatrix(ptiMatrix *mtx);
void ptiFreeMatrix(ptiMatrix *mtx);

int ptiLoadSparseTensor(ptiSparseTensor *tsr, ptiIndex start_index, char const * const fname);
int ptiDumpSparseTensor(const ptiSparseTensor *tsr, ptiIndex start_index, FILE *fp);
void ptiFreeSparseTensor(ptiSparseTensor *tsr);
int ptiCopySparseTensor(ptiSparseTensor *dest, const ptiSparseTensor *src, int const nt);
int ptiSparseTensorMulScalar(ptiSparseTensor *X, ptiValue const a);
int ptiSparseTensorDivScalar(ptiSparseTensor *X, ptiValue const a);
int ptiSparseTensorAdd(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiSparseTensorSub(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiSparseTensorAddOMP(ptiSparseTensor *Y, ptiSparseTensor *X, int const nthreads);
int ptiSparseTensorSubOMP(ptiSparseTensor *Y, ptiSparseTensor *X, int const nthreads);
int ptiSparseTensorDotMul(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiSparseTensorDotMulEq(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiOmpSparseTensorDotMulEq(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiCudaSparseTensorDotMulEq(ptiSparseTensor *Z, const ptiSparseTensor *X, const ptiSparseTensor *Y);
int ptiSparseTensorDotDiv(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor * const Y);
int ptiSparseTensorMulMatrix(ptiSemiSparseTensor *Y, ptiSparseTensor * const X, ptiMatrix *const U, ptiIndex mode);
int ptiOmpSparseTensorMulMatrix(ptiSemiSparseTensor *Y, ptiSparseTensor * const X, ptiMatrix * const U, ptiIndex mode);
int ptiCudaSparseTensorMulMatrix(ptiSemiSparseTensor *Y, ptiSparseTensor *X, const ptiMatrix *U, ptiIndex const mode);
int ptiCudaSparseTensorMulMatrixOneKernel(ptiSemiSparseTensor *Y, ptiSparseTensor *X, const ptiMatrix *U, ptiIndex const mode, ptiIndex const impl_num, ptiNnzIndex const smen_size);
int ptiSparseTensorMulVector(ptiSemiSparseTensor *Y, ptiSparseTensor * const X, ptiValueVector * const V, ptiIndex mode);
int ptiSparseTensorMulTensor(ptiSparseTensor *Z, ptiSparseTensor * const X, ptiSparseTensor *const Y, ptiIndex num_cmodes, ptiIndex * cmodes_X, ptiIndex * cmodes_Y);



""")
