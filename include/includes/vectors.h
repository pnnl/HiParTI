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

#ifndef HIPARTI_VECTORS_H
#define HIPARTI_VECTORS_H

/* Dense Array functions */
ptiNnzIndex ptiMaxNnzIndexArray(ptiNnzIndex const * const indices, ptiNnzIndex const size);
ptiIndex ptiMaxIndexArray(ptiIndex const * const indices, ptiNnzIndex const size);
void ptiPairArraySort(ptiKeyValuePair * kvarray, ptiIndex const length);
int ptiDumpIndexArray(ptiIndex const *array, ptiNnzIndex const n, FILE *fp);
int ptiDumpNnzIndexArray(ptiNnzIndex const *array, ptiNnzIndex const n, FILE *fp);
void ptiQuickSortNnzIndexArray(ptiNnzIndex * array, ptiNnzIndex l, ptiNnzIndex r);
long int ptiInArray(ptiIndex * array, ptiNnzIndex len, ptiIndex value);

/* Dense vector, with ptiValueVector type */
int ptiNewValueVector(ptiValueVector *vec, uint64_t len, uint64_t cap);
int ptiConstantValueVector(ptiValueVector * const vec, ptiValue const val);
int ptiRandomValueVector(ptiValueVector * const vec);
int ptiCopyValueVector(ptiValueVector *dest, const ptiValueVector *src, int const nt);
int ptiAppendValueVector(ptiValueVector *vec, ptiValue const value);
int ptiAppendValueVectorWithVector(ptiValueVector *vec, const ptiValueVector *append_vec);
int ptiResizeValueVector(ptiValueVector *vec, ptiNnzIndex const size);
void ptiFreeValueVector(ptiValueVector *vec);
int ptiDumpValueVector(ptiValueVector *vec, FILE *fp);

/* Dense vector, with ptiIndexVector type */
int ptiNewIndexVector(ptiIndexVector *vec, uint64_t len, uint64_t cap);
int ptiConstantIndexVector(ptiIndexVector * const vec, ptiIndex const num);
int ptiCopyIndexVector(ptiIndexVector *dest, const ptiIndexVector *src, int const nt);
int ptiAppendIndexVector(ptiIndexVector *vec, ptiIndex const value);
int ptiAppendIndexVectorWithVector(ptiIndexVector *vec, const ptiIndexVector *append_vec);
int ptiResizeIndexVector(ptiIndexVector *vec, ptiNnzIndex const size);
void ptiFreeIndexVector(ptiIndexVector *vec);
int ptiDumpIndexVector(ptiIndexVector *vec, FILE *fp);

/* Dense vector, with ptiElementIndexVector type */
int ptiNewElementIndexVector(ptiElementIndexVector *vec, uint64_t len, uint64_t cap);
int ptiConstantElementIndexVector(ptiElementIndexVector * const vec, ptiElementIndex const num);
int ptiCopyElementIndexVector(ptiElementIndexVector *dest, const ptiElementIndexVector *src);
int ptiAppendElementIndexVector(ptiElementIndexVector *vec, ptiElementIndex const value);
int ptiAppendElementIndexVectorWithVector(ptiElementIndexVector *vec, const ptiElementIndexVector *append_vec);
int ptiResizeElementIndexVector(ptiElementIndexVector *vec, ptiNnzIndex const size);
void ptiFreeElementIndexVector(ptiElementIndexVector *vec);
int ptiDumpElementIndexVector(ptiElementIndexVector *vec, FILE *fp);

/* Dense vector, with ptiBlockIndexVector type */
int ptiNewBlockIndexVector(ptiBlockIndexVector *vec, uint64_t len, uint64_t cap);
int ptiConstantBlockIndexVector(ptiBlockIndexVector * const vec, ptiBlockIndex const num);
int ptiCopyBlockIndexVector(ptiBlockIndexVector *dest, const ptiBlockIndexVector *src);
int ptiAppendBlockIndexVector(ptiBlockIndexVector *vec, ptiBlockIndex const value);
int ptiAppendBlockIndexVectorWithVector(ptiBlockIndexVector *vec, const ptiBlockIndexVector *append_vec);
int ptiResizeBlockIndexVector(ptiBlockIndexVector *vec, ptiNnzIndex const size);
void ptiFreeBlockIndexVector(ptiBlockIndexVector *vec);
int ptiDumpBlockIndexVector(ptiBlockIndexVector *vec, FILE *fp);

/* Dense vector, with ptiNnzIndexVector type */
int ptiNewNnzIndexVector(ptiNnzIndexVector *vec, uint64_t len, uint64_t cap);
int ptiConstantNnzIndexVector(ptiNnzIndexVector * const vec, ptiNnzIndex const num);
int ptiCopyNnzIndexVector(ptiNnzIndexVector *dest, const ptiNnzIndexVector *src);
int ptiAppendNnzIndexVector(ptiNnzIndexVector *vec, ptiNnzIndex const value);
int ptiAppendNnzIndexVectorWithVector(ptiNnzIndexVector *vec, const ptiNnzIndexVector *append_vec);
int ptiResizeNnzIndexVector(ptiNnzIndexVector *vec, ptiNnzIndex const size);
void ptiFreeNnzIndexVector(ptiNnzIndexVector *vec);
int ptiDumpNnzIndexVector(ptiNnzIndexVector *vec, FILE *fp);


#endif
