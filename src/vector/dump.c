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
 * Dump a dense ptiElementIndexVector to file
 *
 * @param vec a pointer to a valid ptiElementIndexVector
 * @param fp a file pointer
 *
 */
int ptiDumpElementIndexVector(ptiElementIndexVector *vec, FILE *fp) {
    int iores;
    ptiNnzIndex len = vec->len;
    iores = fprintf(fp, "ptiElementIndexVector length: %"HIPARTI_PRI_NNZ_INDEX "\n", len);
    pti_CheckOSError(iores < 0, "EleIdxVec Dump");
    for(ptiNnzIndex i=0; i < len; ++i) {
        iores = fprintf(fp, "%"HIPARTI_PRI_ELEMENT_INDEX "\t", vec->data[i]);
        pti_CheckOSError(iores < 0, "EleIdxVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense ptiIndexVector to file
 *
 * @param vec a pointer to a valid ptiIndexVector
 * @param fp a file pointer
 *
 */
int ptiDumpIndexVector(ptiIndexVector *vec, FILE *fp) {
    int iores;
    ptiNnzIndex len = vec->len;
    iores = fprintf(fp, "ptiIndexVector length: %"HIPARTI_PRI_NNZ_INDEX "\n", len);
    pti_CheckOSError(iores < 0, "IdxVec Dump");
    for(ptiNnzIndex i=0; i < len; ++i) {
        iores = fprintf(fp, "%"HIPARTI_PRI_INDEX "\t", vec->data[i]);
        pti_CheckOSError(iores < 0, "IdxVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense ptiBlockIndexVector to file
 *
 * @param vec a pointer to a valid ptiBlockIndexVector
 * @param fp a file pointer
 *
 */
int ptiDumpBlockIndexVector(ptiBlockIndexVector *vec, FILE *fp) {
    int iores;
    ptiNnzIndex len = vec->len;
    iores = fprintf(fp, "ptiBlockIndexVector length: %"HIPARTI_PRI_NNZ_INDEX "\n", len);
    pti_CheckOSError(iores < 0, "BlkIdxVec Dump");
    for(ptiNnzIndex i=0; i < len; ++i) {
        iores = fprintf(fp, "%"HIPARTI_PRI_BLOCK_INDEX "\t", vec->data[i]);
        pti_CheckOSError(iores < 0, "BlkIdxVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense ptiNnzIndexVector to file
 *
 * @param vec a pointer to a valid ptiNnzIndexVector
 * @param fp a file pointer
 *
 */
int ptiDumpNnzIndexVector(ptiNnzIndexVector *vec, FILE *fp) {
    int iores;
    ptiNnzIndex len = vec->len;
    iores = fprintf(fp, "ptiNnzIndexVector length: %"HIPARTI_PRI_NNZ_INDEX "\n", len);
    pti_CheckOSError(iores < 0, "NnzIdxVec Dump");
    for(ptiNnzIndex i=0; i < len; ++i) {
        iores = fprintf(fp, "%"HIPARTI_PRI_NNZ_INDEX "\t", vec->data[i]);
        pti_CheckOSError(iores < 0, "NnzIdxVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense ptiValueVector to file
 *
 * @param vec a pointer to a valid ptiValueVector
 * @param fp a file pointer
 *
 */
int ptiDumpValueVector(ptiValueVector *vec, FILE *fp) {
    int iores;
    ptiNnzIndex len = vec->len;
    iores = fprintf(fp, "ptiValueVector length: %"HIPARTI_PRI_NNZ_INDEX "\n", len);
    pti_CheckOSError(iores < 0, "ValVec Dump");
    for(ptiNnzIndex i=0; i < len; ++i) {
        iores = fprintf(fp, "%"HIPARTI_PRI_VALUE "\t", vec->data[i]);
        pti_CheckOSError(iores < 0, "ValVec Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense ptiIndex array to file
 *
 * @param array a pointer to a valid ptiIndex array
 * @param size of the array
 * @param fp a file pointer
 *
 */
int ptiDumpIndexArray(ptiIndex const *array, ptiNnzIndex const n, FILE *fp) {
    int iores;
    iores = fprintf(fp, "ptiIndex array length: %"HIPARTI_PRI_NNZ_INDEX "\n", n);
    pti_CheckOSError(iores < 0, "IdxArray Dump");
    for(ptiNnzIndex i=0; i < n; ++i) {
        iores = fprintf(fp, "%"HIPARTI_PRI_INDEX "\t", array[i]);
        pti_CheckOSError(iores < 0, "IdxArray Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}


/**
 * Dump a dense ptiNnzIndex array to file
 *
 * @param array a pointer to a valid ptiNnzIndex array
 * @param size of the array
 * @param fp a file pointer
 *
 */
int ptiDumpNnzIndexArray(ptiNnzIndex const *array, ptiNnzIndex const n, FILE *fp) {
    int iores;
    iores = fprintf(fp, "ptiNnzIndex array length: %"HIPARTI_PRI_NNZ_INDEX "\n", n);
    pti_CheckOSError(iores < 0, "NnzIdxArray Dump");
    for(ptiNnzIndex i=0; i < n; ++i) {
        iores = fprintf(fp, "%"HIPARTI_PRI_NNZ_INDEX "\t", array[i]);
        pti_CheckOSError(iores < 0, "NnzIdxArray Dump");
    }
    iores = fprintf(fp, "\n");

    return 0;
}
