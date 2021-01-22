#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "HiParTI.h"

/*function declarations*/
static void ptiLexiOrderPerMode(ptiSparseMatrix * mtx, ptiIndex mode, ptiIndex ** orgIds, int tk);

static double u_seconds(void)
{
    struct timeval tp;
    
    gettimeofday(&tp, NULL);
    
    return (double) tp.tv_sec + (double) tp.tv_usec / 1000000.0;
    
};

void ptiIndexRelabel(ptiSparseMatrix * mtx, ptiIndex ** newIndices, int renumber, ptiIndex iterations, int tk)
{
    /*
     newIndices is of size [nmodes][ndims[modes]] and assumed to be allocted.
     It will be overwritten. No need to initialize.
     
     We will need to reshuffle nonzeros. In order to not to touch mtx, we copy the indices of nonzeros
     to a local variable coords. This is sort of transposed wrt mtx: its size is nnz * n, instead of n * nnz used in mtx.
     */
    ptiIndex const nmodes = 2;  // for matrices
    // ptiNnzIndex const nnz = mtx->nnz;

    ptiIndex i, m;
    ptiIndex its;

    if (renumber == 1) {    /* Lexi-order renumbering */
        printf("[Lexi-order]\n");
        /* copy the indices */
        ptiSparseMatrix mtx_temp;
        ptiCopySparseMatrix(&mtx_temp, mtx, tk);

        ptiIndex ** orgIds = (ptiIndex **) malloc(sizeof(ptiIndex*) * nmodes);

        orgIds[0] = (ptiIndex *) malloc(sizeof(ptiIndex) * mtx->nrows);
        for (i = 0; i < mtx->nrows; i++)
            orgIds[0][i] = i;
        orgIds[1] = (ptiIndex *) malloc(sizeof(ptiIndex) * mtx->ncols);
        for (i = 0; i < mtx->ncols; i++)
            orgIds[1][i] = i;

        for (its = 0; its < iterations; its++)
        {
            printf("[Lexi-order] Optimizing the numbering for its %u\n", its+1);
            ptiLexiOrderPerMode(&mtx_temp, 0, orgIds, tk);
            // ptiDumpIndexArray(orgIds[0], mtx->nrows, stdout);

            ptiLexiOrderPerMode(&mtx_temp, 1, orgIds, tk);
            // ptiDumpIndexArray(orgIds[1], mtx->ncols, stdout);
        }

        /* compute newIndices from orgIds. Reverse perm */
        for (i = 0; i < mtx->nrows; i++)
            newIndices[0][orgIds[0][i]] = i;
        for (i = 0; i < mtx->ncols; i++)
            newIndices[1][orgIds[1][i]] = i;

        ptiFreeSparseMatrix(&mtx_temp);
        for (m = 0; m < nmodes; m++)
            free(orgIds[m]);
        free(orgIds);

    } else if (renumber == 2 ) {    /* BFS-like renumbering */
       /*
        REMARK (10 May 2018): this is the old bfs-like kind of thing. I hoped it would reduce the number of iterations,
        but on a few cases it did not help much. Just leaving it in case we want to use it.
        */
        printf("[BFS-like]\n");
        // ptiBFSLike(tsr, newIndices);
    }    
    
}


// void ptiBFSLike(ptiSparseMatrix * mtx, ptiIndex ** newIndices)
// {
//     /*PRE: newIndices is allocated
     
//      POST:
//      newIndices[0][0...n_0-1] gives the new ids for dim 0
//      newIndices[1][0...n_1-1] gives the new ids for dim 1
//      ...
//      newIndices[d-1][0...n_{d-1}-1] gives the new ids for dim d-1
     
//      This implements a simple idea close to BFS/Cuthill-McKee/Maximum cardinality search.
//      */
//     ptiIndex const nmodes = 2;
//     ptiNnzIndex const nnz = mtx->nnz;
//     ptiIndex * ndims = tsr->ndims;
//     ptiIndexVector * inds = tsr->inds;
    
//     ptiIndex *dimsPrefixSum;
//     basicHypergraph hg;
//     ptiIndex *newIndicesHg;
//     ptiIndex d, i;
    
//     dimsPrefixSum = (ptiIndex*) calloc(nmodes, sizeof(ptiIndex));
//     for (d = 1; d < nmodes; d++)
//         dimsPrefixSum[d] = ndims[d-1] + dimsPrefixSum[d-1];
    
//     fillHypergraphFromCoo(&hg, nmodes,  nnz, ndims, inds);

//     newIndicesHg = (ptiIndex*) malloc(sizeof(ptiIndex) * hg.nvrt);
//     for (i = 0; i < hg.nvrt; i++)
//         newIndicesHg[i] = i;
    
//     for (d = 0; d < nmodes; d++) /*order d*/
//         orderforHiCOOaDim(&hg, newIndicesHg, dimsPrefixSum[d], dimsPrefixSum[d] + ndims[d]-1);
    
//     /*copy from newIndices to newIndicesOut*/
//     for (d = 0; d < nmodes; d++)
//         for (i = 0; i < ndims[d]; i++)
//             newIndices[d][i] = newIndicesHg[dimsPrefixSum[d] + i] - dimsPrefixSum[d];
    
//     free(newIndicesHg);
//     freeHypergraphData(&hg);
//     free(dimsPrefixSum);
    
// }


static void lexOrderThem( ptiNnzIndex m, ptiIndex n, ptiNnzIndex *ia, ptiIndex *cols, ptiIndex *cprm, int const tk)
{
    /*m, n are the num of rows and cols, respectively. We lex order cols,
     given rows.
     
     BU notes as of 4 May 2018: I am hoping that I will not be asked the details of this function, and its memory use;) A quick and dirty update from something else I had since some time. I did not think through if the arrays could be reduced. Right now we have 10 arrays of size n each (where n is the length of a single dimension of the tensor.
     */
    
    ptiNnzIndex *flag, j, jcol, jend;
    ptiIndex *svar,  *var, numBlocks;
    ptiIndex *prev, *next, *sz, *setnext, *setprev, *tailset;
    
    ptiIndex *freeIdList, freeIdTop;
    
    ptiIndex k, s, acol;
    
    ptiIndex firstset, set, pos;
    
    svar = (ptiIndex*) calloc(sizeof(ptiIndex),(n+2));
    flag = (ptiNnzIndex*) calloc(sizeof(ptiNnzIndex),(n+2));
    var  = (ptiIndex*) calloc(sizeof(ptiIndex),(n+2));
    prev = (ptiIndex*) calloc(sizeof(ptiIndex),(n+2));
    next = (ptiIndex*) calloc(sizeof(ptiIndex),(n+2));
    sz   = (ptiIndex*) calloc(sizeof(ptiIndex),(n+2));
    setprev = (ptiIndex*)calloc(sizeof(ptiIndex),(n+2));
    setnext = (ptiIndex*)calloc(sizeof(ptiIndex),(n+2));
    tailset = (ptiIndex*)calloc(sizeof(ptiIndex),(n+2));
    freeIdList = (ptiIndex*)calloc(sizeof(ptiIndex),(n+2));
    
    next[1] = 2;
    prev[0] =  prev[1] = 0;
    next[n] = 0;
    prev[n] = n-1;
    svar[1] = svar[n] = 1;
    flag[1] = flag[n] = flag[n+1] = 0;
    cprm[1] = cprm[n] = 2 * n ;
    setprev[1] = setnext[1] = 0;
    // #pragma omp parallel for num_threads(tk)
    for(ptiIndex jj = 2; jj<=n-1; jj++)/*init all in a single svar*/
    {
        svar[jj] = 1;
        next[jj] = jj+1;
        prev[jj] = jj-1;
        flag[jj] = 0;
        sz[jj] = 0;
        setprev[jj] = setnext[jj] = 0;
        cprm[jj] = 2 * n;
    }
    var[1] = 1;
    sz[1] = n;
    sz[n] = sz[n+1] =  0;
    
    setprev[n] = setnext[n] = 0;
    setprev[n+1] = setnext[n+1] = 0;
    
    tailset[1] = n;
    
    firstset = 1;
    freeIdList[0] = 0;
    
    // #pragma omp parallel for num_threads(tk)
    for(ptiIndex jj= 1; jj<=n; jj++)
        freeIdList[jj] = jj+1;/*1 is used as a set id*/
    
    freeIdTop = 1;
    for(j=1; j<=m; j++)
    {
        jend = ia[j+1]-1;
        for(jcol = ia[j]; jcol <= jend ; jcol++)
        {
            acol= cols[jcol];
            s = svar[acol];
            if( flag[s] < j)/*first occurence of supervar s in j*/
            {
                flag[s] = j;
                if(sz[s] == 1 && tailset[s] != acol)
                {
                    printf("this should not happen (sz 1 but tailset not ok)\n");
                    exit(12);
                }
                if(sz[s] > 1)
                {
                    ptiIndex newId;
                    /*remove acol from s*/
                    if(tailset[s] == acol) tailset[s] = prev[acol];
                    
                    next[prev[acol]] = next[acol];
                    prev[next[acol]] = prev[acol];
                    
                    sz[s] = sz[s] - 1;
                    /*create a new supervar ns=newId
                     and make i=acol its only var*/
                    if(freeIdTop == n+1) {
                        printf("this should not happen (no index)\n");
                        exit(12);
                    }
                    newId = freeIdList[freeIdTop++];
                    svar[acol] = newId;
                    var[newId] = acol;
                    flag[newId] = j;
                    sz[newId ] = 1;
                    next[acol] = 0;
                    prev[acol] = 0;
                    var[s] = acol;
                    tailset[newId] = acol;
                    
                    setnext[newId] = s;
                    setprev[newId] = setprev[s];
                    if(setprev[s])
                        setnext[setprev[s]] = newId;
                    setprev[s] = newId;
                    
                    if(firstset == s)
                        firstset = newId;
                    
                }
            }
            else/*second or later occurence of s for row j*/
            {
                k = var[s];
                svar[acol] = svar[k];
                
                /*remove acol from its current chain*/
                if(tailset[s] == acol) tailset[s] = prev[acol];
                
                next[prev[acol]] = next[acol];
                prev[next[acol]] = prev[acol];
                
                sz[s] = sz[s] - 1;
                if(sz[s] == 0)/*s is a free id now..*/
                {
                    
                    freeIdList[--freeIdTop] = s; /*add s to the free id list*/
                    
                    if(setnext[s])
                        setprev[setnext[s]] = setprev[s];
                    if(setprev[s])
                        setnext[setprev[s]] = setnext[s];
                    
                    setprev[s] = setnext[s] = 0;
                    tailset[s] = 0;
                    var[s] = 0;
                    flag[s] = 0;
                }
                /*add to chain containing k (as the last element)*/
                prev[acol] = tailset[svar[k]];
                next[acol]  = 0;/*BU next[tailset[svar[k]]];*/
                next[tailset[svar[k]]] = acol;
                tailset[svar[k]] = acol;
                sz[svar[k]] = sz[svar[k]] + 1;
            }
        }
    }
    
    pos = 1;
    numBlocks = 0;
    for(set = firstset; set != 0; set = setnext[set])
    {
        ptiIndex item = tailset[set];
        ptiIndex headset = 0;
        numBlocks ++;
        
        while(item != 0 )
        {
            headset = item;
            item = prev[item];
        }
        /*located the head of the set. output them (this is for keeping the initial order*/
        while(headset)
        {
            cprm[pos++] = headset;
            headset = next[headset];
        }
    }
    
    free(tailset);
    free(sz);
    free(next);
    free(prev);
    free(var);
    free(flag);
    free(svar);
    free(setnext);
    free(setprev);
    if(pos-1 != n){
        printf("**************** Error ***********\n");
        printf("something went wrong and we could not order everyone\n");
        exit(12);
    }
    
    return ;
}


/**************************************************************/
static void ptiLexiOrderPerMode(ptiSparseMatrix * mtx, ptiIndex mode, ptiIndex ** orgIds, int tk)
{
    ptiNnzIndex const nnz = mtx->nnz;
    // ptiIndex const nmodes = 2;  // for matrices
    ptiIndex mode_dim;
    ptiIndexVector * mode_ind;
    if (mode == 0) {
        mode_dim = mtx->nrows;
        mode_ind = &(mtx->rowind);
    }
    else if (mode == 1) {
        mode_dim = mtx->ncols;
        mode_ind = &(mtx->colind);
    }

    ptiNnzIndex * rowPtrs = NULL;
    ptiIndex * colIds = NULL;
    ptiIndex * cprm = NULL, * invcprm = NULL, * saveOrgIds = NULL;
    ptiNnzIndex atRowPlus1, mtxNrows, mtrxNnz;

    ptiIndex c;
    ptiNnzIndex z;
    double t1, t0;

    t0 = u_seconds();
    ptiIndex sort_mode = 0;
    /* reverse to get the sort_mode */
    if (mode == 0) sort_mode = 1;
    else if (mode == 1) sort_mode = 0;
    ptiSparseMatrixSortIndexSingleMode(mtx, 1, sort_mode, tk);
    t1 = u_seconds()-t0;
    printf("mode %u, sort time %.2f\n", mode, t1); fflush(stdout);

    /* we matricize this (others x thisDim), whose columns will be renumbered */
    /* on the matrix all arrays are from 1, and all indices are from 1. */
    
    rowPtrs = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * (nnz + 2)); /*large space*/
    colIds = (ptiIndex *) malloc(sizeof(ptiIndex) * (nnz + 2)); /*large space*/
    
    if(rowPtrs == NULL || colIds == NULL)
    {
        printf("could not allocate.exiting \n");
        exit(12);
    }
    
    rowPtrs[0] = 0; /* we should not access this, that is why. */
    rowPtrs [1] = 1;
    colIds[1] = mode_ind->data[0] + 1;
    atRowPlus1 = 2;
    mtrxNnz = 2;/* start filling from the second element */
    
    t0 = u_seconds();
    for (z = 1; z < nnz; z++)
    {
        int cmp_res = pti_SparseMatrixCompareIndicesSingleMode(mtx, z, mtx, z-1, sort_mode);
        if(cmp_res != 0)
            rowPtrs[atRowPlus1++] = mtrxNnz; /* close the previous row and start a new one. */
        
        colIds[mtrxNnz ++] = mode_ind->data[z] + 1;
    }
    rowPtrs[atRowPlus1] = mtrxNnz;
    mtxNrows = atRowPlus1-1;
    t1 =u_seconds()-t0;
    printf("mode %u, create time %.2f\n", mode, t1); fflush(stdout);
    
    rowPtrs = realloc(rowPtrs, (sizeof(ptiNnzIndex) * (mtxNrows + 2)));
    cprm = (ptiIndex *) malloc(sizeof(ptiIndex) * (mode_dim + 1));
    invcprm = (ptiIndex *) malloc(sizeof(ptiIndex) * (mode_dim + 1));
    saveOrgIds = (ptiIndex *) malloc(sizeof(ptiIndex) * (mode_dim + 1));

    // printf("rowPtrs: \n");
    // ptiDumpNnzIndexArray(rowPtrs, mtxNrows + 2, stdout);
    // printf("colIds: \n");
    // ptiDumpIndexArray(colIds, nnz + 2, stdout);
    
    t0 = u_seconds();
    lexOrderThem(mtxNrows, mode_dim, rowPtrs, colIds, cprm, tk);
    t1 =u_seconds()-t0;
    printf("mode %u, lexorder time %.2f\n", mode, t1); fflush(stdout);
    // printf("cprm: \n");
    // ptiDumpIndexArray(cprm, mode_dim + 1, stdout);

    /* update orgIds and modify coords */
    for (c=0; c < mode_dim; c++)
    {
        invcprm[cprm[c+1]-1] = c;
        saveOrgIds[c] = orgIds[mode][c];
    }
    for (c=0; c < mode_dim; c++)
        orgIds[mode][c] = saveOrgIds[cprm[c+1]-1];

    // printf("invcprm: \n");
    // ptiDumpIndexArray(invcprm, mode_dim + 1, stdout);
    
    /* rename the dim component of nonzeros */
    for (z = 0; z < nnz; z++)
        mode_ind->data[z] = invcprm[mode_ind->data[z]];
    // ptiAssert(ptiDumpSparseMatrix(mtx, 0, stdout) == 0);
    
    free(saveOrgIds);
    free(invcprm);
    free(cprm);
    free(colIds);
    free(rowPtrs);
}

