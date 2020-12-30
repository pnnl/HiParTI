#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "HiParTI.h"
#include "sptensor.h"
#include "renumber.h"

/*Interface to everything in this file is orderit(.., ..)*/

/*function declarations*/
static void ptiLexiOrderPerMode(ptiSparseTensor * tsr, ptiIndex mode, ptiIndex ** orgIds, ptiElementIndex sb_bits, int tk, int impl_num);
void ptiBFSLike(ptiSparseTensor * tsr, ptiIndex ** newIndices);

static double u_seconds(void)
{
    struct timeval tp;
    
    gettimeofday(&tp, NULL);
    
    return (double) tp.tv_sec + (double) tp.tv_usec / 1000000.0;
    
};

void ptiIndexRenumber(ptiSparseTensor * tsr, ptiIndex ** newIndices, int renumber, ptiIndex iterations, ptiElementIndex sb_bits, int tk, int impl_num)
{
    /*
     newIndices is of size [nmodes][ndims[modes]] and assumed to be allocted.
     It will be overwritten. No need to initialize.
     
     We will need to reshuffle nonzeros. In order to not to touch tsr, we copy the indices of nonzeros
     to a local variable coords. This is sort of transposed wrt tsr: its size is nnz * n, instead of n * nnz used in tsr.
     */
    ptiIndex const nmodes = tsr->nmodes;
    ptiNnzIndex const nnz = tsr->nnz;

    ptiIndex i, m;
    ptiNnzIndex z;
    ptiIndex its;

    if (renumber == 1) {    /* Lexi-order renumbering */
        /* copy the indices */
        ptiSparseTensor tsr_temp;
        ptiCopySparseTensor(&tsr_temp, tsr, tk);

        ptiIndex ** orgIds = (ptiIndex **) malloc(sizeof(ptiIndex*) * nmodes);

        for (m = 0; m < nmodes; m++)
        {
            orgIds[m] = (ptiIndex *) malloc(sizeof(ptiIndex) * tsr->ndims[m]);
            // #pragma omp parallel for num_threads(tk) private(i)
            for (i = 0; i < tsr->ndims[m]; i++)
                orgIds[m][i] = i;
        }

        // FILE * debug_fp = fopen("new.txt", "w");
        // fprintf(stdout, "orgIds:\n");
        for (its = 0; its < iterations; its++)
        {
            printf("[Lexi-order] Optimizing the numbering for its %u\n", its+1);
            for (m = 0; m < nmodes; m++)
                ptiLexiOrderPerMode(&tsr_temp, m, orgIds, sb_bits, tk, impl_num);

            // fprintf(stdout, "\niter %u:\n", its);
            // for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
            //     ptiDumpIndexArray(orgIds[m], tsr->ndims[m], stdout);
            // }
        }
        // fclose(debug_fp);

        /* compute newIndices from orgIds. Reverse perm */
        for (m = 0; m < nmodes; m++)
            for (i = 0; i < tsr->ndims[m]; i++)
                newIndices[m][orgIds[m][i]] = i;

        ptiFreeSparseTensor(&tsr_temp);
        for (m = 0; m < nmodes; m++)
            free(orgIds[m]);
        free(orgIds);

    } else if (renumber == 2 ) {    /* BFS-like renumbering */
       /*
        REMARK (10 May 2018): this is the old bfs-like kind of thing. I hoped it would reduce the number of iterations,
        but on a few cases it did not help much. Just leaving it in case we want to use it.
        */
        printf("[BFS-like]\n");
        ptiBFSLike(tsr, newIndices);
    }    
    
}

static void lexOrderThem(ptiNnzIndex m, ptiIndex n, ptiNnzIndex *ia, ptiIndex *cols, ptiIndex *cprm, int const tk)
{
    /*m, n are the num of rows and cols, respectively. We lex order cols,
     given rows.    
     */

    ptiNnzIndex j, jcol, jend;
    ptiIndex jj;
    
    ptiIndex *freeIdList, freeIdTop;
    ptiIndex k, s, acol;
    ptiIndex firstset, set, pos, item, headset;
    
    colStruct *clms;
    setStruct *csets;
    clms = (colStruct *) calloc(sizeof(colStruct), n+2);
    csets = (setStruct *) calloc(sizeof(setStruct), n+2);

    freeIdList = (ptiIndex*)calloc(sizeof(ptiIndex),(n+2));
    
    initColDLL(clms, n);
    initSetDLL(csets,  n);

    for(jj = 1; jj<=n; jj++)
        cprm[jj] = 2 * n;
    
    firstset = 1;
    freeIdList[0] = 0;
    
    for(jj= 1; jj<=n; jj++)
        freeIdList[jj] = jj+1;/*1 is used as a set id*/

    freeIdTop = 1;
    for(j=1; j<=m; j++){
        jend = ia[j+1]-1;
        for(jcol = ia[j]; jcol <= jend ; jcol++){
            acol= cols[jcol];
            s = clms[acol].svar;

            if( csets[s].flag < j){/*first occurence of supervar s in j*/            
                csets[s].flag = j;
                if(csets[s].sz == 1 && csets[s].tail != acol){
                    printf("this should not happen (sz 1 but tailset not ok)\n");
                    exit(12);
                }
                if(csets[s].sz > 1) {
                    ptiIndex newId;
                    /*remove acol from s*/
                    removeAColfromSet(csets, s, clms, acol);

                    /*create a new supervar ns=newId
                     and make i=acol its only var*/
                    if(freeIdTop == n+1) {
                        printf("this should not happen (no index)\n");
                        exit(12);
                    }
                    newId = freeIdList[freeIdTop++];

                    appendAColtoSet(csets, newId, clms, acol);
                    csets[s].var = acol; /*the new set's important var is acol*/

                    insertSetBefore(csets, newId, s);/*newId is before s*/                    
                    if(firstset == s)
                        firstset = newId;
                    
                }
            }
            else{/*second or later occurence of s for row j*/
                k = csets[s].var;
                /*remove acol from its current chain*/               
                removeAColfromSet(csets, s, clms, acol);

                if(csets[s].sz == 0){/*s is a free id now..*/                
                    freeIdList[--freeIdTop] = s; /*add s to the free id list*/                    
                    setEmpty(csets, s);/*no need to adjust firstset, as this is the second occ of s*/
                }
                /*add to chain containing k (as the last element)*/
                appendAColtoSet(csets, clms[k].svar, clms, acol);
            }
        }
    }
    
    /*we are done. Let us read the cprm from the ordered sets*/
    pos = 1;
    for(set = firstset; set != 0; set = csets[set].next){
        item = csets[set].tail;
        headset = 0;
        
        while(item != 0 ){
            headset = item;
            item = clms[item].prev;
        }
        /*located the head of the set. output them (this is for keeping the order)*/
        while(headset){
            cprm[pos++] = headset;
            headset = clms[headset].next;
        }
    }
    
    free(freeIdList);
    free(csets);
    free(clms);
    
    if(pos-1 != n){
        printf("**************** Error ***********\n");
        printf("something went wrong and we could not order everyone\n");
        exit(12);
    }
    
    return ;
}


// static void lexOrderThem( ptiNnzIndex m, ptiIndex n, ptiNnzIndex *ia, ptiIndex *cols, ptiIndex *cprm, int const tk)
// {
//     /*m, n are the num of rows and cols, respectively. We lex order cols,
//      given rows.
     
//      BU notes as of 4 May 2018: I am hoping that I will not be asked the details of this function, and its memory use;) A quick and dirty update from something else I had since some time. I did not think through if the arrays could be reduced. Right now we have 10 arrays of size n each (where n is the length of a single dimension of the tensor.
//      */
    
//     ptiNnzIndex *flag, j, jcol, jend;
//     ptiIndex *svar,  *var, numBlocks;
//     ptiIndex *prev, *next, *sz, *setnext, *setprev, *tailset;
    
//     ptiIndex *freeIdList, freeIdTop;
    
//     ptiIndex k, s, acol;
    
//     ptiIndex firstset, set, pos;
    
//     svar = (ptiIndex*) calloc(sizeof(ptiIndex),(n+2));
//     flag = (ptiNnzIndex*) calloc(sizeof(ptiNnzIndex),(n+2));
//     var  = (ptiIndex*) calloc(sizeof(ptiIndex),(n+2));
//     prev = (ptiIndex*) calloc(sizeof(ptiIndex),(n+2));
//     next = (ptiIndex*) calloc(sizeof(ptiIndex),(n+2));
//     sz   = (ptiIndex*) calloc(sizeof(ptiIndex),(n+2));
//     setprev = (ptiIndex*)calloc(sizeof(ptiIndex),(n+2));
//     setnext = (ptiIndex*)calloc(sizeof(ptiIndex),(n+2));
//     tailset = (ptiIndex*)calloc(sizeof(ptiIndex),(n+2));
//     freeIdList = (ptiIndex*)calloc(sizeof(ptiIndex),(n+2));
    
//     next[1] = 2;
//     prev[0] =  prev[1] = 0;
//     next[n] = 0;
//     prev[n] = n-1;
//     svar[1] = svar[n] = 1;
//     flag[1] = flag[n] = flag[n+1] = 0;
//     cprm[1] = cprm[n] = 2 * n ;
//     setprev[1] = setnext[1] = 0;
//     // #pragma omp parallel for num_threads(tk)
//     for(ptiIndex jj = 2; jj<=n-1; jj++)/*init all in a single svar*/
//     {
//         svar[jj] = 1;
//         next[jj] = jj+1;
//         prev[jj] = jj-1;
//         flag[jj] = 0;
//         sz[jj] = 0;
//         setprev[jj] = setnext[jj] = 0;
//         cprm[jj] = 2 * n;
//     }
//     var[1] = 1;
//     sz[1] = n;
//     sz[n] = sz[n+1] =  0;
    
//     setprev[n] = setnext[n] = 0;
//     setprev[n+1] = setnext[n+1] = 0;
    
//     tailset[1] = n;
    
//     firstset = 1;
//     freeIdList[0] = 0;
    
//     // #pragma omp parallel for num_threads(tk)
//     for(ptiIndex jj= 1; jj<=n; jj++)
//         freeIdList[jj] = jj+1;/*1 is used as a set id*/
    
//     freeIdTop = 1;
//     for(j=1; j<=m; j++)
//     {
//         jend = ia[j+1]-1;
//         for(jcol = ia[j]; jcol <= jend ; jcol++)
//         {
//             acol= cols[jcol];
//             s = svar[acol];
//             if( flag[s] < j)/*first occurence of supervar s in j*/
//             {
//                 flag[s] = j;
//                 if(sz[s] == 1 && tailset[s] != acol)
//                 {
//                     printf("this should not happen (sz 1 but tailset not ok)\n");
//                     exit(12);
//                 }
//                 if(sz[s] > 1)
//                 {
//                     ptiIndex newId;
//                     /*remove acol from s*/
//                     if(tailset[s] == acol) tailset[s] = prev[acol];
                    
//                     next[prev[acol]] = next[acol];
//                     prev[next[acol]] = prev[acol];
                    
//                     sz[s] = sz[s] - 1;
//                     create a new supervar ns=newId
//                      and make i=acol its only var
//                     if(freeIdTop == n+1) {
//                         printf("this should not happen (no index)\n");
//                         exit(12);
//                     }
//                     newId = freeIdList[freeIdTop++];
//                     svar[acol] = newId;
//                     var[newId] = acol;
//                     flag[newId] = j;
//                     sz[newId ] = 1;
//                     next[acol] = 0;
//                     prev[acol] = 0;
//                     var[s] = acol;
//                     tailset[newId] = acol;
                    
//                     setnext[newId] = s;
//                     setprev[newId] = setprev[s];
//                     if(setprev[s])
//                         setnext[setprev[s]] = newId;
//                     setprev[s] = newId;
                    
//                     if(firstset == s)
//                         firstset = newId;
                    
//                 }
//             }
//             else/*second or later occurence of s for row j*/
//             {
//                 k = var[s];
//                 svar[acol] = svar[k];
                
//                 /*remove acol from its current chain*/
//                 if(tailset[s] == acol) tailset[s] = prev[acol];
                
//                 next[prev[acol]] = next[acol];
//                 prev[next[acol]] = prev[acol];
                
//                 sz[s] = sz[s] - 1;
//                 if(sz[s] == 0)/*s is a free id now..*/
//                 {
                    
//                     freeIdList[--freeIdTop] = s; /*add s to the free id list*/
                    
//                     if(setnext[s])
//                         setprev[setnext[s]] = setprev[s];
//                     if(setprev[s])
//                         setnext[setprev[s]] = setnext[s];
                    
//                     setprev[s] = setnext[s] = 0;
//                     tailset[s] = 0;
//                     var[s] = 0;
//                     flag[s] = 0;
//                 }
//                 /*add to chain containing k (as the last element)*/
//                 prev[acol] = tailset[svar[k]];
//                 next[acol]  = 0;/*BU next[tailset[svar[k]]];*/
//                 next[tailset[svar[k]]] = acol;
//                 tailset[svar[k]] = acol;
//                 sz[svar[k]] = sz[svar[k]] + 1;
//             }
//         }
//     }
    
//     pos = 1;
//     numBlocks = 0;
//     for(set = firstset; set != 0; set = setnext[set])
//     {
//         ptiIndex item = tailset[set];
//         ptiIndex headset = 0;
//         numBlocks ++;
        
//         while(item != 0 )
//         {
//             headset = item;
//             item = prev[item];
//         }
//         /*located the head of the set. output them (this is for keeping the initial order*/
//         while(headset)
//         {
//             cprm[pos++] = headset;
//             headset = next[headset];
//         }
//     }
    
//     free(tailset);
//     free(sz);
//     free(next);
//     free(prev);
//     free(var);
//     free(flag);
//     free(svar);
//     free(setnext);
//     free(setprev);
//     if(pos-1 != n){
//         printf("**************** Error ***********\n");
//         printf("something went wrong and we could not order everyone\n");
//         exit(12);
//     }
    
//     return ;
// }


/**************************************************************/
#define myAbs(x) (((x) < 0) ? -(x) : (x))

static void ptiLexiOrderPerMode(ptiSparseTensor * tsr, ptiIndex mode, ptiIndex ** orgIds, ptiElementIndex sb_bits, int tk, int impl_num)
{
    ptiIndexVector * inds = tsr->inds;
    ptiNnzIndex const nnz = tsr->nnz;
    ptiIndex const nmodes = tsr->nmodes;
    ptiIndex * ndims = tsr->ndims;
    ptiIndex const mode_dim = ndims[mode];
    ptiNnzIndex * rowPtrs = NULL;
    ptiIndex * colIds = NULL;
    ptiIndex * cprm = NULL, * invcprm = NULL, * saveOrgIds = NULL;
    ptiNnzIndex atRowPlus1, mtxNrows, mtrxNnz;
    ptiIndex * mode_order = (ptiIndex *) malloc (sizeof(ptiIndex) * (nmodes - 1));

    ptiIndex c;
    ptiNnzIndex z;
    double t1, t0;

    t0 = u_seconds();
    ptiIndex i = 0;
    for(ptiIndex m = 0; m < nmodes; ++m) {
        if (m != mode) {
            mode_order[i] = m;
            ++ i;
        }
    }
    if (impl_num == 1) {
        ptiSparseTensorSortIndexExceptSingleMode(tsr, 1, mode_order, tk);
    } else if (impl_num == 2) {
        ptiSparseTensorSortIndexExceptSingleModeRowBlock(tsr, 1, 0, tsr->nnz, mode_order, sb_bits, tk);
    } else if (impl_num == 3) { // Not work
        ptiSparseTensorSortIndexExceptSingleModeMorton(tsr, 1, mode_order, sb_bits, tk);
    }
    // mySort(coords,  nnz-1, nmodes, ndims, mode);
    t1 = u_seconds()-t0;
    printf("mode %u, sort time %.2f\n", mode, t1);
    // ptiAssert(ptiDumpSparseTensor(tsr, 0, stdout) == 0);

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
    colIds[1] = inds[mode].data[0] + 1;
    atRowPlus1 = 2;
    mtrxNnz = 2;/* start filling from the second element */
    
    t0 = u_seconds();
    for (z = 1; z < nnz; z++)
    {
        int cmp_res;
        if (impl_num == 1) {
            cmp_res = pti_SparseTensorCompareIndicesExceptSingleMode(tsr, z, tsr, z-1, mode_order);
            // cmp_res = pti_SparseTensorCompareIndicesExceptSingleModeCantor(tsr, z, tsr, z-1, mode_order);
        } else if (impl_num == 2) {
            cmp_res = pti_SparseTensorCompareIndicesExceptSingleModeRowBlock(tsr, z, tsr, z-1, mode_order, sb_bits);
        } else if (impl_num == 3) {
            cmp_res = pti_SparseTensorCompareIndicesMorton2D(tsr, z, tsr, z-1, mode_order, sb_bits);
        }
        
        if(cmp_res != 0)
            rowPtrs[atRowPlus1++] = mtrxNnz; /* close the previous row and start a new one. */
        
        colIds[mtrxNnz ++] = inds[mode].data[z] + 1;
    }
    rowPtrs[atRowPlus1] = mtrxNnz;
    mtxNrows = atRowPlus1-1;
    t1 =u_seconds()-t0;
    printf("mode %u, create time %.2f\n", mode, t1);
    
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
    printf("mode %u, lexorder time %.2f\n", mode, t1);
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
        inds[mode].data[z] = invcprm[inds[mode].data[z]];
    // ptiAssert(ptiDumpSparseTensor(tsr, 0, stdout) == 0);
    
    free(saveOrgIds);
    free(invcprm);
    free(cprm);
    free(colIds);
    free(rowPtrs);
    free(mode_order);
}

/**************************************************************/

typedef struct{
    ptiIndex nvrt; /* number of vertices. This nvrt = n_0 + n_1 + ... + n_{d-1} for a d-dimensional tensor
                   where the ith dimension is of size n_i.*/
    ptiNnzIndex *vptrs, *vHids; /*starts of hedges containing vertices, and the ids of the hedges*/
    
    ptiNnzIndex nhdg; /*this will be equal to the number of nonzeros in the tensor*/
    ptiNnzIndex *hptrs, *hVids; /*starts of vertices in the hedges, and the ids of the vertices*/
} basicHypergraph;

static void allocateHypergraphData(basicHypergraph *hg, ptiIndex nvrt, ptiNnzIndex nhdg, ptiNnzIndex npins)
{
    hg->nvrt = nvrt;
    hg->vptrs = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * (nvrt+1));
    hg->vHids = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * npins);
    
    hg->nhdg = nhdg;
    hg->hptrs = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * (nhdg+1));
    hg->hVids = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * npins);
}


static void freeHypergraphData(basicHypergraph *hg)
{
    hg->nvrt = 0;
    if (hg->vptrs) free(hg->vptrs);
    if (hg->vHids) free(hg->vHids);
    
    hg->nhdg = 0;
    if (hg->hptrs) free(hg->hptrs);
    if (hg->hVids) free(hg->hVids);
}


static void setVList(basicHypergraph *hg)
{
    /*PRE: We assume hg->hptrs and hg->hVids are set; hg->nvrts is set, and
     hg->vptrs and hg->vHids are allocated appropriately.
     */
    
    ptiNnzIndex j, h, v, nhdg = hg->nhdg;
    
    ptiIndex nvrt = hg->nvrt;
    
    /*vertices */
    ptiNnzIndex *vptrs = hg->vptrs, *vHids = hg->vHids;
    /*hyperedges*/
    ptiNnzIndex *hptrs = hg->hptrs, *hVids = hg->hVids;
    
    for (v = 0; v <= nvrt; v++)
        vptrs[v] = 0;
    
    for (h = 0; h < nhdg; h++)
    {
        for (j = hptrs[h]; j < hptrs[h+1]; j++)
        {
            v = hVids[j];
            vptrs[v] ++;
        }
    }
    for (v=1; v <= nvrt; v++)
        vptrs[v] += vptrs[v-1];
    
    for (h = nhdg; h >= 1; h--)
    {
        for (j = hptrs[h-1]; j < hptrs[h]; j++)
        {
            v = hVids[j];
            vHids[--(vptrs[v])] = h-1;
        }
    }
}

static void fillHypergraphFromCoo(basicHypergraph *hg, ptiIndex nm, ptiNnzIndex nnz, ptiIndex *ndims, ptiIndexVector * inds)
{
    
    ptiIndex  totalSizes;
    ptiNnzIndex h, toAddress;
    ptiIndex *dimSizesPrefixSum;
    
    ptiIndex i;
    
    dimSizesPrefixSum = (ptiIndex *) malloc(sizeof(ptiIndex) * (nm+1));
    totalSizes = 0;
    for (i=0; i < nm; i++)
    {
        dimSizesPrefixSum[i] = totalSizes;
        totalSizes += ndims[i];
    }
    printf("allocating hyp %u %lu\n", nm, nnz);
    
    allocateHypergraphData(hg, totalSizes, nnz, nnz * nm);
    
    toAddress = 0;
    for (h = 0; h < nnz; h++)
    {
        hg->hptrs[h] = toAddress;
        for (i = 0;  i < nm; i++)
            hg->hVids[toAddress + i] = dimSizesPrefixSum[i] + inds[i].data[h];
        toAddress += nm;
    }
    hg->hptrs[hg->nhdg] = toAddress;
    
    setVList(hg);
    free(dimSizesPrefixSum);
}

static inline ptiIndex locateVertex(ptiNnzIndex indStart, ptiNnzIndex indEnd, ptiNnzIndex *lst, ptiNnzIndex sz)
{
    ptiNnzIndex i;
    for (i = 0; i < sz; i++)
        if(lst[i] >= indStart && lst[i] <= indEnd)
            return lst[i];
    
    printf("could not locate in a hyperedge !!!\n");
    exit(1);
    return sz+1;
}

#define SIZEV( vid ) vptrs[(vid)+1]-vptrs[(vid)]
static void heapIncreaseKey(ptiIndex *heapIds, ptiNnzIndex *key, ptiNnzIndex *vptrs, ptiIndex sz, ptiIndex id, ptiIndex *inheap, ptiNnzIndex newKey)
{
    
    ptiIndex i = inheap[id]; /*location in heap*/
    if( i > 0 && i <=sz )
    {
        key[id] = newKey;
        
        while ((i>>1)>0 && ( (key[id] > key[heapIds[i>>1]]) ||
                            (key[id] == key[heapIds[i>>1]] && SIZEV(id) > SIZEV(heapIds[i>>1])))
               )
        {
            heapIds[i] = heapIds[i>>1];
            inheap[heapIds[i]] = i;
            i = i>>1;
        }
        heapIds[i] = id;
        inheap[id] = i;
    }
}


static void heapify(ptiIndex *heapIds, ptiNnzIndex *key, ptiNnzIndex *vptrs, ptiIndex sz, ptiIndex i,  ptiIndex *inheap)
{
    ptiIndex largest, j, l,r, tmp;
    
    largest = j = i;
    while(j<=sz/2)
    {
        l = 2*j;
        r = 2*j + 1;
        
        if ( (key[heapIds[l]] > key[heapIds[j]] ) ||
            (key[heapIds[l]] == key[heapIds[j]]  && SIZEV(heapIds[l]) < SIZEV(heapIds[j]) )
            )
            largest = l;
        else
            largest = j;
        
        if (r<=sz && (key[heapIds[r]]>key[heapIds[largest]] ||
                      (key[heapIds[r]]==key[heapIds[largest]] && SIZEV(heapIds[r]) < SIZEV(heapIds[largest])))
            )
            largest = r;
        
        if (largest != j)
        {
            tmp = heapIds[largest];
            heapIds[largest] = heapIds[j];
            inheap[heapIds[j]] = largest;
            
            heapIds[j] = tmp;
            inheap[heapIds[j]] = j;
            j = largest;
        }
        else
            break;
    }
}

static ptiIndex heapExtractMax(ptiIndex *heapIds, ptiNnzIndex *key, ptiNnzIndex *vptrs, ptiIndex *sz, ptiIndex *inheap)
{
    ptiIndex maxind ;
    if (*sz < 1){
        printf("Error: heap underflow\n"); exit(12);
    }
    maxind = heapIds[1];
    heapIds[1] = heapIds[*sz];
    inheap[heapIds[1]] = 1;
    
    *sz = *sz - 1;
    inheap[maxind] = 0;
    
    heapify(heapIds, key, vptrs, *sz, 1, inheap);
    return maxind;
    
}

static void heapBuild(ptiIndex *heapIds, ptiNnzIndex *key, ptiNnzIndex *vptrs, ptiIndex sz, ptiIndex *inheap)
{
    ptiIndex i;
    for (i=sz/2; i>=1; i--)
        heapify(heapIds, key, vptrs, sz, i, inheap);
}

static void orderforHiCOOaDim(basicHypergraph *hg, ptiIndex *newIndicesHg, ptiIndex indStart, ptiIndex indEnd)
{
    /* we re-order the vertices of the hypergraph with ids in the range [indStart, indEnd]*/
    
    ptiIndex i, v, heapSz, *inHeap, *heapIds;
    ptiNnzIndex j, jj, hedge, hedge2, k, w, ww;
    ptiNnzIndex *vptrs = hg->vptrs, *vHids = hg->vHids, *hptrs = hg->hptrs, *hVids = hg->hVids;
    
    ptiNnzIndex *keyvals, newKeyval;
    int *markers, mark;
    
    mark = 0;
    
    heapIds = (ptiIndex*) malloc(sizeof(ptiIndex) * (indEnd-indStart + 2));
    inHeap = (ptiIndex*) malloc(sizeof(ptiIndex) * hg->nvrt);/*this is large*/
    keyvals = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * hg->nvrt);
    markers = (int*) malloc(sizeof(int)* hg->nvrt);
    
    heapSz = 0;
    
    for (i = indStart; i<=indEnd; i++)
    {
        keyvals[i] = 0;
        heapIds[++heapSz] = i;
        inHeap[i] = heapSz;
        markers[i] = -1;
    }
    heapBuild(heapIds, keyvals, vptrs, heapSz, inHeap);
    
    for (i = indStart; i <= indEnd; i++)
    {
        v = heapExtractMax(heapIds, keyvals, vptrs, &heapSz, inHeap);
        newIndicesHg[v] = i;
        markers[v] = mark;
        for (j = vptrs[v]; j < vptrs[v+1]; j++)
        {
            hedge = vHids[j];
            for (k = hptrs[hedge]; k < hptrs[hedge+1]; k++)
            {
                w = hVids[k];
                if (markers[w] != mark)
                {
                    markers[w] = mark;
                    for(jj = vptrs[w]; jj < vptrs[w+1]; jj++)
                    {
                        hedge2 = vHids[jj];
                        ww = locateVertex(indStart, indEnd, hVids + hptrs[hedge2], hptrs[hedge2+1]-hptrs[hedge2]);
                        if( inHeap[ww] )
                        {
                            newKeyval = keyvals[ww] + 1;
                            heapIncreaseKey(heapIds, keyvals, vptrs, heapSz, ww, inHeap, newKeyval);
                        }
                    }
                }
            }
        }
    }
    
    free(markers);
    free(keyvals);
    free(inHeap);
    free(heapIds);
}


/**************************************************************/
void ptiBFSLike(ptiSparseTensor * tsr, ptiIndex ** newIndices)
{
    /*PRE: newIndices is allocated
     
     POST:
     newIndices[0][0...n_0-1] gives the new ids for dim 0
     newIndices[1][0...n_1-1] gives the new ids for dim 1
     ...
     newIndices[d-1][0...n_{d-1}-1] gives the new ids for dim d-1
     
     This implements a simple idea close to BFS/Cuthill-McKee/Maximum cardinality search.
     */
    ptiIndex const nmodes = tsr->nmodes;
    ptiNnzIndex const nnz = tsr->nnz;
    ptiIndex * ndims = tsr->ndims;
    ptiIndexVector * inds = tsr->inds;
    
    ptiIndex *dimsPrefixSum;
    basicHypergraph hg;
    ptiIndex *newIndicesHg;
    ptiIndex d, i;
    
    dimsPrefixSum = (ptiIndex*) calloc(nmodes, sizeof(ptiIndex));
    for (d = 1; d < nmodes; d++)
        dimsPrefixSum[d] = ndims[d-1] + dimsPrefixSum[d-1];
    
    fillHypergraphFromCoo(&hg, nmodes,  nnz, ndims, inds);

    newIndicesHg = (ptiIndex*) malloc(sizeof(ptiIndex) * hg.nvrt);
    for (i = 0; i < hg.nvrt; i++)
        newIndicesHg[i] = i;
    
    for (d = 0; d < nmodes; d++) /*order d*/
        orderforHiCOOaDim(&hg, newIndicesHg, dimsPrefixSum[d], dimsPrefixSum[d] + ndims[d]-1);
    
    /*copy from newIndices to newIndicesOut*/
    for (d = 0; d < nmodes; d++)
        for (i = 0; i < ndims[d]; i++)
            newIndices[d][i] = newIndicesHg[dimsPrefixSum[d] + i] - dimsPrefixSum[d];
    
    free(newIndicesHg);
    freeHypergraphData(&hg);
    free(dimsPrefixSum);
    
}
/********************** Internals end *************************/
