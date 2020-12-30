#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "HiParTI.h"
#include "renumber.h"

/*Interface to everything in this file is orderit(.., ..)*/

/*function declarations*/
void orderDim(ptiIndex ** coords, ptiNnzIndex const nnz, ptiIndex const nm, ptiIndex * ndims, ptiIndex const dim, ptiIndex ** newIndices);

void orderforHiCOObfsLike(ptiIndex const nm, ptiNnzIndex const nnz, ptiIndex * ndims, ptiIndex ** coords, ptiIndex ** newIndices);

static double u_seconds(void)
{
    struct timeval tp;
    
    gettimeofday(&tp, NULL);
    
    return (double) tp.tv_sec + (double) tp.tv_usec / 1000000.0;
    
};
static void printCSR(ptiNnzIndex m, ptiIndex n, ptiNnzIndex *ia, ptiIndex *cols)
{
    ptiNnzIndex r, jend, jcol;
    printf("matrix of size %llu %u with %llu\n", m, n, ia[m+1]);
    
    for (r = 1; r <=m; r++)
    {
        jend = ia[r+1]-1;
        printf("r=%llu (%llu %llu)): ", r, ia[r], ia[r+1]);
        for(jcol = ia[r]; jcol <= jend ; jcol++)
            printf("%u ", cols[jcol]);
        printf("\n");
    }
}

static void checkRepeatIndex(ptiNnzIndex mtxNrows, ptiNnzIndex *rowPtrs, ptiIndex *cols, ptiIndex n )
{
    printf("\tChecking repeat indices\n");
    ptiIndex *marker = (ptiIndex *) calloc(n+1, sizeof(ptiIndex));
    ptiNnzIndex r,  jcol, jend;
    for (r = 1; r <= mtxNrows; r++)
    {
        jend = rowPtrs[r+1]-1;
        for (jcol = rowPtrs[r]; jcol <= jend; jcol++)
        {
            if( marker[cols[jcol]] < r )
                marker[cols[jcol]] = r;
            else if (marker[cols[jcol]] == r)
            {
                printf("*************************\n");
                printf("error duplicate col index %u at row %llu\n", cols[jcol], r);
                printf("*************************\n");
                
                exit(12);
            }
        }
        
    }
    free(marker);
}
static void checkEmptySlices(ptiIndex **coords, ptiNnzIndex nnz, ptiIndex nm, ptiIndex *ndims)
{
    ptiIndex m, i;
    ptiNnzIndex z;
    ptiIndex **marker;
    
    marker = (ptiIndex **) malloc(sizeof(ptiIndex*) * nm);
    for (m = 0; m < nm; m++)
        marker[m] = (ptiIndex*) calloc(ndims[m], sizeof(ptiIndex) );
    
    for (z = 0; z < nnz; z++)
        for (m=0; m < nm; m++)
            marker[m][coords[z][m]] = m + 1;
    
    for (m=0; m < nm; m++)
    {
        ptiIndex emptySlices = 0;
        for (i = 0; i < ndims[m]; i++)
            if(marker[m][i] != m+1)
                emptySlices ++;
        if(emptySlices)
            printf("dim %u, empty slices %u of %u\n", m, emptySlices,ndims[m] );
    }
    for (m = 0; m < nm; m++)
        free(marker[m]);
    free(marker);
}

static void checkNewIndices(ptiIndex **newIndices, ptiIndex nm, ptiIndex *ndims)
{
    ptiIndex m, i;
    ptiIndex **marker, leftVoid;
    
    marker = (ptiIndex **) malloc(sizeof(ptiIndex*) * nm);
    for (m = 0; m < nm; m++)
        marker[m] = (ptiIndex*) calloc(ndims[m], sizeof(ptiIndex) );
    
    for (m=0; m < nm; m++)
        for (i = 0; i < ndims[m]; i++)
            marker[m][newIndices[m][i]] = m + 1;
    
    leftVoid = 0;
    for (m=0; m < nm; m++)
    {
        for (i = 0; i < ndims[m]; i++)
            if(marker[m][i] != m+1)
                leftVoid ++;
        if(leftVoid)
            printf("dim %u, left void %u of %u\n", m, leftVoid, ndims[m] );
    }
    for (m = 0; m < nm; m++)
        free(marker[m]);
    free(marker);
}


void orderit(ptiSparseTensor * tsr, ptiIndex ** newIndices, int const renumber, ptiIndex const iterations)
{
    /*
     newIndices is of size [nmodes][ndims[modes]] and assumed to be allocted.
     It will be overwritten. No need to initialize.
     
     We will need to reshuffle nonzeros. In order to not to touch tsr, we copy the indices of nonzeros
     to a local variable coords. This is sort of transposed wrt tsr: its size is nnz * n, instead of n * nnz used in tsr.
     */
    ptiIndex i, m, nm = tsr->nmodes;
    ptiNnzIndex z, nnz = tsr->nnz;
    ptiIndex ** coords;
    ptiIndex its;
    
    /* copy the indices */
    ptiTimer copy_coord_timer;
    ptiNewTimer(&copy_coord_timer, 0);
    ptiStartTimer(copy_coord_timer);

    coords = (ptiIndex **) malloc(sizeof(ptiIndex*) * nnz);
    for (z = 0; z < nnz; z++)
    {
        coords[z] = (ptiIndex *) malloc(sizeof(ptiIndex) * nm);
        for (m = 0; m < nm; m++) {
            coords[z][m] = tsr->inds[m].data[z];
        }
    }

    ptiStopTimer(copy_coord_timer);
    ptiPrintElapsedTime(copy_coord_timer, "Copy coordinate time");
    ptiFreeTimer(copy_coord_timer);
    
    /* checkEmptySlices(coords, nnz, nm, tsr->ndims); */

    if (renumber == 1) {    /* Lexi-order renumbering */

        ptiIndex ** orgIds = (ptiIndex **) malloc(sizeof(ptiIndex*) * nm);

        for (m = 0; m < nm; m++)
        {
            orgIds[m] = (ptiIndex *) malloc(sizeof(ptiIndex) * tsr->ndims[m]);
            for (i = 0; i < tsr->ndims[m]; i++)
                orgIds[m][i] = i;
        }

        // FILE * debug_fp = fopen("old.txt", "w");
        // fprintf(stdout, "orgIds:\n");
        for (its = 0; its < iterations; its++)
        {
            printf("[Lexi-order] Optimizing the numbering for its %u\n", its+1);
            for (m = 0; m < nm; m++)
                orderDim(coords, nnz, nm, tsr->ndims, m, orgIds);
            
            // fprintf(stdout, "\niter %u:\n", its);
            // for(ptiIndex m = 0; m < tsr->nmodes; ++m) {
            //     ptiDumpIndexArray(orgIds[m], tsr->ndims[m], stdout);
            // }
        }
        // fclose(debug_fp);

        /* compute newIndices from orgIds. Reverse perm */
        for (m = 0; m < nm; m++)
            for (i = 0; i < tsr->ndims[m]; i++)
                newIndices[m][orgIds[m][i]] = i;

        for (m = 0; m < nm; m++)
            free(orgIds[m]);
        free(orgIds);

    } else if (renumber == 2 ) {    /* BFS-like renumbering */
       /*
        REMARK (10 May 2018): this is the old bfs-like kind of thing. I hoped it would reduce the number of iterations,
        but on a few cases it did not help much. Just leaving it in case we want to use it.
        */
        printf("[BFS-like]\n");
        orderforHiCOObfsLike(nm, nnz, tsr->ndims, coords, newIndices);
    }
    
    // printf("set the new indices\n");
/*    checkNewIndices(newIndices, nm, tsr->ndims);*/
    
    for (z = 0; z < nnz; z++)
        free(coords[z]);
    free(coords);
    
}
/******************** Internals begin ***********************/
/*beyond this line savages....
 **************************************************************/
static void printCoords(ptiIndex **coords, ptiNnzIndex nnz, ptiIndex nm)
{
    ptiNnzIndex z;
    ptiIndex m;
    for (z = 0; z < nnz; z++)
    {
        for (m=0; m < nm; m++)
            printf("%d ", coords[z][m]);
        printf("\n");
    }
}
/**************************************************************/
// static inline int isLessThanOrEqualToCoord(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex *ndims, ptiIndex dim)
static inline int isLessThanOrEqualTo(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex *ndims, ptiIndex dim)
{
    /*is z1 less than or equal to z2 for all indices except dim?*/
    ptiIndex m;
    
    for (m = 0; m < nm; m++)
    {
        if(m != dim)
        {
            if (z1[m] < z2[m])
                return -1;
            if (z1[m] > z2[m])
                return 1;
        }
    }
    return 0; /*are equal*/
}


static inline int isLessThanOrEqualToFast(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex * mode_order)
{
    /*is z1 less than or equal to z2 for all indices except dim?*/
    ptiIndex i, m;
    
    for (i = 0; i < nm - 1; i ++)
    {
        m = mode_order[i];
        if (z1[m] < z2[m])
            return -1;
        if (z1[m] > z2[m])
            return 1;
    }
    return 0; /*are equal*/
}


static inline int isLessThanOrEqualToNewSum(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex *ndims, ptiIndex dim)
// static inline int isLessThanOrEqualTo(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex *ndims, ptiIndex dim)
{
    /*
     to sort the nonzeros first on i_1+i_2+...+i_4, if ties then on
     i_1+i_2+...+3, if ties then on i_1+i_2, if ties then on i_1 only.
     We do not include dim in the comparisons.
     
    */
    ptiIndex m;
    ptiIndex v1 = 0, v2 = 0;
    
    for (m = 0; m < nm; m++)
    {
        if(m != dim)
        {
            v1 += z1[m];
            v2 += z2[m];
        }
    }
    if(v1 < v2) return -1;
    else if(v1 > v2) return 1;
    else{
        for (m = 0; m < nm; m++)
        {
            if(m != dim)
            {
                v1 -= z1[m];
                v2 -= z2[m];
                if (v1 < v2) return -1;
                else if (v1 > v2) return 1;
            }
        }
    }
    return 0; /*are equal*/
}
/**************************************************************/
static inline void buSwap(ptiIndex *z1, ptiIndex *z2, ptiIndex nm, ptiIndex *wspace)
{
    ptiIndex m;
    
    for (m=0; m < nm; m++)
        wspace[m] = z2[m];
    
    for (m=0; m < nm; m++)
        z2[m] = z1[m];
    
    for (m=0; m < nm; m++)
        z1[m] = wspace[m];
    
}

static inline void writeInto(ptiIndex *target, ptiIndex *source, ptiIndex nm)
{
    ptiIndex m;
    for (m = 0; m < nm; m++)
        target[m] = source[m];
}

static void insertionSort(ptiIndex **coords, ptiNnzIndex lo, ptiNnzIndex hi, ptiIndex nm, ptiIndex *ndims, ptiIndex dim, ptiIndex *tmpNnz, ptiIndex *wspace)
{
    ptiNnzIndex z, z2plus;
    for (z = lo+1; z <= hi; z++)
    {
        writeInto(tmpNnz, coords[z], nm);
        /*find place for z*/
        z2plus = z;
        while ( z2plus > 0  && isLessThanOrEqualTo(coords[z2plus-1], tmpNnz, nm, ndims, dim)== 1)
        {
            writeInto(coords[z2plus], coords[z2plus-1], nm);
            z2plus --;
        }
        writeInto(coords[z2plus], tmpNnz, nm);
    }
}

static void insertionSortFast(ptiIndex **coords, ptiNnzIndex lo, ptiNnzIndex hi, ptiIndex nm, ptiIndex *ndims, ptiIndex dim, ptiIndex * mode_order, ptiIndex *tmpNnz, ptiIndex *wspace)
{
    ptiNnzIndex z, z2plus;
    for (z = lo+1; z <= hi; z++)
    {
        writeInto(tmpNnz, coords[z], nm);
        /*find place for z*/
        z2plus = z;
        while ( z2plus > 0  && isLessThanOrEqualToFast(coords[z2plus-1], tmpNnz, nm, mode_order)== 1)
        {
            writeInto(coords[z2plus], coords[z2plus-1], nm);
            z2plus --;
        }
        writeInto(coords[z2plus], tmpNnz, nm);
    }
}

static inline ptiNnzIndex buPartition(ptiIndex **coords, ptiNnzIndex lo, ptiNnzIndex hi, ptiIndex nm, ptiIndex *ndims, ptiIndex dim, ptiIndex *tmpNnz, ptiIndex *wspace)
{
    /* copied from the web http://ndevilla.free.fr/median/median/src/quickselect.c */
    ptiNnzIndex low, high, median, middle, ll, hh;
    
    low = lo; high = hi; median = (low+high)/2;
    for(;;)
    {
        if (high<=low) return median;
        if(high == low + 1)
        {
            if(isLessThanOrEqualTo(coords[low], coords[high], nm, ndims, dim)== 1)
                buSwap (coords[high], coords[low], nm, wspace);
            return median;
        }
        middle = (low+high)/2;
        if(isLessThanOrEqualTo(coords[middle], coords[high], nm, ndims, dim) == 1)
            buSwap (coords[middle], coords[high], nm, wspace);
        
        if(isLessThanOrEqualTo(coords[low], coords[high], nm, ndims, dim) == 1)
            buSwap (coords[low], coords[high], nm, wspace);
        
        if(isLessThanOrEqualTo(coords[middle], coords[low], nm, ndims, dim) == 1)
            buSwap (coords[low], coords[middle], nm, wspace);
        
        buSwap (coords[middle], coords[low+1], nm, wspace);
        
        ll = low + 1;
        hh = high;
        for (;;){
            do ll++; while (isLessThanOrEqualTo(coords[low], coords[ll], nm, ndims, dim) == 1);
            do hh--; while (isLessThanOrEqualTo(coords[hh], coords[low], nm, ndims, dim) == 1);
            
            if (hh < ll) break;
            
            buSwap (coords[ll], coords[hh], nm, wspace);
        }
        buSwap (coords[low], coords[hh], nm,wspace);
        if (hh <= median) low = ll;
        if (hh >= median) high = hh - 1;
    }
    
}

static inline ptiNnzIndex buPartitionFast(ptiIndex **coords, ptiNnzIndex lo, ptiNnzIndex hi, ptiIndex nm, ptiIndex *ndims, ptiIndex dim, ptiIndex * mode_order, ptiIndex *tmpNnz, ptiIndex *wspace)
{
    /* copied from the web http://ndevilla.free.fr/median/median/src/quickselect.c */
    ptiNnzIndex low, high, median, middle, ll, hh;
    
    low = lo; high = hi; median = (low+high)/2;
    for(;;)
    {
        if (high<=low) return median;
        if(high == low + 1)
        {
            if(isLessThanOrEqualToFast(coords[low], coords[high], nm, mode_order)== 1)
                buSwap (coords[high], coords[low], nm, wspace);
            return median;
        }
        middle = (low+high)/2;
        if(isLessThanOrEqualToFast(coords[middle], coords[high], nm, mode_order) == 1)
            buSwap (coords[middle], coords[high], nm, wspace);
        
        if(isLessThanOrEqualToFast(coords[low], coords[high], nm, mode_order) == 1)
            buSwap (coords[low], coords[high], nm, wspace);
        
        if(isLessThanOrEqualToFast(coords[middle], coords[low], nm, mode_order) == 1)
            buSwap (coords[low], coords[middle], nm, wspace);
        
        buSwap (coords[middle], coords[low+1], nm, wspace);
        
        ll = low + 1;
        hh = high;
        for (;;){
            do ll++; while (isLessThanOrEqualToFast(coords[low], coords[ll], nm, mode_order) == 1);
            do hh--; while (isLessThanOrEqualToFast(coords[hh], coords[low], nm, mode_order) == 1);
            
            if (hh < ll) break;
            
            buSwap (coords[ll], coords[hh], nm, wspace);
        }
        buSwap (coords[low], coords[hh], nm,wspace);
        if (hh <= median) low = ll;
        if (hh >= median) high = hh - 1;
    }   
    
}

/**************************************************************/
static void mySort(ptiIndex ** coords,  ptiNnzIndex last, ptiIndex nm, ptiIndex * ndims, ptiIndex dim)
{    
    /* sorts coords accourding to all dims except dim, where items are refereed with newIndices*/
    /* an iterative quicksort */
    ptiNnzIndex *stack, top, lo, hi, pv;
    ptiIndex *tmpNnz, *wspace;
    
    tmpNnz = (ptiIndex*) malloc(sizeof(ptiIndex) * nm);
    wspace = (ptiIndex*) malloc(sizeof(ptiIndex) * nm);
    stack = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * 2 * (last+2));
    
    if(stack == NULL) {
        printf("could not allocated stack. returning\n");
        exit(14);
    }
    top = 0;
    stack[top++] = 0;
    stack[top++] = last;
    while (top>=2)
    {
        hi = stack[--top];
        lo = stack[--top];
        pv = buPartition(coords, lo, hi, nm, ndims, dim, tmpNnz, wspace);
        
        if(pv > lo+1)
        {
            if(pv - lo > 128)
            {
                stack[top++] = lo;
                stack[top++] = pv-1 ;
            }
            else
                insertionSort(coords, lo, pv-1,  nm, ndims, dim, tmpNnz, wspace);
        }
        if(top >= 2 * (last+2)){
            printf("\thow come this tight?\n");
            exit(13);
        }
        if(pv + 1 < hi)
        {
            if(hi - pv > 128)
            {
                stack[top++] = pv + 1 ;
                stack[top++] = hi;
            }
            else
                insertionSort(coords, pv+1, hi,  nm, ndims, dim, tmpNnz, wspace);
        }
        if( top >= 2 * (last+2)) {
            printf("\thow come this tight?\n");
            exit(13);
        }
    }
    free(stack);
    free(wspace);
    free(tmpNnz);
}


static void mySortFast(ptiIndex ** coords,  ptiNnzIndex last, ptiIndex nm, ptiIndex * ndims, ptiIndex dim, ptiIndex * mode_order)
{
    /* sorts coords accourding to all dims except dim, where items are refereed with newIndices*/
    /* an iterative quicksort */
    ptiNnzIndex *stack, top, lo, hi, pv;
    ptiIndex *tmpNnz, *wspace;
    
    tmpNnz = (ptiIndex*) malloc(sizeof(ptiIndex) * nm);
    wspace = (ptiIndex*) malloc(sizeof(ptiIndex) * nm);
    stack = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * 2 * (last+2));
    
    if(stack == NULL) {
        printf("could not allocated stack. returning\n");
        exit(14);
    }
    top = 0;
    stack[top++] = 0;
    stack[top++] = last;
    while (top>=2)
    {
        hi = stack[--top];
        lo = stack[--top];
        pv = buPartitionFast(coords, lo, hi, nm, ndims, dim, mode_order, tmpNnz, wspace);
        
        if(pv > lo+1)
        {
            if(pv - lo > 128)
            {
                stack[top++] = lo;
                stack[top++] = pv-1 ;
            }
            else
                insertionSortFast(coords, lo, pv-1,  nm, ndims, dim, mode_order, tmpNnz, wspace);
        }
        if(top >= 2 * (last+2)){
            printf("\thow come this tight?\n");
            exit(13);
        }
        if(pv + 1 < hi)
        {
            if(hi - pv > 128)
            {
                stack[top++] = pv + 1 ;
                stack[top++] = hi;
            }
            else
                insertionSortFast(coords, pv+1, hi,  nm, ndims, dim, mode_order, tmpNnz, wspace);
        }
        if( top >= 2 * (last+2)) {
            printf("\thow come this tight?\n");
            exit(13);
        }
    }

    free(stack);
    free(wspace);
    free(tmpNnz);
}

static ptiIndex countNumItems(ptiIndex *setnext, ptiIndex *tailset, ptiIndex firstset, ptiIndex *prev)
{
    ptiIndex cnt = 0, set;
    for(set = firstset; set != 0; set = setnext[set])
    {
        ptiIndex item = tailset[set];
        
        while(item != 0 )
        {
            cnt ++;
            item = prev[item];
        }
    }
    return cnt;
}



void initColDLL(colStruct *clms, ptiIndex n)
{
    ptiIndex jj;

    clms[1].next = 2;
    clms[0].prev =  clms[1].prev = 0;
    clms[n].next = 0;
    clms[n].prev = n-1;
    clms[1].svar = clms[n].svar = 1;
    for(jj = 2; jj<=n-1; jj++)/*init all in a single svar*/
    {
        clms[jj].svar = 1;
        clms[jj].next = jj+1;
        clms[jj].prev = jj-1;
    }
}

void initSetDLL(setStruct *csets, ptiIndex n)
{
    ptiIndex jj;

    csets[1].flag = 0;
    csets[1].prev = csets[1].next =  0;
    csets[1].var = 1;
    csets[1].sz = n;
    csets[1].tail = n;

    for(jj = 2; jj<=n+1; jj++){/*init all in a single svar*/    
        csets[jj].flag = 0;    
        csets[jj].sz = 0;
        csets[jj].prev =  csets[jj].next = 0;
    }
}


void lexOrderThem(ptiNnzIndex m, ptiIndex n, ptiNnzIndex *ia, ptiIndex *cols, ptiIndex *cprm)
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


/**************************************************************/
#define myAbs(x) (((x) < 0) ? -(x) : (x))

void orderDim(ptiIndex ** coords, ptiNnzIndex const nnz, ptiIndex const nm, ptiIndex * ndims, ptiIndex const dim, ptiIndex ** orgIds)
{
    ptiNnzIndex * rowPtrs=NULL, z, atRowPlus1, mtxNrows;
    ptiIndex * colIds=NULL, c;
    ptiIndex * cprm=NULL, * invcprm = NULL, * saveOrgIds;
    ptiNnzIndex mtrxNnz;

   ptiIndex * mode_order = (ptiIndex *) malloc (sizeof(ptiIndex) * (nm - 1));
    ptiIndex i = 0;
    for(ptiIndex m = 0; m < nm; ++m) {
        if (m != dim) {
            mode_order[i] = m;
            ++ i;
        }
    }
    
    double t1, t0;
    t0 = u_seconds();
    // mySort(coords,  nnz-1, nm, ndims, dim);
    mySortFast(coords,  nnz-1, nm, ndims, dim, mode_order);
    t1 = u_seconds()-t0;
    printf("dim %u, sort time %.2f\n", dim, t1);
    // printCoords(coords, nnz, nm);
    /* we matricize this (others x thisDim), whose columns will be renumbered */
    
    /* on the matrix all arrays are from 1, and all indices are from 1. */
    
    rowPtrs = (ptiNnzIndex *) malloc(sizeof(ptiNnzIndex) * (nnz+2)); /*large space*/
    colIds = (ptiIndex *) malloc(sizeof(ptiIndex) * (nnz+2)); /*large space*/
    
    if(rowPtrs == NULL || colIds == NULL)
    {
        printf("could not allocate.exiting \n");
        exit(12);
    }
    
    rowPtrs[0] = 0; /* we should not access this, that is why. */
    rowPtrs [1] = 1;
    colIds[1] = coords[0][dim]+1;
    atRowPlus1 = 2;
    mtrxNnz = 2;/* start filling from the second element */
    
    t0 = u_seconds();
    for (z = 1; z < nnz; z++)
    {
        // if(isLessThanOrEqualTo( coords[z], coords[z-1], nm, ndims, dim) != 0)
        if(isLessThanOrEqualToFast( coords[z], coords[z-1], nm, mode_order) != 0)
            rowPtrs[atRowPlus1 ++] = mtrxNnz; /* close the previous row and start a new one. */
        
        colIds[mtrxNnz++] = coords[z][dim]+1;
    }
    rowPtrs[atRowPlus1] = mtrxNnz;
    mtxNrows = atRowPlus1-1;
    t1 =u_seconds()-t0;
    printf("dim %u create time %.2f\n", dim, t1);
    
    rowPtrs = realloc(rowPtrs, (sizeof(ptiNnzIndex) * (mtxNrows+2)));
    cprm = (ptiIndex *) malloc(sizeof(ptiIndex) * (ndims[dim]+1));
    invcprm = (ptiIndex *) malloc(sizeof(ptiIndex) * (ndims[dim]+1));
    saveOrgIds = (ptiIndex *) malloc(sizeof(ptiIndex) * (ndims[dim]+1));
    /*    checkRepeatIndex(mtxNrows, rowPtrs, colIds, ndims[dim] );*/

    // printf("rowPtrs: \n");
    // ptiDumpNnzIndexArray(rowPtrs, mtxNrows + 2, stdout);
    // printf("colIds: \n");
    // ptiDumpIndexArray(colIds, nnz + 2, stdout);
    
    t0 = u_seconds();
    lexOrderThem(mtxNrows, ndims[dim], rowPtrs, colIds, cprm);
    t1 =u_seconds()-t0;
    printf("dim %u lexorder time %.2f\n", dim, t1);
    // printf("cprm: \n");
    // ptiDumpIndexArray(cprm, ndims[dim] + 1, stdout);

    /* update orgIds and modify coords */
    for (c=0; c < ndims[dim]; c++)
    {
        invcprm[cprm[c+1]-1] = c;
        saveOrgIds[c] = orgIds[dim][c];
    }
    for (c=0; c < ndims[dim]; c++)
        orgIds[dim][c] = saveOrgIds[cprm[c+1]-1];
    
    // printf("invcprm: \n");
    // ptiDumpIndexArray(invcprm, ndims[dim] + 1, stdout);

    /*rename the dim component of nonzeros*/
    for (z = 0; z < nnz; z++)
        coords[z][dim] = invcprm[coords[z][dim]];
    
    free(mode_order);
    free(saveOrgIds);
    free(invcprm);
    free(cprm);
    free(colIds);
    free(rowPtrs);
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

static void fillHypergraphFromCoo(basicHypergraph *hg, ptiIndex nm, ptiNnzIndex nnz, ptiIndex *ndims, ptiIndex **coords)
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
    printf("allocating hyp %u %llu\n", nm, nnz);
    
    allocateHypergraphData(hg, totalSizes, nnz, nnz * nm);
    
    toAddress = 0;
    for (h = 0; h < nnz; h++)
    {
        hg->hptrs[h] = toAddress;
        for (i = 0;  i < nm; i++)
            hg->hVids[toAddress + i] = dimSizesPrefixSum[i] + coords[h][i];
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
void orderforHiCOObfsLike(ptiIndex const nm, ptiNnzIndex const nnz, ptiIndex * ndims, ptiIndex ** coords, ptiIndex ** newIndices)
{
    /*PRE: newIndices is allocated
     
     POST:
     newIndices[0][0...n_0-1] gives the new ids for dim 0
     newIndices[1][0...n_1-1] gives the new ids for dim 1
     ...
     newIndices[d-1][0...n_{d-1}-1] gives the new ids for dim d-1
     
     This implements a simple idea close to BFS/Cuthill-McKee/Maximum cardinality search.
     */
    ptiIndex d, i;
    ptiIndex *dimsPrefixSum;
    
    basicHypergraph hg;
    
    ptiIndex *newIndicesHg;
    
    dimsPrefixSum = (ptiIndex*) calloc(nm, sizeof(ptiIndex));
    for (d = 1; d < nm; d++)
        dimsPrefixSum[d] = ndims[d-1] + dimsPrefixSum[d-1];
    
    fillHypergraphFromCoo(&hg, nm,  nnz, ndims, coords);
    newIndicesHg = (ptiIndex*) malloc(sizeof(ptiIndex) * hg.nvrt);
    
    for (i = 0; i < hg.nvrt; i++)
        newIndicesHg[i] = i;
    
    for (d = 0; d < nm; d++) /*order d*/
        orderforHiCOOaDim(&hg, newIndicesHg, dimsPrefixSum[d], dimsPrefixSum[d] + ndims[d]-1);
    
    /*copy from newIndices to newIndicesOut*/
    for (d = 0; d < nm; d++)
        for (i = 0; i < ndims[d]; i++)
            newIndices[d][i] = newIndicesHg[dimsPrefixSum[d] + i] - dimsPrefixSum[d];
    
    free(newIndicesHg);
    freeHypergraphData(&hg);
    free(dimsPrefixSum);
    
}
/********************** Internals end *************************/
