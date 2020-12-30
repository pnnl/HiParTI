typedef struct{
    ptiIndex svar;
    ptiIndex prev;
    ptiIndex next;
} colStruct;


typedef struct{
    ptiNnzIndex flag;
    ptiIndex var;
    ptiIndex prev;
    ptiIndex next;
    ptiIndex sz;
    ptiIndex tail;
} setStruct;

static inline void removeAColfromSet(setStruct *csets, ptiIndex aset, colStruct *clms, ptiIndex acol)
{
    if(csets[aset].tail == acol) csets[aset].tail = clms[acol].prev;
    csets[aset].sz = csets[aset].sz - 1;

    clms[clms[acol].prev].next = clms[acol].next;
    clms[clms[acol].next].prev = clms[acol].prev;

}

static inline void setEmpty(setStruct *csets, ptiIndex aset)
{
    if(csets[aset].next)
        csets[csets[aset].next].prev = csets[aset].prev;
    if(csets[aset].prev)
        csets[csets[aset].prev].next = csets[aset].next;

    csets[aset].prev = csets[aset].next = 0;
    csets[aset].tail = 0;
    csets[aset].var = 0;
    csets[aset].flag = 0;
    csets[aset].sz = 0;
}

static inline void appendAColtoSet(setStruct *csets, ptiIndex aset, colStruct *clms, ptiIndex acol)
{
    clms[acol].prev = csets[aset].tail;
    clms[acol].next = 0;
    clms[acol].svar = aset;

    if(csets[aset].tail)
        clms[csets[aset].tail].next = acol;
    csets[aset].tail = acol;
    csets[aset].sz = csets[aset].sz + 1;
}

static inline void insertSetBefore(setStruct *csets, ptiIndex newset, ptiIndex aset)
{
    csets[newset].next = aset;
    csets[newset].prev = csets[aset].prev;
    if(csets[aset].prev)
        csets[csets[aset].prev].next = newset;
    csets[aset].prev = newset;
}

