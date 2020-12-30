from HiParTIPy.PtiCffi import pti,PTI

import HiParTIPy.Vector as vec
from math import sqrt
import os as os
import sys

class COOMatrix:
    def __init__(self):
        self.address = pti.new("ptiSparseMatrix *")
        self.nthreads = (int)(os.popen('grep -c cores /proc/cpuinfo').read())
        self.map = None

    # Reads the matrix into a file, path specified in filename.
    # @:param filename, string path to the output file
    def loadMatrix(self, filename):
        file = PTI.fopen(bytes(filename, 'ascii'), b"r")
        PTI.ptiLoadSparseMatrix(self.address, 0, file)

    # Writes the matrix into a file, path specified in filename.
    # @:param filename, string path to the output file
    def dumpMatrix(self, filename):
        file = PTI.fopen(bytes(filename, 'ascii'), b"w")
        PTI.ptiDumpSparseMatrix(self.address, 0, file)

    def statusMatrix(self):
        PTI.ptiSparseMatrixStatus(self.address,sys.stdout)

    # These get methods are to make an easy way for a user to get the instances of the structures in C
    # A user can still call self.address.<>, but we want to avoid that if possible.
    def ncols(self): return int(self.address.ncols)
    def nrows(self): return int(self.address.nrows)
    def nnz(self): return int(self.address.nnz)

    # This function converts the COO matrix to a corresponding matrix
    # @:param b, this declares how big to make blocks, at 2^bitsize.  Default is seven.
    # @:param k, this is a superblock bitsize for parallelization, Default is equal to b.
    #       A proper combination of b and k speed up HiCOO methods!!
    # @:returns HiCOOMatrix/CSRMatrix/DenseMatrix, equivalent version of this previous one.
    def convert(self,type="",blockbits=7,superblockbits=7):
        if(type=="hicoo"):
            result = HiCOOMatrix()
            nnz = pti.new("uint64_t *", self.nnz())
            PTI.ptiSparseMatrixToHiCOO(result.address, nnz, self.address, blockbits, superblockbits)
        elif(type=="csr"):
            result=CSRMatrix()
            PTI.ptiSparseMatrixToCSR(result.address,self.address)
        elif(type=="dense"):
            result=DenseMatrix(self.address.nrows,self.address.ncols)
            for i in range(self.address.nnz):
                m=self.address.rowind.data[i]
                n=self.address.colind.data[i]
                data=self.address.values.data[i]
                result.setValue(m,n,data)
        else:
            print("[ERROR] Wrong sparse matrix type.")
        return result

    # TODO: Make Calls to make method for only multiply
    # This function multiplies the matrix by a vector
    # @:param vector, Value Vector of size ncols
    # @:return ValueVector, the resulting value vector of the multiplication is returned
    def multiplyVector(self, vector, type="default", testing=False):
        result=vec.ValueVector(self.address.nrows)
        size=self.address.nnz
        sp=size/self.address.ncols
        if type=="default" :
            if size<1246221 : type="serial"
            elif sp<28.8 : type="CPU"
            else: type="CPU_Buff"
        if(type=="serial"): PTI.ptiSparseMatrixMulVector(result.address,self.address,vector.address)
        elif(type=="CPU"): PTI.ptiOmpSparseMatrixMulVector(result.address,self.address,vector.address)
        elif(type=="CPU_Buff"): PTI.ptiOmpSparseMatrixMulVectorReduce(result.address,self.address,vector.address)
        if not testing: return result
        else: PTI.ptiFreeValueVector(result.address)

    # This function multiplies the matrix by a dense matrix
    # @:param mat, Dense Matrix where their nrows=self.ncols.
    # @:return Dense Matrix, the resulting product is returned.  Note of size self.nrows,mat.ncols
    def multiplyMatrix(self, mat, type="default", testing=False):
        result=DenseMatrix(self.address.nrows,mat.address.ncols)
        if type=="default":
            nnz = self.address.nnz
            r = mat.address.ncols
            s = self.address.nrows
            a = 0.1780 + 1.189E-8 * nnz / r / r / r - 11.58 / sqrt(r * s) + 3.363 / sqrt(r)
            b = sqrt(s*nnz)*r
            if a<1 or nnz<1000 : PTI.ptiSparseMatrixMulMatrix(result.address,self.address,mat.address)
            elif b<300000000 : PTI.ptiOmpSparseMatrixMulMatrix(result.address,self.address,mat.address)
            else: PTI.ptiOmpSparseMatrixMulMatrixReduce(result.address,self.address,mat.address)
        elif type=="serial": PTI.ptiSparseMatrixMulMatrix(result.address,self.address,mat.address)
        elif type=="CPU": PTI.pitOmpSparseMatrixMulMatrix(result.address,self.address,mat.address)
        elif type=="CPU_Buff": PTI.ptiOmpSparseMatrixMulMatrixReduce(result.address,self.address,mat.address)
        PTI.ptiSparseMatrixMulMatrix(result.address,self.address,mat.address)
        if not testing: return result
        else: PTI.ptiFreeMatrix(result.address)

    # This function multiplies the matrix by a vector, using the buffered parallel method only
    # @:param vector, A Value Vector for multplication
    # @:param buffer, A vector buffer for the paralleled method.  This can be reused from old calls to save allocation time
    # @:param testing, To either return the value (not a test) or deallocate result for calling many calls and time.
    def multiplyVectorBuff(self, vector, buffer,testing=False):
        result=vec.ValueVector(self.address.nrows)
        PTI.ptiOmpSparseMatrixMulVector_Reduce(result.address,buffer.address,self.address,vector.address)
        if not testing: return result
        else: PTI.ptiFreeValueVector(result.address)

    # These functions sort the matrix by a set of differnet members
    # Rowsort sorts by first index, row index
    def sort(self, type="row", blockbits=1):
        if(type=="row"): # row->column order
            PTI.ptiSparseMatrixSortIndexSingleMode(self.address,1,0,self.nthreads)
        elif(type=="col"): # column->row order
            PTI.ptiSparseMatrixSortIndexSingleMode(self.address,1,1,self.nthreads)
        elif(type=="block"): # natural blocking sort
            PTI.ptiSparseMatrixSortIndexRowBlock(self.address,1,0,self.address.nnz,blockbits)
        elif(type=="morton"): # Z-morton sort
            PTI.ptiSparseMatrixSortIndexMorton(self.address,1,0,self.address.nnz,blockbits)
        else:
            print("[ERROR] Wrong sorting type.")

    # This helper functions separates a potential one time big data allocation from a repetitive method.
    def makeMap(self):
        self.map = pti.new("ptiIndex **")
        self.map = pti.cast("ptiIndex **", PTI.malloc(2 * pti.sizeof(self.map)))
        self.map[0] = pti.cast("ptiIndex *", PTI.malloc(self.address.nrows * pti.sizeof(pti.new("ptiIndex *"))))
        for i in range(self.address.nrows):
            self.map[0][i] = i
        self.map[1] = pti.cast("ptiIndex *", PTI.malloc(self.address.ncols * pti.sizeof(pti.new("ptiIndex *"))))
        for i in range(self.address.ncols):
            self.map[1][i] = i

    #this reorders your matrix to the best form
    #the type says what type of reordering to perform
    #   Lexi: Buts to diagonal
    #   Anything else: Random distribution
    #Other params are only necessary for Lexi, mostly the niters, number of cycles to move to diagonal
    def reorder(self,type="lexi",niters=5):
        if(self.map is None): self.makeMap()
        if(type=="lexi"):
            relabel = 1
            PTI.ptiIndexRelabel(self.address,self.map,relabel,niters,1)
        elif(type=="bfs"):
            relabel = 2
            PTI.ptiIndexRelabel(self.address,self.map,relabel,niters,1)
        elif(type=="random"):
            PTI.ptiGetRandomShuffledIndicesMat(self.address,self.map)
        else:
            print("[ERROR] Wrong reordering type.")
        PTI.ptiSparseMatrixShuffleIndices(self.address,self.map)


class HiCOOMatrix:
    def __init__(self):
        self.address = pti.new("ptiSparseMatrixHiCOO *")

    # Writes the matrix into a file, path specified in filename.
    # @:param filename, string path to the output file
    def dumpMatrix(self, filename):
        file = PTI.fopen(bytes(filename, 'ascii'), b'w')
        PTI.ptiDumpSparseMatrixHiCOO(self.address, file)

    # Output the brief information of the matrix
    def statusMatrix(self):
        PTI.ptiSparseMatrixStatusHiCOO(self.address,sys.stdout)

    # These get methods are to make an easy way for a user to get the instances of the structures in C
    # A user can still call self.address.<>, but we want to avoid that if possible.
    def ncols(self): return int(self.address.ncols)
    def nrows(self): return int(self.address.nrows)
    def nnz(self): return int(self.address.nnz)

    # This function multiplies the matrix by a vector
    # @:param vector, Value Vector of size ncols
    # @:return ValueVector, the resulting value vector of the multiplication is returned
    def multiplyVector(self, vector,type="default",testing=False):
        result = vec.ValueVector(self.address.nrows)
        size = self.address.nnz
        sp=size/self.address.ncols
        if (type == "default"):
            if (size < 1252512): PTI.ptiSparseMatrixMulVectorHiCOO(result.address,self.address,vector.address)
            elif (sp < 47.2): PTI.ptiOmpSparseMatrixMulVectorHiCOO(result.address,self.address,vector.address)
            else: PTI.ptiOmpSparseMatrixMulVectorHiCOOReduce(result.address,self.address,vector.address)
        elif (type == "serial"): PTI.ptiSparseMatrixMulVectorHiCOO(result.address,self.address,vector.address)
        elif (type == "CPU"): PTI.ptiOmpSparseMatrixMulVectorHiCOO(result.address,self.address,vector.address)
        elif (type == "CPU_Buff"):  PTI.ptiOmpSparseMatrixMulVectorHiCOOReduce(result.address,self.address,vector.address)
        if not testing: return result
        else: PTI.ptiFreeValueVector(result.address)


    # This function multiplies the matrix by a dense matrix
    # @:param mat, Dense Matrix where their nrows=self.ncols.
    # @:return Dense Matrix, the resulting product is returned.  Note of size self.nrows,mat.ncols
    def multiplyMatrix(self, mat, type="default", testing=False):
        result=DenseMatrix(self.address.nrows,mat.address.ncols)
        if type=="default":
            n = self.address.nnz
            r = mat.address.ncols
            s = self.address.nrows
            a = -0.2676 + 7.343E-5 * n / r / r / r - 17.91 / sqrt(r * s) + 5.729 / sqrt(r)
            b = sqrt(s*n)*r
            if a<1 or n<1000: PTI.ptiSparseMatrixMulMatrixHiCOO(result.address,self.address,mat.address)
            elif b<200000000: PTI.ptiOmpSparseMatrixMulMatrixHiCOO(result.address,self.address,mat.address)
            else: PTI.ptiOmpSparseMatrixMulMatrixHiCOOReduce(result.address,self.address,mat.address)
        elif type=="serial": PTI.ptiSparseMatrixMulMatrixHiCOO(result.address,self.address,mat.address)
        elif type=="CPU": PTI.ptiOmpSparseMatrixMulMatrixHiCOO(result.address,self.address,mat.address)
        elif type=="CPU_Buff": PTI.ptiOmpSparseMatrixMulMatrixHiCOOReduce(result.address,self.address,mat.address)
        if not testing: return result
        else: PTI.ptiFreeMatrix(result.address)

    def multiplyVectorBuff(self, vector, buffer,testing=False):
        result=vec.ValueVector(self.address.nrows)
        PTI.ptiOmpSparseMatrixMulVectorHiCOO_Schedule_Reduce(result.address,buffer.address,self.address,vector.address)
        if not testing: return result
        else: PTI.ptiFreeValueVector(result.address)

class CSRMatrix:
    def __init__(self):
        self.address=pti.new("ptiSparseMatrixCSR *")

    # Writes the matrix into a file, path specified in filename.
    # @:param filename, string path to the output file
    def dumpMatrix(self, filename):
        file=PTI.fopen(bytes(filename,'ascii'),b'w')
        PTI.ptiDumpSparseMatrixCSR(self.address,file)

    def statusMatrix(self):
        PTI.ptiSparseMatrixStatusCSR(self.address,sys.stdout)

    # These get methods are to make an easy way for a user to get the instances of the structures in C
    # A user can still call self.address.<>, but we want to avoid that if possible.
    def ncols(self): return int(self.address.ncols)
    def nrows(self): return int(self.address.nrows)
    def nnz(self): return int(self.address.nnz)

    # These get methods are to make an easy way for a user to get the instances of the structures in C
    # A user can still call self.address.<>, but we want to avoid that if possible.
    def ncols(self): return int(self.address.ncols)
    def nrows(self): return int(self.address.nrows)
    def nnz(self): return int(self.address.nnz)

    # This function multiplies the matrix by a vector
    # @:param vector, Value Vector of size ncols
    # @:return ValueVector, the resulting value vector of the multiplication is returned
    def multiplyVector(self,vector,type="default",testing=False):
        result = vec.ValueVector(self.address.nrows)
        size = self.address.nnz
        sp=size/self.address.ncols
        if (type == "default"):
            if (size < 6198): PTI.ptiSparseMatrixMulVectorCSR(result.address, self.address, vector.address)
            elif (sp<444.3): PTI.ptiOmpSparseMatrixMulVectorCSR(result.address, self.address, vector.address)
            else: PTI.ptiOmpSparseMatrixMulVectorCSRReduce(result.address, self.address, vector.address)
        elif (type == "serial"): PTI.ptiSparseMatrixMulVectorCSR(result.address, self.address, vector.address)
        elif (type == "CPU"): PTI.ptiOmpSparseMatrixMulVectorCSR(result.address, self.address, vector.address)
        elif (type == "CPU_Buff"):PTI.ptiOmpSparseMatrixMulVectorCSRReduce(result.address, self.address, vector.address)
        if not testing: return result
        else: PTI.ptiFreeValueVector(result.address)


    # This function multiplies the matrix by a dense matrix
    # @:param mat, Dense Matrix where their nrows=self.ncols.
    # @:return Dense Matrix, the resulting product is returned.  Note of size self.nrows,mat.ncols
    def multiplyMatrix(self, mat, type="default", testing=False):
        result=DenseMatrix(self.address.nrows,mat.address.ncols)
        n = self.address.nnz
        if type=="default":
            if n<1000: PTI.ptiSparseMatrixMulMatrixCSR(result.address,self.address,mat.address)
            else: PTI.ptiOmpSparseMatrixMulMatrixCSR(result.address,self.address,mat.address)
        elif type=="serial": PTI.ptiSparseMatrixMulMatrixCSR(result.address,self.address,mat.address)
        elif type=="CPU": PTI.ptiOmpSparseMatrixMulMatrixCSR(result.address,self.address,mat.address)
        elif type=="CPU_Buff": PTI.ptiOmpSparseMatrixMulMatrixCSRReduce(result.address,self.address,mat.address)
        if not testing: return result
        else: PTI.ptiFreeMatrix(result.address)

    def multiplyVectorBuff(self, vector, buffer,testing=False):
        result=vec.ValueVector(self.address.nrows)
        PTI.ptiOmpSparseMatrixMulVectorCSR_Reduce(result.address,buffer.address,self.address,vector.address)
        if not testing: return result
        else: PTI.ptiFreeValueVector(result.address)

class DenseMatrix:
    def __init__(self,nrows,ncols):
        self.address=pti.new("ptiMatrix *")
        PTI.ptiNewMatrix(self.address,nrows,ncols)
        PTI.ptiConstantMatrix(self.address,0)

    # Writes the matrix into a file, path specified in filename.
    # @:param filename, string path to the output file
    def dumpMatrix(self, filename):
        file=PTI.fopen(bytes(filename,'ascii'),b"w")
        PTI.ptiDumpMatrix(self.address,file)

    # These get methods are to make an easy way for a user to get the instances of the structures in C
    # A user can still call self.address.<>, but we want to avoid that if possible.
    def ncols(self): return int(self.address.ncols)
    def nrows(self): return int(self.address.nrows)

    # This method fills the matrix with a constant value, useful for resetting a matrix to zeros.
    # @:param value, number that represents the value to fill the matrix
    def constMatrix(self, value):
        PTI.ptiConstantMatrix(self.address,value)

    # This method set a specific location of a matrix with a specific value
    # @:param i, integer row number
    # @:param j, integer column number
    # @:param value, number of the value to be stored
    def setValue(self, i, j, value):
        self.address.values[i*self.address.stride+j]=value;

    def randomMatrix(self):
        PTI.ptiRandomizeMatrix(self.address)

    def get(self,i,j):\
        return self.address.values[i*self.address.stride+j]
