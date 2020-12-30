from HiParTIPy.PtiCffi import pti,PTI

import HiParTIPy.Matrix as mat
import os as os

class COOTensor:
    def __init__(self):
        self.address=pti.new("ptiSparseTensor *")
        self.dimentions=0
        self.nthreads = (int)(os.popen('grep -c cores /proc/cpuinfo').read())
    
    def dim_mode(self, mode): return int(self.address.ndims[mode])
    def nmodes(self): return int(self.address.nmodes)
    def nmodes(self): return int(self.address.nnz)

    def load(self,filename):
        file = bytes(filename, 'ascii')
        PTI.ptiLoadSparseTensor(self.address,1,file)
        self.dimentions=self.address.nmodes

    def dump(self,filename):
        file = PTI.fopen(bytes(filename, 'ascii'), b"w")
        PTI.ptiDumpSparseTensor(self.address,1,file)

    def free(self):
        PTI.ptiFreeSparseTensor(self.address)

    def toSemiSparse(self,mode=0):
        result=sCOOTensor()
        PTI.ptiSparseTensorToSemiSparseTensor(self.address,result.address,mode)

    def toHiCOO(self,b=7,k=7,c=7):
        result=HiCOOTensor()
        PTI.ptiSparseTensorToHiCOO(result.address,0,self.address,b,k,c,1)

    def mulValue(self,scalar):
        PTI.ptiSparseTensorMulScalar(self.address,scalar)
    def divValue(self,scalar):
        PTI.ptiSparseTensorDivScalar(self.address,scalar)
    def mulVector(self,vector,mode,testing=False):
        result=sCOOTensor()
        PTI.ptiSparseTensorMulVector(result.address,self.address,vector.address,mode)
        if not testing: return result
        else: result.free()

    def mulMatrix(self,matrix,mode,type="default",testing=False):
        result=sCOOTensor()
        if type=="default":
            print("Default Cases not learned yet.  Make sure you spedify your run type.")
        if type=="serial": PTI.ptiSparseTensorMulMatrix(result.address,self.address,matrix.address,mode)
        elif type=="CPU": PTI.ptiOmpSparseTensorMulMatrix(result.address,self.address.matrix.address,mode)
        elif type=="GPU": PTI.ptiCudaSparseTensorMulMatrix(result.address,self.address,matrix.address,mode)
        elif type=="GPU-1K": PTI.ptiCudaSparseTensorMulMatrixOneKernal(result.address,self.address,matrix.address,mode)#TODO Learn more about rest of params (impl_num and smen_size)
        else: exit("Invalid Type")
        if not testing: return result
        else: result.free()

    def addTensor(self,tensor,nthreads=0,testing=False):
        if nthreads==0:
            print("Default Cases not learned yet.  Make sure you spedify your run type.")
        if nthreads == 1: type="serial"
        else: type = "CPU"
        if type=="serial":
            result=COOTensor()
            PTI.ptiSparseTensorAdd(result.address,self.address,tensor.address)
            if not testing: return result
            else: PTI.ptiFreeSparseTensor(result.address)
        elif type=="CPU":
            if nthreads==0: nthreads=self.nthreads
            PTI.ptiSparseTensorAddOMP(tensor.address,self.address,nthreads)
            if not testing: return tensor
        else: exit("Invalid Type")

    def subtractTensor(self,tensor,nthreads=0,testing=False):
        if nthreads == 0:
            print("Default Cases not learned yet.  Make sure you spedify your run type.")
        if nthreads == 1: type = "serial"
        else: type = "CPU"
        if type=="serial":
            result=COOTensor()
            PTI.ptiSparseTensorSub(result.address,self.address,tensor.address)
            if not testing: return result
            else: PTI.ptiFreeSparseTensor(result.address)
        elif type=="CPU":
            if nthreads==0: nthreads=self.nthreads
            PTI.ptiSparseTensorSubOMP(tensor.address,self.address,nthreads)
            if not testing: return tensor
        else: exit("Invalid Type")

    def dotMulTensor(self,tensor,type="default",testing=False):
        result=COOTensor()
        if type=="default":
            print("Default Cases not learned yet.  Make sure you spedify your run type.")
        if type=="serial": PTI.ptiSparseTensorDotMul(result.address,self.address,tensor.address)
        elif type=="serial_EQ": PTI.ptiSparseTensorDotMulEq(result.address,self.address,tensor.address)
        elif type=="CPU": PTI.ptiOmpSparseTensorDotMulEq(result.address,self.address,tensor.address)
        elif type=="GPU": PTI.ptiCudaSparseTensorDotDiv(result.address,self.address,tensor.address)
        else: exit("Invalid Type")
        if not testing: return result
        else: PTI.ptiFreeSparseTensor(result.address)

    def dotDivTensor(self,tensor,type="default",testing=False):
        result=COOTensor()
        PTI.ptiSparseTensorDotDiv(result.address,self.address,tensor.address)
        if not testing: return result
        else: PTI.ptiFreeSparseTensor(result.address)

    def crossMulTensor(self,tensor,mode,modex,modey,type="default",testing=False):
        print("This method is actually not implemented #SAD")
        return None
        result = COOTensor()
        PTI.ptiSparseTensorMulTensor(result.address,self.address,tensor.address,mode,modex,modey)
        return result

    def Kronecker(self,tensor, testing=False):
        result=COOTensor()
        PTI.ptiSparseTensorKroneckerMul(result.address,self.address,tensor.address)
        if not testing: return result
        else: result.free()

    def KhatriRao(self,tensor, testing=False):
        result=COOTensor()
        PTI.ptiSparseTensorKhatriRaoMul(result.address,self.address,tensor.address)
        if not testing: return result
        else: result.free()

    #MTTKRP Methods
    def MTTKRP(self, mode, mats=None, matsorder=None, type="default",testing=False):
        if mats==None:
            serial=0#TODO make code to make mats inside here

        if type=="default":
            print("Default Cases not learned yet.  Make sure you spedify your run type.")

        if type=="serial": PTI.ptiMTTKRPHiCOO(self.address,mats.address,matsorder,mode)
        elif type=="CPU": PTI.ptiOMPMTTKRPHiCOO(self.address,mats.address,matsorder,mode)
        elif type=="CPU_Reduce": PTI.ptiOmpMTTKRP_Reduce()
        elif type=="CPU_Lock": PTI.ptiOmpMTTKRP_Lock()
        elif type=="CUDA": PTI.ptiCudaMTTKRP()
        elif type=="CUDA_1K": PTI.ptiCudaMTTKRPOneKernal()
        elif type=="CUDA_SM": PTI.ptiCudaMTTKRPSM()
        elif type=="CUDA_Device": PTI.ptiCudaMTTKRPDevice()
        elif type=="CUDA_Coarse": PTI.ptiCudaCoareMTTKRP()
        elif type=="Splitted": PTI.ptiSplittedMTTKRP()
        if not testing: return mats

    def MTTKRPReduce(self,mode,mats,mats_copy,mats_order):

        return mats


class HiCOOTensor:
    def __init__(self):
        self.address=pti.new("ptiSparseTensorHiCOO *")

    def dump(self,filename):
        file = PTI.fopen(bytes(filename, 'ascii'), b"r")
        PTI.ptiDumpSparseTensorHiCOO(self.address,file)


    #MTTKRP Methods
    def MTTKRP(self, mode, mats=None, matsorder=None, type="default", testing=False):
        if mats == None:
            serial = 0  # TODO make code to make mats inside here

        if type == "default":
            serial = 0  # TODO make params

        if type == "serial": PTI.ptiMTTKRP(self.address, mats.address, matsorder, mode)
        if type == "CPU": PTI.ptiOMPMTTKRP(self.address, mats.address, matsorder, mode)
        if type == "CPU_Reduce": PTI.ptiOmpMTTKRP_Reduce()
        if type == "CPU_Lock": PTI.ptiOmpMTTKRP_Lock()
        if type == "CUDA": PTI.ptiCudaMTTKRP()
        if type == "CUDA_1K": PTI.ptiCudaMTTKRPOneKernal()
        if type == "CUDA_SM": PTI.ptiCudaMTTKRPSM()
        if type == "CUDA_Device": PTI.ptiCudaMTTKRPDevice()
        if type == "CUDA_Coarse": PTI.ptiCudaCoareMTTKRP()
        if type == "Splitted": PTI.ptiSplittedMTTKRP()
        return mats

    def MTTKRPReduce(self, mode, mats, mats_copy, mats_order, ):

        return mats

class sCOOTensor:
    def __init__(self):
        self.address=pti.new("ptiSemiSparseTensor *")

    def free(self):
        PTI.ptiFreeSemiSparseTensor(self.address)

    def toCOO(self):
        result=COOTensor()
        PTI.ptiSemiSparseTensorToSparseTensor(result.address,self.address,1e-6)

    def mulMatrix(self,tensor,mode,type="default",testing=False):
        result=sCOOTensor()
        if type=="default":
            serial=0 #TODO make params
        if type=="serial": PTI.ptiSemiSparseTensorMulMatrix(result.address,self.address,tensor.address,mode)
        elif type=="GPU": PTI.ptiCudaSemiSparseTensorMulMatrix(result.address,self.address,tensor.address,mode)
        if not testing: return result
        else: PTI.ptiFreeSemiSparseTensor(result.address)

class gCOOTensor:
    def __init__(self):
        self.address=pti.new("ptiSemiSparseTensorGeneral *")