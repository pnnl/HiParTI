from HiParTIPy.PtiCffi import pti,PTI
import os

class VecBuff:
    def __init__(self,size):
        self.nthreads = (int)(os.popen('grep -c cores /proc/cpuinfo').read())
        print(self.nthreads)
        self.address = pti.cast("ptiValueVector *", PTI.malloc(self.nthreads * pti.sizeof(pti.new("ptiValueVector *"))))
        PTI.ptiMakeVectorBuff(self.address, size)

    def free(self):
        PTI.ptiFreeVecBuff(self.address)


class MatBuff:
    def __init__(self,ncols,nrows):
        self.nthreads = (int)(os.popen('grep -c cores /proc/cpuinfo').read())
        print(self.nthreads)
        self.address = pti.cast("ptiMatrix *", PTI.malloc(self.nthreads * pti.sizeof(pti.new("ptiMatrix *"))))
        PTI.ptiMakeMatrixBuff(self.address, ncols,nrows)
