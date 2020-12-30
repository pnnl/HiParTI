from time import time
import sys
sys.path.insert(0, "./")
import HiParTIPy.Tensor as ptiTen
import HiParTIPy.Matrix as ptiMat

if len(sys.argv) < 2:
  sys.exit("Usage: python3 benchmarkPy/tensor/mttkrp.py [.tns] [mode] [rank] [niters]")

name = sys.argv[1]
mode = sys.argv[2]
R=int(sys.argv[3])
niters=int(sys.argv[4])
print("Load Tensor from: {}".format(name))

coo=ptiTen.COOTensor()
coo.load(name)

nmodes = coo.nmodes()
ndims_mode = coo.dim_mode(mode)

input_mat = ptiMat.DenseMatrix(ndims_mode, R)
input_mat.randomMatrix()

coo.mulMatrix(mats,mode,type='serial', testing=True)
start = time()
for i in range(niters): 
	coo.mulMatrix(mats,mode,type='serial', testing=True)
end=time()
print("TTM: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

hicoo=coo.toHiCOO()