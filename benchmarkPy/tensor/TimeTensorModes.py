import HiParTIPy.Tensor as ten
import HiParTIPy.Matrix as mat
import HiParTIPy.Vector as vec
import sys
from time import time

name = sys.argv[1]

coo=ten.COOTensor()
coo.load(name)

print("File Name: {}".format(name))

niters=int(sys.argv[2])
equ=int(niters/100)
mode = int(sys.argv[3])

vector=vec.ValueVector(coo.address.ndims[mode])

for i in range(equ): coo.mulVector(vector,mode,testing=True)
start = time()
for i in range(niters): coo.mulVector(vector,mode,testing=True)
end=time()
print("MulVector: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

var=sys.argv[4]

matrix = mat.DenseMatrix(coo.address.ndims[mode],var)

for i in range(equ): coo.mulMatrix(matrix,mode,testing=True)
start = time()
for i in range(niters): coo.mulMatrix(matrix,mode,testing=True)
end=time()
print("MulMatrix: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

