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

var=sys.argv[4]

matrix = mat.DenseMatrix(coo.address.ndims[mode],var)
matrix.randomMatrix()

for i in range(equ): coo.mulMatrix(matrix,mode,type="GPU",testing=True)
start = time()
for i in range(niters): coo.mulMatrix(matrix,mode,type="GPU",testing=True)
end=time()
print("MulMatrix: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

