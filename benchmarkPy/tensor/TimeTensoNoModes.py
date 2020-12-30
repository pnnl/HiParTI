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

for i in range(equ): coo.mulValue(i/equ)
start = time()
for i in range(niters): coo.mulValue(i/equ)
end=time()
print("MulScalar: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

for i in range(equ): coo.divValue(i/equ+1)
start = time()
for i in range(niters): coo.divValue(i/equ+1)
end=time()
print("DivScalar: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

for i in range(equ): coo.addTensor(coo)
start = time()
for i in range(niters): coo.addTensor(coo)
end=time()
print("AddTensor: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

for i in range(equ): coo.subtractTensor(coo)
start = time()
for i in range(niters): coo.subtractTensor(coo)
end=time()
print("Subtract Tensor: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

for i in range(equ): coo.dotMulTensor(coo)
start = time()
for i in range(niters): coo.dotMulTensor(coo)
end=time()
print("DotMul: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

for i in range(equ): coo.dotDivTensor(coo)
start = time()
for i in range(niters): coo.dotDivTensor(coo)
end=time()
print("DotDiv: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))