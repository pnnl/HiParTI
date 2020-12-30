import HiParTIPy.Tensor as ten
import sys
from time import time

name = sys.argv[1]

coo=ten.COOTensor()
coo.load(name)

print("File Name: {}".format(name))

niters=int(sys.argv[2])
equ=int(niters/100)

for i in range(equ): coo.Kronecker(coo)
start = time()
for i in range(niters): coo.Kronecker(coo)
end=time()
print("Kronecker: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

niters*=1

for i in range(equ): coo.KhatriRao(coo)
start = time()
for i in range(niters): coo.KhatriRao(coo)
end=time()
print("KhatriRao: Time = {},\n Time per Cycles = {}".format(end-start,(end-start)/niters))

