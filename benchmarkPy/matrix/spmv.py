import time as time
import sys
sys.path.insert(0, "./")
import HiParTIPy.Matrix as ptiMat
import HiParTIPy.Vector as ptiVec

if len(sys.argv) < 2:
  sys.exit("Usage: python3 benchmarkPy/matrix/spmv.py [.mtx] [niters]")

mtxname = sys.argv[1]
print("Load Matrix from: {}".format(mtxname))
niters = int(sys.argv[2])  # Specify number of iterations

coo = ptiMat.COOMatrix()

coo.loadMatrix(mtxname)  # Specify what matrix to use.
sb_bits = 7
hicoo = coo.convert(type="hicoo", blockbits=sb_bits, superblockbits=sb_bits)  # Specify blocking for hicoo matricies
csr = coo.convert(type="csr")

ncols = coo.ncols()
nrows = coo.nrows()
nnz = coo.nnz()

input_vector = ptiVec.ValueVector(ncols)
input_vector.makeRandom()

######## COO-SpMV #########
# warm-up
coo.multiplyVector(input_vector,testing=True)

start = time.time()
for i in range(niters):
    coo.multiplyVector(input_vector,testing=True)
end = time.time()

coo.statusMatrix()
print("Time: {} sec\nniters: {}\nTime per Call: {} sec/cycle\n"
      .format((end - start), niters, (end - start) / niters))


######## HiCOO-SpMV #########
# warm-up
hicoo.multiplyVector(input_vector,testing=True)

start = time.time()
for i in range(niters):
    hicoo.multiplyVector(input_vector,testing=True)
end = time.time()

print("HiCOO Matrix:")
hicoo.statusMatrix()
print("Time: {} sec\nniters: {}\nTime per Call: {} sec/cycle\n"
      .format((end - start), niters, (end - start) / niters))


######## CSR-SpMV #########
# warm-up
csr.multiplyVector(input_vector,testing=True)

start = time.time()
for i in range(niters):
    csr.multiplyVector(input_vector,testing=True)
end = time.time()

print("CSR Matrix:")
csr.statusMatrix()
print("Time: {} sec\nniters: {}\nTime per Call: {} sec/cycle\n"
      .format((end - start), niters, (end - start) / niters))



