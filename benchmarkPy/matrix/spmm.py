import time as time
import sys
sys.path.insert(0, "./")
import HiParTIPy.Matrix as ptiMat

if len(sys.argv) < 2:
  sys.exit("Usage: python3 benchmarkPy/matrix/spmm.py [.mtx] [ncols] [niters]")

mtxname = sys.argv[1]
print("Load Matrix from: {}".format(mtxname))
R = int(sys.argv[2])  # Specify number of columns for the dense matrix
if R==0: R=ncols
niters = int(sys.argv[3])  # Specify number of iterations

coo = ptiMat.COOMatrix()
coo.loadMatrix(mtxname)  # Specify what matrix to use
sb_bits = 7
hicoo = coo.convert(type="hicoo", blockbits=sb_bits, superblockbits=sb_bits)  # Specify blocking for hicoo matricies
csr = coo.convert(type="csr")

ncols = coo.ncols()
nrows = coo.nrows()
nnz = coo.nnz()

input_mat = ptiMat.DenseMatrix(ncols, R)
input_mat.randomMatrix()

######## COO-SpMM #########
# warm-up
coo.multiplyMatrix(input_mat,testing=True)

start = time.time()
for i in range(niters):
    coo.multiplyMatrix(input_mat,testing=True)
end = time.time()

print("COO Matrix: \nTime: {} sec\nniters: {}\nTime per Call: {} sec/cycle\n"
      .format((end - start), niters, (end - start) / niters))

######## HiCOO-SpMM #########
# warm-up
hicoo.multiplyMatrix(input_mat,testing=True)

start = time.time()
for i in range(niters):
    hicoo.multiplyMatrix(input_mat,testing=True)
end = time.time()

print("HiCOO Matrix: \nTime: {} sec\nniters: {}\nTime per Call: {} sec/cycle\n"
      .format((end - start), niters, (end - start) / niters))

######## CSR-SpMM #########
# warm-up
csr.multiplyMatrix(input_mat,testing=True)

start = time.time()
for i in range(niters):
    csr.multiplyMatrix(input_mat,testing=True)
end = time.time()

print("CSR Matrix: \nTime: {} sec\nniters: {}\nTime per Call: {} sec/cycle\n"
      .format((end - start), niters, (end - start) / niters))

