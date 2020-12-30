import time as time
import sys
sys.path.insert(0, "./")
import HiParTIPy.Matrix as ptiMat

if len(sys.argv) < 2:
	sys.exit("Usage: python3 benchmarkPy/matrix/sort.py [.mtx]")

mtxname = sys.argv[1]
print("Load Matrix from: {}".format(mtxname))

coo_in = ptiMat.COOMatrix()
coo_in.loadMatrix(mtxname)
coo_in.statusMatrix()

# Soring in row->column order
start=time.time()
coo_in.sort(type="row")
end=time.time()
print("RowSort: {}".format(end-start))

# Soring in column->row order
coo_in.sort(type="col")

sb_bits = 2 # block size = 2^sb_bits
# Soring in natural block order
coo_in.sort(type="block", blockbits=sb_bits)

# Soring in Z-Morton order
coo_in.sort(type="morton", blockbits=sb_bits)