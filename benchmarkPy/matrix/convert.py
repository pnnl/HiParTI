import time as time
import sys
sys.path.insert(0, "./")
import HiParTIPy.Matrix as ptiMat

if len(sys.argv) < 2:
	sys.exit("Usage: python3 benchmarkPy/matrix/convert.py [.mtx]")

mtxname = sys.argv[1]
print("Load Matrix from: {}".format(mtxname))

coo_in = ptiMat.COOMatrix()
coo_in.loadMatrix(mtxname)
coo_in.statusMatrix()

csr_mat = coo_in.convert(type="csr")
csr_mat.statusMatrix()

sb_bits = 7 # block size = 2^sb_bits
hicoo_mat = coo_in.convert(type="hicoo", blockbits=sb_bits, superblockbits=sb_bits)
hicoo_mat.statusMatrix()
