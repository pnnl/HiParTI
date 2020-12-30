import time as time
import sys
sys.path.insert(0, "./")
import HiParTIPy.Matrix as ptiMat

if len(sys.argv) < 2:
	sys.exit("Usage: python3 benchmarkPy/matrix/loadstore_mat.py [.mtx] [.txt]")

mtxname = sys.argv[1]
outname = sys.argv[2]
print("Load Matrix from: {}".format(mtxname))

coo_in = ptiMat.COOMatrix()
coo_in.loadMatrix(mtxname)
coo_in.statusMatrix()

print("Store Matrix to: {}".format(outname))
coo_in.dumpMatrix(outname)	# TODO: a bug in store