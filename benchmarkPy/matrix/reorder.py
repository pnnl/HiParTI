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

# Reordering in Lexi-order
coo_in.reorder(type="lexi")

# Reordering in random order
coo_in.reorder(type="random")