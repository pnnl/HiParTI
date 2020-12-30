import time as time
import sys
sys.path.insert(0, "./")
import HiParTIPy.Tensor as ptiTen

if len(sys.argv) < 2:
	sys.exit("Usage: python3 benchmarkPy/matrix/loadstore.py [.tns] [.tns]")

tnsname = sys.argv[1]
outname = sys.argv[2]
print("Load Tensor from: {}".format(tnsname))

coo_in = ptiTen.COOTensor()
coo_in.load(tnsname)
# coo_in.status()

print("Store Tensor to: {}".format(outname))
coo_in.dump(outname)