import HiParTIPy.Tensor as ten
import HiParTIPy.Matrix as mat
import HiParTIPy.Vector as vec
import HiParTIPy.Buffers as buff

coo3d=ten.COOTensor()
coo3d.load("data/tensors/3d-24.tns")
print("HERE")
coo3d.mulValue(4)
coo3d.divValue(2)
coo3d2=ten.COOTensor()
print("HERE")
coo3d2.load("data/tensors/3d-24.tns")
coo3d.addTensor(coo3d2,1)
print("HERE")
coo3d.subtractTensor(coo3d,1)
coo3d.addTensor(coo3d2,8)
coo3d.mulValue(4)
coo3d.subtractTensor(coo3d2,10)
print("HERE")