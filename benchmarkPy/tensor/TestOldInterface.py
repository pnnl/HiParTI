import HiParTIPy.Matrix as mat
import HiParTIPy.Vector as vec
import HiParTIPy.Buffers as buff

coo = mat.COOMatrix()
coo.loadMatrix("data/matrices/tiny.mtx")
hicoo = coo.convertToHiCOO()
print(coo.ncols())

vec = vec.ValueVector(coo.ncols())
vec.makeRandom()

buf = buff.VecBuff(coo.ncols())

print("HERE")
coo.multiplyVector(vec)
print("HREE")
hicoo.multiplyVectorBuff(vec,buf)
print("HERE")
buf.free()
print("HERE")