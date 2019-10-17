import clpy
import numpy

gen = clpy.backend.opencl.random.createGenerator()
print("reveal:", gen.reveal())

arr = clpy.asarray(range(10), dtype=numpy.float32)
print(arr)

clpy.backend.opencl.random.generateUniform(gen, arr)
print(arr)
print("reveal:", gen.reveal())


arr = clpy.asarray(range(30), dtype=numpy.float64)
clpy.backend.opencl.random.generateUniformDouble(gen, arr)
print(arr)
print("reveal:", gen.reveal())
