from clpy.core.core cimport ndarray

cdef class clrandGenerator:
    cdef ndarray a
    cdef ndarray b
    cdef ndarray c
    cdef ndarray d
    cdef ndarray counter
    cdef int inner_state_size


cpdef clrandGenerator createGenerator()
cpdef setPseudoRandomGeneratorSeed(clrandGenerator generator, seed)

cpdef generateUniform(clrandGenerator, ptr, size)
cpdef generateUniformDouble(clrandGenerator, ptr, size)

cpdef destroyGenerator(clrandGenerator generator)
