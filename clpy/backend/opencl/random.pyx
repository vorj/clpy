import numpy
import clpy

import math

xorwow_src = """
ulong xorwow(
    __global ulong* a,
    __global ulong* b,
    __global ulong* c,
    __global ulong* d,
    __global ulong* counter
){

    ulong t = *d;
    ulong const s = *a;
    *d = *c;
    *c = *b;
    *b = s;

    t ^= t >> 2;
    t ^= t << 1;
    t ^= s ^ ( s << 4 );
    *a = t;

    *counter += 362437;

    return t + *counter;
}

"""

expand_kernel_src = xorwow_src + """
__kernel void clpy_expand_inner_state_array(
    CArray<ulong, 1> a,
    CArray<ulong, 1> b,
    CArray<ulong, 1> c,
    CArray<ulong, 1> d,
    CArray<ulong, 1> counter,
    ulong stride
){
    size_t const id = get_global_id(0);

    size_t const base = id * stride * 2;
    size_t const target = base + stride;

    a[target] = xorwow( &a[base], &b[base], &c[base], &d[base], &counter[base] );
    b[target] = xorwow( &a[base], &b[base], &c[base], &d[base], &counter[base] );
    c[target] = xorwow( &a[base], &b[base], &c[base], &d[base], &counter[base] );
    d[target] = xorwow( &a[base], &b[base], &c[base], &d[base], &counter[base] );
    counter[target] = xorwow( &a[base], &b[base], &c[base], &d[base], &counter[base] );
}
"""

expand_module = clpy.core.core.compile_with_cache(expand_kernel_src)
expand_function = expand_module.get_function("clpy_expand_inner_state_array")


roll_kernel_src = xorwow_src + """
__kernel void clpy_rng_roll(
    CArray<ulong, 1> a,
    CArray<ulong, 1> b,
    CArray<ulong, 1> c,
    CArray<ulong, 1> d,
    CArray<ulong, 1> counter,
    CArray<ulong, 1> output
){
    size_t const id = get_global_id(0);
    output[id] = xorwow( &a[id], &b[id], &c[id], &d[id], &counter[id] );
}
"""

roll_module = clpy.core.core.compile_with_cache(roll_kernel_src)
roll_function = roll_module.get_function("clpy_rng_roll")

cdef class clrandGenerator:

    def __init__(self):
        numpy.random.seed(0)
        self.a = clpy.asarray([numpy.random.random_integers(1e10)], dtype=numpy.uint64)
        self.b = clpy.asarray([numpy.random.random_integers(1e10)], dtype=numpy.uint64)
        self.c = clpy.asarray([numpy.random.random_integers(1e10)], dtype=numpy.uint64)
        self.d = clpy.asarray([numpy.random.random_integers(1e10)], dtype=numpy.uint64)
        self.counter = clpy.asarray([numpy.random.random_integers(1e10)], dtype=numpy.uint64)
        self.inner_state_size = 1

    def seed(self, seed):
        # seed: numpy.ndarray (uint64)
        numpy.random.seed(seed[0])
        self.a = clpy.asarray([numpy.random.random_integers(1e10)], dtype=numpy.uint64)
        self.b = clpy.asarray([numpy.random.random_integers(1e10)], dtype=numpy.uint64)
        self.c = clpy.asarray([numpy.random.random_integers(1e10)], dtype=numpy.uint64)
        self.d = clpy.asarray([numpy.random.random_integers(1e10)], dtype=numpy.uint64)
        self.counter = clpy.asarray([numpy.random.random_integers(1e10)], dtype=numpy.uint64)
        self.inner_state_size = 1

    def expand(self, size):
        if self.inner_state_size >= size:
            return
        old_inner_state_size = self.inner_state_size
        new_inner_state_size = 2 ** math.ceil(math.log2(size))  # 2べきに繰り上げ

        new_a = clpy.empty((new_inner_state_size,), dtype=numpy.uint64)
        new_a[0:old_inner_state_size] = self.a
        new_b = clpy.empty((new_inner_state_size,), dtype=numpy.uint64)
        new_b[0:old_inner_state_size] = self.b
        new_c = clpy.empty((new_inner_state_size,), dtype=numpy.uint64)
        new_c[0:old_inner_state_size] = self.c
        new_d = clpy.empty((new_inner_state_size,), dtype=numpy.uint64)
        new_d[0:old_inner_state_size] = self.d
        new_counter = clpy.empty((new_inner_state_size,), dtype=numpy.uint64)
        new_counter[0:old_inner_state_size] = self.counter

        self.a = new_a
        self.b = new_b
        self.c = new_c
        self.d = new_d
        self.counter = new_counter

        stride = new_inner_state_size // 2
        while stride >= 1:
            expand_function(
                    global_work_size=(new_inner_state_size // (stride*2) ,),
                    local_work_size=(1,),
                    args=( self.a, self.b, self.c, self.d, self.counter, stride ),
                    local_mem=0)
            stride = stride // 2

        self.inner_state_size = new_inner_state_size


    def roll(self):
        output = clpy.empty_like(self.a)
        roll_function(
                global_work_size=(output.size,),
                local_work_size=(1,),
                args=( self.a, self.b, self.c, self.d, self.counter, output ),
                local_mem=0)
        return output

    def reveal(self):
        return self.a, self.b, self.c, self.d


cpdef clrandGenerator createGenerator():
    return clrandGenerator()

cpdef setPseudoRandomGeneratorSeed(clrandGenerator generator, seed):
    generator.seed(seed)


u64_shrinkto_fp = clpy.core.core.ElementwiseKernel(
    '', 'T in, U out',
    '''
    out = (U)(in)/(U)(0xFFFFFFFFFFFFFFFF);
    ''',
    'clpy_u64_shrinkto_fp'
)

cpdef generateUniform(clrandGenerator generator, output_array, size):
    #todo size不要?
    generator.expand(size)
    state = generator.roll()
    state_view = state[0:size]
    u64_shrinkto_fp(state_view, output_array)
    # フラットのまま返す

cpdef generateUniformDouble(clrandGenerator generator, output_array, size):
    # ptr でなく clpy.ndarray を受け付けられるので、型の場合分けが不要
    generateUniform(generator, output_array, size)

cpdef destroyGenerator(clrandGenerator generator):
    pass
