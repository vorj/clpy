include "../types.pxi"
include "clblast.pxi"

cdef void clblast_sgemm(layout, a_transpose, b_transpose,
                   m, n, k,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   b_buffer, b_offset, b_ld,
                   beta,
                   c_buffer, c_offset, c_ld)
cpdef sgemm(str_layout, t_a_transpose, t_b_transpose,
            m, n, k, alpha,
	    A, lda,
            B, ldb,
            beta,
	    C, ldc)
