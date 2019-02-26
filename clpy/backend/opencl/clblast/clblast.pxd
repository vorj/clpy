include "../types.pxi"
include "clblast.pxi"

cdef void clblast_sgemm(
    CLBlastLayout layout,
    CLBlastTranspose a_transpose,
    CLBlastTranspose b_transpose,
    size_t m, size_t n, size_t k,
    float alpha,
    cl_mem a_buffer, size_t a_offset, size_t a_ld,
    cl_mem b_buffer, size_t b_offset, size_t b_ld,
    float beta,
    cl_mem c_buffer, size_t c_offset, size_t c_ld) except *

cpdef sgemm(str_layout, t_a_transpose, t_b_transpose,
            m, n, k, alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc)

cdef void clblast_dgemm(
    CLBlastLayout layout,
    CLBlastTranspose a_transpose,
    CLBlastTranspose b_transpose,
    size_t m, size_t n, size_t k,
    double alpha,
    cl_mem a_buffer, size_t a_offset, size_t a_ld,
    cl_mem b_buffer, size_t b_offset, size_t b_ld,
    double beta,
    cl_mem c_buffer, size_t c_offset, size_t c_ld) except *

cpdef dgemm(str_layout, t_a_transpose, t_b_transpose,
            m, n, k, alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc)


cdef void clblast_sgemm_batched(
    CLBlastLayout layout,
    CLBlastTranspose a_transpose,
    CLBlastTranspose b_transpose,
    size_t m,
    size_t n,
    size_t k,
    float *alphas,
    cl_mem a_buffer,
    size_t *a_offsets,
    size_t a_ld,
    cl_mem b_buffer,
    size_t *b_offsets,
    size_t b_ld,
    float *betas,
    cl_mem c_buffer,
    size_t *c_offsets,
    size_t c_ld,
    size_t batch_count) except *
cpdef sgemm_batched(str_layout, t_a_transpose, t_b_transpose,
            m, n, k,
            alpha, # CuPy's api takes single alpha
            A, offsets_a, lda,
            B, offsets_b, ldb,
            beta,  # same as above
            C, offsets_c, ldc,
            batch_count)

cdef void clblast_dgemm_batched(
    CLBlastLayout layout,
    CLBlastTranspose a_transpose,
    CLBlastTranspose b_transpose,
    size_t m,
    size_t n,
    size_t k,
    double *alphas,
    cl_mem a_buffer,
    size_t *a_offsets,
    size_t a_ld,
    cl_mem b_buffer,
    size_t *b_offsets,
    size_t b_ld,
    double *betas,
    cl_mem c_buffer,
    size_t *c_offsets,
    size_t c_ld,
    size_t batch_count) except *
cpdef dgemm_batched(str_layout, t_a_transpose, t_b_transpose,
            m, n, k,
            alpha, # CuPy's api takes single alpha
            A, offsets_a, lda,
            B, offsets_b, ldb,
            beta,  # same as above
            C, offsets_c, ldc,
            batch_count)
