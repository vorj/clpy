cimport clpy.backend.opencl.api as api
import clpy.backend.opencl.env
cimport clpy.backend.opencl.env
import clpy.backend.opencl.types
cimport clpy.backend.opencl.types
from clpy.backend.opencl.types cimport *

cdef void clblast_sgemm(CLBlastLayout layout, CLBlastTranspose a_transpose, CLBlastTranspose b_transpose,
                   size_t m, size_t n, size_t k,
                   float alpha,
                   cl_mem a_buffer, size_t a_offset, size_t a_ld,
                   cl_mem b_buffer, size_t b_offset, size_t b_ld,
                   float beta,
                   cl_mem c_buffer, size_t c_offset, size_t c_ld) except *:
    cdef cl_event event = NULL
    cdef cl_command_queue command_queue=clpy.backend.opencl.env.get_command_queue()

    cdef CLBlastStatusCode status = CLBlastSgemm(
        layout, a_transpose, b_transpose,
        m, n, k,
        alpha,
        a_buffer, a_offset, a_ld,
        b_buffer, b_offset, b_ld,
        beta,
        c_buffer, c_offset, c_ld,
        &command_queue,
        &event
        )
    if (status == CLBlastSuccess):
        api.WaitForEvents(1, &event)
	# TODO api.ReleaseEvent should be implemented
        # api.ReleaseEvent
    return

cpdef sgemm(str_layout, transa, transb,
            m, n, k, alpha,
            A, lda,
            B, ldb,
            beta,
	    C, ldc):
    cdef CLBlastLayout layout
    cdef a_transpose
    cdef b_transpose
    if (str_layout == 'R'):
        layout = CLBlastLayoutRowMajor
    elif (str_layout == 'C'):
        layout = CLBlastLayoutColMajor
    else:
        raise ValueError("layout should be \'R\' or \'c\'")

    if (transa == 'n' or transa == 0):
        a_transpose = CLBlastTransposeNo
    elif (transa == 't' or transa == 1):
        a_transpose = CLBlastTransposeYes
    else:
        raise ValueError("transa should be n(0) or t(1)")

    if (transb == 'n' or transb == 0):
        b_transpose = CLBlastTransposeNo
    elif (transb == 't' or transb == 1):
        b_transpose = CLBlastTransposeYes
    else:
        raise ValueError("transb should be n(0) or t(1)")

    cdef size_t a_buffer = A.data.buf.get()
    cdef size_t b_buffer = B.data.buf.get()
    cdef size_t c_buffer = C.data.buf.get()

    clblast_sgemm(
        layout,
        a_transpose,
        b_transpose,
        m, n, k,
        alpha,
        <cl_mem>a_buffer, A.data.cl_mem_offset() // A.itemsize, lda,
        <cl_mem>b_buffer, B.data.cl_mem_offset() // B.itemsize, ldb,
        beta,
        <cl_mem>c_buffer, C.data.cl_mem_offset() // C.itemsize, ldc)
