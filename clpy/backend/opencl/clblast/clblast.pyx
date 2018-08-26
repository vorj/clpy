include "clblast.pxi"

cimport clpy.backend.opencl.api as api
import clpy.backend.opencl.env
cimport clpy.backend.opencl.env
import clpy.backend.opencl.types
cimport clpy.backend.opencl.types
from clpy.backend.opencl.types cimport cl_event

cdef void clblast_sgemm(layout, a_transpose, b_transpose,
                   m, n, k,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   b_buffer, b_offset, b_ld,
                   beta,
                   c_buffer, c_offset, c_ld):
    cdef cl_event event = NULL
    cdef cl_command_queue command_queue=clpy.backend.opencl.env.get_command_queue()
    cdef int tmp = <int> a_buffer
    cdef cl_mem a_mem = <cl_mem>tmp
    tmp = <int>b_buffer
    cdef cl_mem b_mem = <cl_mem>tmp
    tmp = <int>c_buffer
    cdef cl_mem c_mem = <cl_mem>tmp

    cdef CLBlastStatusCode status = CLBlastSgemm(
        <const CLBlastLayout>layout,
        <const CLBlastTranspose>a_transpose,
        <const CLBlastTranspose>b_transpose,
        <const size_t>m,
        <const size_t>n,
        <const size_t>k,
        <const float>alpha,
        <const cl_mem>a_mem,
        <const size_t>a_offset,
        <const size_t>a_ld,
        <const cl_mem>b_mem,
        <const size_t>b_offset,
        <const size_t>b_ld,
        <const float>beta,
        <cl_mem>c_mem,
        <const size_t>c_offset,
        <const size_t>c_ld,
        <cl_command_queue*>&command_queue,
        <cl_event*>&event
        )
    if (status == CLBlastSuccess):
        api.WaitForEvents(1, <cl_event*>&event)
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

    clblast_sgemm(
        layout,
        a_transpose,
        b_transpose,
        m, n, k,
        alpha,
        A.data.buf.get(), A.data.cl_mem_offset() // A.itemsize, lda,
        B.data.buf.get(), B.data.cl_mem_offset() // B.itemsize, ldb,
        beta,
        C.data.buf.get(), C.data.cl_mem_offset() // C.itemsize, ldc)
