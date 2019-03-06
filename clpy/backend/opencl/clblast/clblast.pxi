cdef extern from "clblast_c.h":
    cdef enum CLBlastStatusCode_:
        CLBlastSuccess,
        CLBlastOpenCLCompilerNotAvailable,
        CLBlastTempBufferAllocFailure,
        CLBlastOpenCLOutOfResources,
        CLBlastOpenCLOutOfHostMemory,
        CLBlastOpenCLBuildProgramFailure,
        CLBlastInvalidValue,
        CLBlastInvalidCommandQueue,
        CLBlastInvalidMemObject,
        CLBlastInvalidBinary,
        CLBlastInvalidBuildOptions,
        CLBlastInvalidProgram,
        CLBlastInvalidProgramExecutable,
        CLBlastInvalidKernelName,
        CLBlastInvalidKernelDefinition,
        CLBlastInvalidKernel,
        CLBlastInvalidArgIndex,
        CLBlastInvalidArgValue,
        CLBlastInvalidArgSize,
        CLBlastInvalidKernelArgs,
        CLBlastInvalidLocalNumDimensions,
        CLBlastInvalidLocalThreadsTotal,
        CLBlastInvalidLocalThreadsDim,
        CLBlastInvalidGlobalOffset,
        CLBlastInvalidEventWaitList,
        CLBlastInvalidEvent,
        CLBlastInvalidOperation,
        CLBlastInvalidBufferSize,
        CLBlastInvalidGlobalWorkSize,
        CLBlastNotImplemented,
        CLBlastInvalidMatrixA,
        CLBlastInvalidMatrixB,
        CLBlastInvalidMatrixC,
        CLBlastInvalidVectorX,
        CLBlastInvalidVectorY,
        CLBlastInvalidDimension,
        CLBlastInvalidLeadDimA,
        CLBlastInvalidLeadDimB,
        CLBlastInvalidLeadDimC,
        CLBlastInvalidIncrementX,
        CLBlastInvalidIncrementY,
        CLBlastInsufficientMemoryA,
        CLBlastInsufficientMemoryB,
        CLBlastInsufficientMemoryC,
        CLBlastInsufficientMemoryX,
        CLBlastInsufficientMemoryY,
        CLBlastInsufficientMemoryTemp,
        CLBlastInvalidBatchCount,
        CLBlastInvalidOverrideKernel,
        CLBlastMissingOverrideParameter,
        CLBlastInvalidLocalMemUsage,
        CLBlastNoHalfPrecision,
        CLBlastNoDoublePrecision,
        CLBlastInvalidVectorScalar,
        CLBlastInsufficientMemoryScalar,
        CLBlastDatabaseError,
        CLBlastUnknownError,
        CLBlastUnexpectedError
    ctypedef CLBlastStatusCode_ CLBlastStatusCode
    cdef enum CLBlastLayout_:
        CLBlastLayoutRowMajor,
        CLBlastLayoutColMajor
    ctypedef CLBlastLayout_ CLBlastLayout
    cdef enum CLBlastTranspose_:
        CLBlastTransposeNo,
        CLBlastTransposeYes
        CLBlastTransposeConjugate
    ctypedef CLBlastTranspose_ CLBlastTranspose
    cdef enum CLBlastSide_:
        CLBlastSideLeft,
        CLBlastSideRight
    ctypedef CLBlastSide_ CLBlastSide
    cdef enum CLBlastDiagonal_:
        CLBlastDiagonalNonUnit,
        CLBlastDiagonalUnit
    ctypedef CLBlastDiagonal_ CLBlastDiagonal
    cdef enum CLBlastTriangle_:
        CLBlastTriangleUpper,
        CLBlastTriangleLower
    ctypedef CLBlastTriangle_ CLBlastTriangle

cdef extern from "clblast_c.h":
    CLBlastStatusCode CLBlastSgemm(
        const CLBlastLayout, const CLBlastTranspose, const CLBlastTranspose,
        const size_t, const size_t, const size_t,
        const float,
        const cl_mem, const size_t, const size_t,
        const cl_mem, const size_t, const size_t,
        const float,
        cl_mem, const size_t, const size_t,
        cl_command_queue*, cl_event*)
    CLBlastStatusCode CLBlastDgemm(
        const CLBlastLayout, const CLBlastTranspose, const CLBlastTranspose,
        const size_t, const size_t, const size_t,
        const double,
        const cl_mem, const size_t, const size_t,
        const cl_mem, const size_t, const size_t,
        const double,
        cl_mem, const size_t, const size_t,
        cl_command_queue*, cl_event*)
    CLBlastStatusCode CLBlastStrsm(
        const CLBlastLayout,
        const CLBlastSide,
        const CLBlastTriangle,
        const CLBlastTranspose,
        const CLBlastDiagonal,
        const size_t, const size_t,
        const float,
        const cl_mem, const size_t, const size_t,
        cl_mem, const size_t, const size_t,
        cl_command_queue*, cl_event*)
    CLBlastStatusCode CLBlastSgemmBatched(
        const CLBlastLayout layout,
        const CLBlastTranspose a_transpose,
        const CLBlastTranspose b_transpose,
        const size_t m,
        const size_t n,
        const size_t k,
        const float *alphas,
        const cl_mem a_buffer,
        const size_t *a_offsets,
        const size_t a_ld,
        const cl_mem b_buffer,
        const size_t *b_offsets,
        const size_t b_ld,
        const float *betas,
        cl_mem c_buffer,
        const size_t *c_offsets,
        const size_t c_ld,
        const size_t batch_count,
        cl_command_queue* queue,
        cl_event* event)
    CLBlastStatusCode CLBlastDgemmBatched(
        const CLBlastLayout layout,
        const CLBlastTranspose a_transpose,
        const CLBlastTranspose b_transpose,
        const size_t m,
        const size_t n,
        const size_t k,
        const double *alphas,
        const cl_mem a_buffer,
        const size_t *a_offsets,
        const size_t a_ld,
        const cl_mem b_buffer,
        const size_t *b_offsets,
        const size_t b_ld,
        const double *betas,
        cl_mem c_buffer,
        const size_t *c_offsets,
        const size_t c_ld,
        const size_t batch_count,
        cl_command_queue* queue,
        cl_event* event)
