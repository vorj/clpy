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
