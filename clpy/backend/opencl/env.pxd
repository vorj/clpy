include "common_decl.pxi"

cdef cl_context get_context()
cdef cl_command_queue get_command_queue()
cpdef int get_device_id()
cpdef set_device_id(int id)
cdef cl_device_id* get_devices()
cdef cl_device_id get_device()
