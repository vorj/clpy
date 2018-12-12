include "common_decl.pxi"

cdef cl_context get_context()
cdef cl_command_queue get_command_queue()
cdef cl_device_id* get_devices()
cdef cl_device_id get_primary_device()
