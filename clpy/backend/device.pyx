# distutils: language = c++

import atexit

import six

# from clpy.backend cimport cublas
# from clpy.backend cimport cusparse
# from clpy.backend cimport runtime

cpdef int get_device_id() except *:
    return 0  # TODO(LWisteria): Always zero until implement multi device
#    return runtime.getDevice()


cdef dict _cublas_handles = {}
cdef dict _cusolver_handles = {}
cdef dict _cusparse_handles = {}


cpdef get_cublas_handle():
    dev_id = get_device_id()
    if dev_id in _cublas_handles:
        return _cublas_handles[dev_id]
    return Device().cublas_handle


cpdef get_cusolver_handle():
    dev_id = get_device_id()
    if dev_id in _cusolver_handles:
        return _cusolver_handles[dev_id]
    return Device().cusolver_handle


cpdef get_cusparse_handle():
    dev_id = get_device_id()
    if dev_id in _cusparse_handles:
        return _cusparse_handles[dev_id]
    return Device().cusparse_handle


cdef class Device:

    """Object that represents a CUDA device.

    This class provides some basic manipulations on CUDA devices.

    It supports the context protocol. For example, the following code is an
    example of temporarily switching the current device::

       with Device(0):
           do_something_on_device_0()

    After the *with* statement gets done, the current device is reset to the
    original one.

    Args:
        device (int or clpy.cuda.Device): Index of the device to manipulate. Be
            careful that the device ID (a.k.a. GPU ID) is zero origin. If it is
            a Device object, then its ID is used. The current device is
            selected by default.

    Attributes:
        id (int): ID of this device.

    """

    def __init__(self, device=None):
        if device is None:
            self.id = get_device_id()
        else:
            self.id = int(device)

        self._device_stack = []

    def __int__(self):
        return self.id

    def __enter__(self):
        # TODO(LWisteria): Nothing to do until implement multi device
        # cdef int id = get_device_id()
        # self._device_stack.append(id)
        # if self.id != id:
        #     self.use()
        return self

    def __exit__(self, *args):
        pass  # TODO(LWisteria): Nothing to do until implement multi device
        # runtime.setDevice(self._device_stack.pop())

    def __repr__(self):
        return '<CUDA Device %d>' % self.id

    cpdef use(self):
        """Makes this device current.

        If you want to switch a device temporarily, use the *with* statement.

        """
        pass  # TODO(LWisteria): Nothing to do until implement multi device
        # runtime.setDevice(self.id)

    cpdef synchronize(self):
        """Synchronizes the current thread to the device."""
        with self:
            raise NotImplementedError("clpy does not support this")
            # runtime.deviceSynchronize()

    @property
    def compute_capability(self):
        """Compute capability of this device.

        The capability is represented by a string containing the major index
        and the minor index. For example, compute capability 3.5 is represented
        by the string '35'.

        """
        raise NotImplementedError("clpy does not support this")
        # major = runtime.deviceGetAttribute(75, self.id)
        # minor = runtime.deviceGetAttribute(76, self.id)
        # return '%d%d' % (major, minor)

    @property
    def cublas_handle(self):
        """The cuBLAS handle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        if self.id in _cublas_handles:
            return _cublas_handles[self.id]
        with self:
            raise NotImplementedError("clpy does not support this")
            # handle = cublas.create()
            # _cublas_handles[self.id] = handle
            # return handle

    @property
    def cusolver_handle(self):
        """The cuSOLVER handle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        raise NotImplementedError("clpy does not support this")

        '''
        if self.id in _cusolver_handles:
            return _cusolver_handles[self.id]
        with self:
            handle = cusolver.create()
            _cusolver_handles[self.id] = handle
            return handle
        '''

    @property
    def cusparse_handle(self):
        """The cuSPARSE handle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        if self.id in _cusparse_handles:
            return _cusparse_handles[self.id]
        with self:
            raise NotImplementedError("clpy does not support this")
            # handle = cusparse.create()
            # _cusparse_handles[self.id] = handle
            # return handle

    def __richcmp__(Device self, Device other, int op):
        if op == 0:
            return self.id < other.id
        if op == 1:
            return self.id <= other.id
        if op == 2:
            return self.id == other.id
        if op == 3:
            return self.id != other.id
        if op == 4:
            return self.id > other.id
        if op == 5:
            return self.id >= other.id
        return NotImplemented


def from_pointer(ptr):
    """Extracts a Device object from a device pointer.

    Args:
        ptr (ctypes.c_void_p): Pointer to the device memory.

    Returns:
        Device: The device whose memory the pointer refers to.

    """
    raise NotImplementedError("clpy does not support this")
    # attrs = runtime.pointerGetAttributes(ptr)
    # return Device(attrs.device)


@atexit.register
def destroy_cublas_handles():
    """Destroys the cuBLAS handles for all devices."""
    global _cublas_handles
    for handle in _cublas_handles.itervalues():
        raise NotImplementedError("clpy does not support this")
        # cublas.destroy(handle)
    _cublas_handles = {}


@atexit.register
def destroy_cusparse_handles():
    """Destroys the cuSPARSE handles for all devices."""
    global _cusparse_handles
    for handle in six.itervalues(_cusparse_handles):
        raise NotImplementedError("clpy does not support this")
        # cusparse.destroy(handle)
    _cusparse_handles = {}
