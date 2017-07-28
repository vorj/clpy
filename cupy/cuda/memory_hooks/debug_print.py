import sys

from cupy.cuda import memory_hook


class DebugPrintHook(memory_hook.MemoryHook):
    """Memory hook that prints debug information.

    This memory hook outputs the debug information of input arguments of
    ``malloc`` and ``free`` methods involved in the hooked functions
    at preprocessing time (that is, just before each method is called).

    Example:
        Code example::

            The basic usage is to use it with ``with`` statement.

            >>> import cupy
            >>> from cupy.cuda import memory_hooks
            >>>
            >>> cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
            >>> with memory_hooks.DebugPrintHook():
            ...     x = cupy.array([1, 2, 3])
            ...     del x  # doctest:+SKIP

        Output example::

            {"hook":"alloc","device_id":0,"mem_size":512,"mem_ptr":150496608256}
            {"hook":"malloc","device_id":0,"size":24,"mem_size":512,"mem_ptr":150496608256}
            {"hook":"free","device_id":0,"mem_size":512,"mem_ptr":150496608256}

        where the output format is JSONL (JSON Lines) and
        ``hook`` is the name of hook point, and
        ``device_id`` is the CUDA Device ID, and
        ``size`` is the requested memory size to allocate, and
        ``mem_size`` is the rounded memory size to be allocated, and
        ``mem_ptr`` is the memory pointer.

    Attributes:
        file: Output file_like object that that redirect to.
        flush: If ``True``, this hook forcibly flushes the text stream
            at the end of print. The default is True.

    """

    name = 'DebugPrintHook'

    def __init__(self, file=sys.stdout, flush=True):
        self.file = file
        self.flush = flush

    def _print(self, msg):
        self.file.write(msg)
        self.file.write('\n')
        if self.flush:
            self.file.flush()

    def malloc_postprocess(self, device_id, size, mem_size, mem_ptr):
        msg = '{"hook":"%s","device_id":%d,' \
              '"size":%d,"mem_size":%d,"mem_ptr":%d}'
        msg %= ('malloc', device_id, size, mem_size, mem_ptr)
        self._print(msg)

    def alloc_postprocess(self, device_id, mem_size, mem_ptr):
        msg = '{"hook":"%s","device_id":%d,' \
              '"mem_size":%d,"mem_ptr":%d}'
        msg %= ('alloc', device_id, mem_size, mem_ptr)
        self._print(msg)

    def free_postprocess(self, device_id, mem_size, mem_ptr):
        msg = '{"hook":"%s","device_id":%d,' \
              '"mem_size":%d,"mem_ptr":%d}'
        msg %= ('free', device_id, mem_size, mem_ptr)
        self._print(msg)
