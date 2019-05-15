import clpy

import os
import subprocess
import tempfile
import time


class TempFile(object):
    def __init__(self, filename, source):
        self.fn = filename
        self.s = source

    def __enter__(self):
        with open(self.fn, 'w') as f:
            f.write(self.s)

    def __exit__(self, exception_type, exception_value, traceback):
        os.remove(self.fn)


def _exec_ultima(source, _clpy_header=''):
    source = _clpy_header + '\n' \
        'static void __clpy_begin_print_out() ' \
        '__attribute__((annotate("clpy_begin_print_out")));\n' \
        + source + '\n' \
        'static void __clpy_end_print_out()' \
        '__attribute__((annotate("clpy_end_print_out")));\n'

    filename = tempfile.gettempdir() + "/" + str(time.monotonic()) + ".cpp"

    with TempFile(filename, source):
        root_dir = os.path.join(clpy.__path__[0], "..")
        proc = subprocess.Popen('{} {} -- -I {}'
                                .format(os.path.join(root_dir,
                                                     "ultima",
                                                     "ultima"),
                                        filename,
                                        os.path.join(root_dir,
                                                     "clpy",
                                                     "core",
                                                     "include"))
                                .strip().split(" "),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True)
        try:
            source, errstream = proc.communicate(timeout=15)
            proc.wait()
        except subprocess.TimeoutExpired:
            proc.kill()
            source, errstream = proc.communicate()

        if proc.returncode != 0 and len(errstream) > 0:
            raise clpy.backend.ultima.exceptions.UltimaRuntimeError(
                proc.returncode, errstream)

    return source
