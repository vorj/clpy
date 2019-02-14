import unittest

import numpy
import six

import clpy
from clpy import testing
from clpy.backend.opencl.exceptions import OpenCLProgramBuildError
from clpy.backend.ultima.exceptions import UltimaRuntimeError

@testing.gpu
class TestBuildExceptions(unittest.TestCase):

    def test_undeclared_identifier(self):
        with six.assertRaisesRegex(self, UltimaRuntimeError,
                                   'undeclared identifier'):
            x = clpy.core.array(numpy.array([1], dtype="float32"))
            clpy.ElementwiseKernel(
                'T x',
                '',
                'undeclared_identifier',
                'use_of_undeclared_indentifier')(x)

    def test_opencl_error(self):
        with six.assertRaisesRegex(self, OpenCLProgramBuildError,
                                   'CL_BUILD_PROGRAM_FAILURE Device#'):
            x = clpy.core.array(numpy.array([1], dtype="float32"))
            clpy.ElementwiseKernel(
                'T x',
                '',
                '__global T t;',
                'test')(x)

#    def test_assign_to_const_qualified_variable(self):
#        with six.assertRaisesRegex(self, OpenCLProgramBuildError,
#                                   'cannot assign|is not assignable'):
#            x = clpy.core.array(numpy.array([1], dtype="float32"))
#            clpy.ElementwiseKernel(
#                'T x',
#                'T y',
#                'x = y',
#                'assign_to_const_qualified_variable')(x)


if __name__ == "__main__":
    unittest.main()
