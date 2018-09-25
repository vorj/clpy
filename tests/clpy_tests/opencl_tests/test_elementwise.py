import unittest

import numpy

import clpy
from clpy import testing


@testing.gpu
class TestUltima(unittest.TestCase):

    def test_implicit_conversion_with_constructor(self):
        x = clpy.core.array(numpy.array([1]))
        clpy.ElementwiseKernel(
            'T x',
            '',
            'test t = 3;',
            'test_implicit_conversion_with_constructor',
            preamble='''
            struct test{
              test(int dummy){}
            };
            ''')(x)


if __name__ == "__main__":
    unittest.main()
