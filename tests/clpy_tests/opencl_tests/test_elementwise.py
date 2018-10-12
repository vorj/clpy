import unittest

import numpy

import clpy
from clpy import testing


@testing.gpu
class TestUltima(unittest.TestCase):

    def test_implicit_conversion_with_constructor(self):
        x = clpy.core.array(numpy.array([1]))
        y = clpy.ElementwiseKernel(
            'T x',
            'T y',
            'test t = 3; y = t.v',
            'test_implicit_conversion_with_constructor',
            preamble='''
            struct test{
              int v;
              test(int v_):v(v_+1){}
            };
            ''')(x)
        self.assertTrue(y[0] == 4)


if __name__ == "__main__":
    unittest.main()
