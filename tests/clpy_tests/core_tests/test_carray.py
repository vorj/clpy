import unittest

import clpy
from clpy.backend.ultima.exceptions import UltimaRuntimeError
from clpy import testing

import six


class TestCArray(unittest.TestCase):

    def test_size(self):  # TODO(vorj): support CArray::size()
        with six.assertRaisesRegex(self, UltimaRuntimeError,
                                   "Current ultima doesn't support "
                                   "CArray::size()"):
            x = clpy.arange(3).astype('i')
            y = clpy.ElementwiseKernel(
                'raw int32 x', 'int32 y', 'y = x.size()', 'test_carray_size',
            )(x, size=1)
            self.assertEqual(int(y[0]), 3)

    def test_shape(self):  # TODO(vorj): support CArray::shape()
        with six.assertRaisesRegex(self, UltimaRuntimeError,
                                   "Current ultima doesn't support "
                                   "CArray::shape()"):
            x = clpy.arange(6).reshape((2, 3)).astype('i')
            y = clpy.ElementwiseKernel(
                'raw int32 x', 'int32 y', 'y = x.shape()[i]',
                'test_carray_shape',
            )(x, size=2)
            testing.assert_array_equal(y, (2, 3))

    def test_strides(self):  # TODO(vorj): support CArray::strides()
        with six.assertRaisesRegex(self, UltimaRuntimeError,
                                   "Current ultima doesn't support "
                                   "CArray::strides()"):
            x = clpy.arange(6).reshape((2, 3)).astype('i')
            y = clpy.ElementwiseKernel(
                'raw int32 x', 'int32 y', 'y = x.strides()[i]',
                'test_carray_strides',
            )(x, size=2)
            testing.assert_array_equal(y, (12, 4))

    def test_getitem_int(self):
        x = clpy.arange(24).reshape((2, 3, 4)).astype('i')
        y = clpy.empty_like(x)
        y = clpy.ElementwiseKernel(
            'raw T x', 'int32 y', 'y = x[i]', 'test_carray_getitem_int',
        )(x, y)
        testing.assert_array_equal(y, x)

    def test_getitem_idx(self):
        x = clpy.arange(24).reshape((2, 3, 4)).astype('i')
        y = clpy.empty_like(x)
        y = clpy.ElementwiseKernel(
            'raw T x', 'int32 y',
            'ptrdiff_t idx[] = {i / 12, i / 4 % 3, i % 4}; y = x[idx]',
            'test_carray_getitem_idx',
        )(x, y)
        testing.assert_array_equal(y, x)
