# -*- coding: utf-8 -*-
import unittest

import numpy

import clpy
from clpy.backend.opencl.clblast import clblast
from clpy.core import core
import functools


def for_each_dtype_and_blasfunc_pair(pairs):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kwargs):
            for pair in pairs:
                try:
                    kwargs['dtype'] = numpy.dtype(pair[0]).type
                    kwargs['blasfunc'] = pair[1]
                    impl(self, *args, **kwargs)
                except Exception:
                    print('dtype:', pair[0], ", blasfunc:", pair[1])
                    raise
        return test_func
    return decorator


GEMM_pairs = [
    ('float32', clblast.sgemm),
    ('float64', clblast.dgemm),
]


class TestBlas3GEMM(unittest.TestCase):
    """test class of CLBlast BLAS-3 GEMM functions"""

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_row_matrix_row_matrix(self, dtype, blasfunc):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=dtype)  # row-major
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype=dtype)  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_row_matrix_row_vector(self, dtype, blasfunc):
        npA = numpy.array([
            [1, 2],
            [4, 5],
            [7, 8]], dtype=dtype)  # row-major
        npB = numpy.array([
            [10],
            [13]], dtype=dtype)  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))
        m = npA.shape[0]  # op(A) rows = (A in row-major) rows = C rows

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_row_vector_row_matrix(self, dtype, blasfunc):
        npA = numpy.array([
            [10, 13, 16]
        ], dtype=dtype)  # row-major
        npB = numpy.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=dtype)  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_column_matrix_column_matrix(self, dtype, blasfunc):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=dtype)  # column-major
        # 1 4
        # 2 5
        # 3 6
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype=dtype)  # column-major
        # 10 13 16
        # 11 14 17
        transa = 1  # A is transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_column_matrix_column_vector(self, dtype, blasfunc):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=dtype)  # column-major
        # 1 4
        # 2 5
        # 3 6
        npB = numpy.array([
            [10, 11]], dtype=dtype)  # column-major
        # 10
        # 11
        transa = 1  # A is transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_column_vector_column_matrix(self, dtype, blasfunc):
        npA = numpy.array([
            [1],
            [4]], dtype=dtype)  # column-major
        # 1 4
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype=dtype)  # column-major
        # 10 13 16
        # 11 14 17
        transa = 1  # A is transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_row_matrix_column_matrix(self, dtype, blasfunc):
        npA = numpy.array([
            [1, 2],
            [4, 5]], dtype=dtype)  # row-major
        # 1 2
        # 4 5
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype=dtype)  # column-major
        # 10 13 16
        # 11 14 17
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major)    rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major)    cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_row_matrix_column_vector(self, dtype, blasfunc):
        npA = numpy.array([
            [1, 2],
            [4, 5]], dtype=dtype)  # row-major
        # 1 2
        # 4 5
        npB = numpy.array([
            [10, 11]], dtype=dtype)  # column-major
        # 10
        # 11
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major)    rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major)    cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_row_vector_column_matrix(self, dtype, blasfunc):
        npA = numpy.array([
            [1, 2]], dtype=dtype)  # row-major
        # 1 2
        # 4 5
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype=dtype)  # column-major
        # 10 13 16
        # 11 14 17
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major)    rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major)    cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_column_matrix_row_matrix(self, dtype, blasfunc):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=dtype)  # column-major
        # 1 4
        # 2 5
        # 3 6
        npB = numpy.array([
            [10, 11],
            [13, 14]], dtype=dtype)  # row-major
        # 10 11
        # 13 14
        transa = 1  # A is transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major)    cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_column_matrix_row_vector(self, dtype, blasfunc):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=dtype)  # column-major
        # 1 4
        # 2 5
        # 3 6
        npB = numpy.array([
            [10],
            [13]], dtype=dtype)  # row-major
        # 10
        # 13
        transa = 1  # A is transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major)    cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_column_vector_row_matrix(self, dtype, blasfunc):
        npA = numpy.array([
            [1],
            [2],
            [3]], dtype=dtype)  # column-major
        # 1 2 3
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype=dtype)  # row-major
        # 10 11
        # 13 14
        # 16 17
        transa = 1  # A is transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major)    cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=dtype)  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb, m, n, k,
                 1.0, clpA, lda,
                 clpB, ldb,
                 0.0, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_invalid_transa(self, dtype, blasfunc):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype=dtype)

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)

        expectedC = numpy.dot(npA, npB).T
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]
        with self.assertRaises(ValueError):
            blasfunc('C', transa='a', transb='t',
                     m=m, n=n, k=k,
                     alpha=1.0, A=clpA, lda=k,
                     B=clpB, ldb=n,
                     beta=0.0,
                     C=clpC, ldc=m
                     )

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_invalid_transb(self, dtype, blasfunc):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype=dtype)

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)

        expectedC = numpy.dot(npA, npB).T
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]
        with self.assertRaises(ValueError):
            blasfunc('C', transa='n', transb='a',
                     m=m, n=n, k=k,
                     alpha=1.0, A=clpA, lda=k,
                     B=clpB, ldb=n,
                     beta=0.0,
                     C=clpC, ldc=m
                     )

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_alpha_matrix_matrix(self, dtype, blasfunc):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          dtype=dtype)  # row-major
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype=dtype)  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 2.0
        beta = 0.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=dtype)  # col major in clpy
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)  # col major in clpy
        clpB.set(npB)

        expectedC = numpy.dot(npA, npB) * alpha
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        # alpha * (A^t x B^T) in col-major = alpha * AxB in row major
        blasfunc('C', transa, transb,
                 m, n, k, alpha,
                 clpA, lda,
                 clpB, ldb,
                 beta, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major

        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_beta_matrix_matrix(self, dtype, blasfunc):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          dtype=dtype)  # row-major
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype=dtype)  # row-major
        npC = numpy.array([[19, 20, 21], [22, 23, 24], [
                          25, 26, 27]], dtype=dtype)  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        alpha = 1.0
        beta = 2.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=dtype)  # col-major in clpy
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)  # col-major in clpy
        clpB.set(npB)

        clpC = clpy.ndarray(npC.shape, dtype=dtype)  # col-major in clpy
        clpC.set(npC.T)  # transpose C

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        # AxB + beta*C
        expectedC = numpy.add(numpy.dot(npA, npB), beta * npC)

        # (A^T x B^T) + C^T in col-major = A x B + C in row-major
        blasfunc('C', transa, transb,
                 m, n, k, alpha,
                 clpA, lda,
                 clpB, ldb,
                 beta, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major

        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_beta_0_matrix_matrix(self, dtype, blasfunc):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          dtype=dtype)  # row-major
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype=dtype)  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 0.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=dtype)  # col major in clpy
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)  # col major in clpy
        clpB.set(npB)

        expectedC = numpy.dot(npA, npB) * alpha
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)
        clpC.fill(numpy.nan)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        # alpha * (A^t x B^T) in col-major = alpha * AxB in row major
        blasfunc('C', transa, transb,
                 m, n, k, alpha,
                 clpA, lda,
                 clpB, ldb,
                 beta, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major

        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_chunk_gemm_A(self, dtype, blasfunc):
        # create chunk and free to prepare chunk in pool
        pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(pool.malloc)
        pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        tmp = pool.malloc(pooled_chunk_size)
        pool.free(tmp.buf, pooled_chunk_size, 0)

        size = 3
        wrong_value = numpy.nan

        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15],
                           [16, 17, 18]], dtype=dtype)
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 0.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # clpA is chunk with offset != 0
        clpA = clpy.empty(npA.shape, dtype=dtype)
        self.assertTrue(clpA.data.mem.offset != 0)
        clpA.set(npA)

        # clpB is chunk with offset == 0
        clpB = clpy.empty(npB.shape, dtype=dtype)
        self.assertTrue(clpB.data.mem.offset == 0)
        clpB.set(npB)

        expectedC = numpy.dot(npA, npB)
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb,
                 m, n, k, alpha,
                 clpA, lda,
                 clpB, ldb,
                 beta, clpC, ldc
                 )

        actualC = clpC.get().T

        clpy.backend.memory.set_allocator()

        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_chunk_gemm_B(self, dtype, blasfunc):
        # create chunk and free to prepare chunk in pool
        pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(pool.malloc)
        pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        tmp = pool.malloc(pooled_chunk_size)
        pool.free(tmp.buf, pooled_chunk_size, 0)

        size = 3
        wrong_value = numpy.nan

        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15],
                           [16, 17, 18]], dtype=dtype)
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 0.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # clpB is chunk with offset != 0
        clpB = clpy.empty(npB.shape, dtype=dtype)
        self.assertTrue(clpB.data.mem.offset != 0)
        clpB.set(npB)

        # clpA is chunk with offset == 0
        clpA = clpy.empty(npA.shape, dtype=dtype)
        self.assertTrue(clpA.data.mem.offset == 0)
        clpA.set(npA)

        expectedC = numpy.dot(npA, npB)
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb,
                 m, n, k, alpha,
                 clpA, lda,
                 clpB, ldb,
                 beta, clpC, ldc
                 )

        actualC = clpC.get().T

        clpy.backend.memory.set_allocator()

        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_chunk_gemm_C(self, dtype, blasfunc):
        # create chunk and free to prepare chunk in pool
        pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(pool.malloc)
        pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        tmp = pool.malloc(pooled_chunk_size)
        pool.free(tmp.buf, pooled_chunk_size, 0)

        size = 3
        wrong_value = numpy.nan

        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15],
                           [16, 17, 18]], dtype=dtype)
        npC = numpy.array([[19, 20, 21], [22, 23, 24],
                           [25, 26, 27]], dtype=dtype)
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 1.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        expectedC = numpy.add(numpy.dot(npA, npB), beta * npC)

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # clpC is chunk with offset != 0
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)
        self.assertTrue(clpC.data.mem.offset != 0)
        clpC.set(npC.T)

        # clpA is chunk with offset == 0
        clpA = clpy.empty(npA.shape, dtype=dtype)
        self.assertTrue(clpA.data.mem.offset == 0)
        clpA.set(npA)

        # clpB is chunk with offset == 0
        clpB = clpy.empty(npB.shape, dtype=dtype)
        self.assertTrue(clpB.data.mem.offset == 0)
        clpB.set(npB)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blasfunc('C', transa, transb,
                 m, n, k, alpha,
                 clpA, lda,
                 clpB, ldb,
                 beta, clpC, ldc
                 )

        actualC = clpC.get().T

        clpy.backend.memory.set_allocator()

        self.assertTrue(numpy.allclose(expectedC, actualC))

    @for_each_dtype_and_blasfunc_pair(GEMM_pairs)
    def test_strides_transpose_A(self, dtype, blasfunc):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          dtype=dtype)  # row-major
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype=dtype)  # row-major
        npC = numpy.array([[19, 20, 21], [22, 23, 24], [
                          25, 26, 27]], dtype=dtype)  # row-major

        alpha = 1.1
        beta = 2.1

        m = npA.shape[1]
        n = npB.shape[1]
        k = npA.shape[0]

        clpA = clpy.ndarray(npA.shape, dtype=dtype)  # col-major in clpy
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)  # col-major in clpy
        clpB.set(npB)

        # change A.strides
        clpA = clpA.transpose(1, 0)
        npA = npA.transpose(1, 0)

        clpC = clpy.ndarray(npC.shape, dtype=dtype)  # col-major in clpy
        clpC.set(npC.T)  # transpose C

        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        # AxB + beta*C
        expectedC = numpy.add(alpha * numpy.dot(npA, npB), beta * npC)

        # (A^T x B^T) + C^T in col-major = A x B + C in row-major
        blasfunc('C', transa, transb,
                 m, n, k, alpha,
                 clpA, lda,
                 clpB, ldb,
                 beta, clpC, ldc
                 )

        actualC = clpC.get().T  # as row-major

        self.assertTrue(numpy.allclose(expectedC, actualC))


TRSM_pairs = [
    ('float32', clblast.strsm)
]


class TestBlas3TRSM(unittest.TestCase):
    """test class of CLBlast BLAS-3 TRSM functions"""

    @for_each_dtype_and_blasfunc_pair(TRSM_pairs)
    def test_strsm_works(self, dtype, blasfunc):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=dtype)
        npB = numpy.array([
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]], dtype=dtype)

        m = npA.shape[0]
        n = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=dtype)
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=dtype)
        clpB.set(npB)

        alpha = 1.0

        layout = 'R'
        side = 'L'
        triangle = 'U'
        transa = 'N'
        diagonal = 'N'

        blasfunc(layout, side, triangle, transa, diagonal,
                 m, n, alpha, clpA, n, clpB, n)

        actualB = clpB.get()
        expectedB = numpy.dot(numpy.linalg.inv(numpy.triu(npA)), npB)
        self.assertTrue(numpy.allclose(expectedB, actualB))


if __name__ == '__main__':
    unittest.main()
