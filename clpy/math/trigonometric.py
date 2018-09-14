from clpy import core
from clpy.math import ufunc


sin = ufunc.create_math_ufunc(
    'sin', 1, 'clpy_sin',
    '''Elementwise sine function.

    .. seealso:: :data:`numpy.sin`

    ''')


cos = ufunc.create_math_ufunc(
    'cos', 1, 'clpy_cos',
    '''Elementwise cosine function.

    .. seealso:: :data:`numpy.cos`

    ''')


tan = ufunc.create_math_ufunc(
    'tan', 1, 'clpy_tan',
    '''Elementwise tangent function.

    .. seealso:: :data:`numpy.tan`

    ''')


arcsin = ufunc.create_math_ufunc(
    'asin', 1, 'clpy_arcsin',
    '''Elementwise inverse-sine function (a.k.a. arcsine function).

    .. seealso:: :data:`numpy.arcsin`

    ''')


arccos = ufunc.create_math_ufunc(
    'acos', 1, 'clpy_arccos',
    '''Elementwise inverse-cosine function (a.k.a. arccosine function).

    .. seealso:: :data:`numpy.arccos`

    ''')


arctan = ufunc.create_math_ufunc(
    'atan', 1, 'clpy_arctan',
    '''Elementwise inverse-tangent function (a.k.a. arctangent function).

    .. seealso:: :data:`numpy.arctan`

    ''')


hypot = ufunc.create_math_ufunc(
    'hypot', 2, 'clpy_hypot',
    '''Computes the hypoteneous of orthogonal vectors of given length.

    This is equivalent to ``sqrt(x1 **2 + x2 ** 2)``, while this function is
    more efficient.

    .. seealso:: :data:`numpy.hypot`

    ''')


arctan2 = ufunc.create_math_ufunc(
    'atan2', 2, 'clpy_arctan2',
    '''Elementwise inverse-tangent of the ratio of two arrays.

    .. seealso:: :data:`numpy.arctan2`

    ''')


deg2rad = core.create_ufunc(
    'clpy_deg2rad',
    (('b->e', 'out0 = convert_float_to_half(in0 * (float)(M_PI / 180))'),
     ('B->e', 'out0 = convert_float_to_half(in0 * (float)(M_PI / 180))'),
     'f->f', 'd->d'),
    'out0 = in0 * (out0_type)(M_PI / 180)',
    doc='''Converts angles from degrees to radians elementwise.

    .. seealso:: :data:`numpy.deg2rad`, :data:`numpy.radians`

    ''')


rad2deg = core.create_ufunc(
    'clpy_rad2deg',
    (('b->e', 'out0 = convert_float_to_half(in0 * (float)(180 / M_PI))'),
     ('B->e', 'out0 = convert_float_to_half(in0 * (float)(180 / M_PI))'),
     'f->f', 'd->d'),
    'out0 = in0 * (out0_type)(180 / M_PI)',
    doc='''Converts angles from radians to degrees elementwise.

    .. seealso:: :data:`numpy.rad2deg`, :data:`numpy.degrees`

    ''')


# TODO(okuta): Implement unwrap


degrees = rad2deg
radians = deg2rad
