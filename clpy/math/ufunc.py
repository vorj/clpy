from clpy import core


def create_math_ufunc(math_name, nargs, name, doc):
    assert 1 <= nargs <= 2
    if nargs == 1:
        return core.create_ufunc(
            name, (('b->e', 'out0 = convert_float_to_half(%s((float)in0))'
                            % math_name),
                   ('B->e', 'out0 = convert_float_to_half(%s((float)in0))'
                            % math_name),
                   'f->f', 'd->d', 'F->F', 'D->D'),
            'out0 = %s(in0)' % math_name, doc=doc)
    else:
        return core.create_ufunc(
            name, (('bb->e', 'out0 = convert_float_to_half'
                             '(%s((float)in0, (float)in1))'
                             % math_name),
                   ('BB->e', 'out0 = convert_float_to_half'
                             '(%s((float)in0, (float)in1))'
                             % math_name),
                   'ff->f', 'dd->d', 'FF->F', 'DD->D'),
            'out0 = %s(in0, in1)' % math_name, doc=doc)
