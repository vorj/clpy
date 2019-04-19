# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import unittest

import clpy
from clpy import testing

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

    with TempFile(filename, source) as tf:
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


class TestUltimaCastConversion(unittest.TestCase):

    def test_function_style_cast(self):
        x = '''
void f() 
{
    (int)(3.F);
}
'''
        y = _exec_ultima(
            '''
            void f(){
              int(3.F);
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_static_cast(self):
        x = '''
void f() 
{
    (int)(3.F);
}
'''
        y = _exec_ultima(
            '''
            void f(){
              static_cast<int>(3.F);
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_const_cast(self):
        x = '''
void f() 
{
    const int a = 3;
    (int *)(&a);
}
'''
        y = _exec_ultima(
            '''
            void f(){
              const int a = 3;
              const_cast<int*>(&a);
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_reinterpret_cast(self):
        x = '''
void f() 
{
    int a = 3;
    (float *)(&a);
}
'''
        y = _exec_ultima(
            '''
            void f(){
              int a = 3;
              reinterpret_cast<float*>(&a);
            }
            ''')
        self.assertEqual(x[1:], y)


class TestUltimaConstructor(unittest.TestCase):

    def test_user_defined_default_constructor(self):
        x = '''
typedef struct A {
    }A;
void constructor_A(A*const this) 
    {
    }
void f() 
{
    A a;constructor_A(&a);
}
'''
        y = _exec_ultima(
            '''
            class A{
            public:
              A(){}
            };
            void f(){
              A a;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_explicit_constructor(self):
        x = '''
typedef struct A {
    int a;
    }A;
void constructor_A(A*const this, int t) 
    {
    this->a = t;
    }
void f() 
{
    A a;constructor_A(&a, 3);
}
'''
        y = _exec_ultima(
            '''
            class A{
              int a;
            public:
              explicit A(int t):a{t}{}
            };
            void f(){
              A a{3};
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_one_arg_constructor(self):
        x = '''
typedef struct A {
    int a;
    }A;
void constructor_A(A*const this, int t) 
    {
    this->a = t;
    }
void f() 
{
    A a;constructor_A(&a, 3);
}
'''
        y = _exec_ultima(
            '''
            class A{
              int a;
            public:
              A(int t):a{t}{}
            };
            void f(){
              A a{3};
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_one_arg_constructor_implicit_conversion(self):
        x = '''
typedef struct A {
    int a;
    }A;
void constructor_A(A*const this, int t) 
    {
    this->a = t;
    }
void f() 
{
    A a;constructor_A(&a, 3);
}
'''
        y = _exec_ultima(
            '''
            class A{
              int a;
            public:
              A(int t):a{t}{}
            };
            void f(){
              A a = 3;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_two_arg_constructor(self):
        x = '''
typedef struct A {
    int a;
    float b;
    }A;
void constructor_A(A*const this, int t, float u) 
    {
    this->a = t;
    this->b = u;
    }
void f() 
{
    A a;constructor_A(&a, 42, 3.1400001F);
}
'''
        y = _exec_ultima(
            '''
            class A{
              int a;
              float b;
            public:
              explicit A(int t, float u):a{t}, b{u}{}
            };
            void f(){
              A a{42, 3.14f};
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_parenthesis_initializer_list_constructor(self):
        x = '''
typedef struct A {
    int a;
    float b;
    }A;
void constructor_A(A*const this, int t, float u) 
    {
    this->a = t;
    this->b = u;
    }
void f() 
{
    A a;constructor_A(&a, 42, 3.1400001F);
}
'''
        y = _exec_ultima(
            '''
            class A{
              int a;
              float b;
            public:
              explicit A(int t, float u):a(t), b(u){}
            };
            void f(){
              A a{42, 3.14f};
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_system_defined_default_constructor(self):
        x = '''
typedef struct A {
}A;
void f() 
{
    A a;
}
'''
        y = _exec_ultima(
            '''
            struct A{
            };
            void f(){
              A a;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_explicitlly_defaulted_default_constructor(self):
        x = '''
typedef struct A {
    }A;
void constructor_A(A*const this){};
void f() 
{
    A a;
}
'''
        y = _exec_ultima(
            '''
            struct A{
              A() = default;
            };
            void f(){
              A a;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_default_constructor_variable_group(self):
        x = '''
typedef struct A {
}A;
void f() 
{
    A a, b;
}
'''
        y = _exec_ultima(
            '''
            struct A{
            };
            void f(){
              A a, b;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_user_defined_default_constructor_variable_group(self):
        x = '''
typedef struct A {
    }A;
void constructor_A(A*const this) 
    {
    }
void f() 
{
    A a, b;constructor_A(&a);constructor_A(&b);
}
'''
        y = _exec_ultima(
            '''
            struct A{
              A(){}
            };
            void f(){
              A a, b;
            }
            ''')
        self.assertEqual(x[1:], y)

    def test_unnamed_struct(self):
        x = '''
void f() 
{
    struct {
        int t;
    } a,  b = {4},  c = {3};
}
'''
        y = _exec_ultima(
            '''
            void f(){
              struct{
                int t;
              }a, b = {4}, c{3};
            }
            ''')
        self.assertEqual(x[1:], y)


if __name__ == "__main__":
    unittest.main()
