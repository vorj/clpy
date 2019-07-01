import clpy

import unittest

import os
import locale
import subprocess


filedir = os.path.dirname(__file__)

headercvt_wd = os.path.join(
    os.path.dirname(__file__),
    "..",  # opencl_tests
    "..",  # clpy_tests
    "..",  # tests
    "..",  # clpy
    "headercvt")
headercvt_abspath = os.path.join(headercvt_wd, "headercvt")


def check_existence_of_headercvt():
    if not os.path.isfile(headercvt_abspath):
        raise FileNotFoundError("headercvt does not exist")


def exec_headercvt(source):
    p = subprocess.run(f"{headercvt_abspath} /dev/stdin --",
            shell=True,
            cwd=filedir,
            input=source.encode(locale.getpreferredencoding()),
            timeout=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True
            )

def get_result_files():
    with open("func_decl.pxi", "r") as f:
        func_decl_str = f.read()
    with open("preprocessor_defines.pxi", "r") as f:
        preprocessor_defines_str = f.read()
    with open("types.pxi", "r") as f:
        types_str = f.read()
    return { \
            "func_decl": func_decl_str,
            "preprocessor_defines": preprocessor_defines_str,
            "types": types_str
           }

def kick_headercvt_and_get_results(source):
    exec_headercvt(source)
    return get_result_files()

def contains(result_string, match_string):
    return match_string in result_string

def compile_with(source):
    source = """
include "func_decl.pxi"
include "preprocessor_defines.pxi"
include "types.pxi"


""" + source

    with open("test_case.pyx", "w") as f:
        f.write(source)
        f.flush()
        os.fsync(f.fileno())
        try:
            subprocess.run(f"cython {f.name}",
                    cwd=filedir,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode(locale.getpreferredencoding()))
            print(e.stderr.decode(locale.getpreferredencoding()))
            return False


class TestHeadercvtWorking(unittest.TestCase):
    def setUp(self):
        check_existence_of_headercvt()

    def test_headercvt_working(self):
        results = kick_headercvt_and_get_results("")
        # subprocess raises an exception if
        # headercvt returned non-zero exit code.

class TestHeadercvtPreprocDefines(unittest.TestCase):
    def setUp(self):
        check_existence_of_headercvt()

    def test_headercvt_preproc_define_accept_case(self):
        results = kick_headercvt_and_get_results("""
        #define CL_SOME_VALUE 1
        """)
        self.assertTrue(contains(results["preprocessor_defines"], "CL_SOME_VALUE"))
        self.assertTrue(compile_with("print(CL_SOME_VALUE)"))

    def test_headercvt_preproc_define_decline_case(self):
        results = kick_headercvt_and_get_results("""
        #define SOME_VALUE 1
        """)
        self.assertTrue(not contains(results["preprocessor_defines"], "SOME_VALUE"))

class TestHeadercvtFuncDecl(unittest.TestCase):
    def setUp(self):
        check_existence_of_headercvt()

    def test_headercvt_funcdecl_accept_case(self):
        results = kick_headercvt_and_get_results("""
        void clSomeFunction(int, void *);
        """)
        self.assertTrue(contains(results["func_decl"], "clSomeFunction(int, void *)"))
        self.assertTrue(compile_with("clSomeFunction(10, <void*>0)"))

    def test_headercvt_funcdecl_decline_case(self):
        results = kick_headercvt_and_get_results("""
        void SomeFunction(int, void *);
        """)
        self.assertTrue(not contains(results["func_decl"], "SomeFunction"))

class TestHeadercvtTypes(unittest.TestCase):
    def setUp(self):
        check_existence_of_headercvt()

    def test_headercvt_typedef(self):
        results = kick_headercvt_and_get_results("""
        typedef int clpy_int;
        """)
        self.assertTrue(contains(results["types"], "ctypedef int clpy_int"))
        self.assertTrue(compile_with("cdef clpy_int foo = 0"))

    def test_headercvt_typedef_to_pointer(self):
        results = kick_headercvt_and_get_results("""
        typedef int* clpy_intptr;
        """)
        self.assertTrue(contains(results["types"], "ctypedef int * clpy_intptr"))
        self.assertTrue(compile_with("cdef clpy_intptr foo = <clpy_intptr>0"))

    def test_headercvt_typedef_to_tagged_struct(self):
        results = kick_headercvt_and_get_results("""
        typedef struct clpy_struct_tag{
            int member;
        } clpy_struct_t;
        """)
        self.assertTrue(contains(results["types"], "ctypedef struct clpy_struct_t:"))
        self.assertTrue(compile_with("cdef clpy_struct_t foo\nfoo.member = 0"))

    def test_headercvt_typedef_to_discretely_tagged_struct(self):
        results = kick_headercvt_and_get_results("""
        typedef struct clpy_struct_tag{
            int member;
        };
        typedef struct clpy_struct_tag clpy_struct_t;
        """)
        # TODO(nsakabe-fixstars):
        # Make headercvt support this case of decl and update this testcase.

    def test_headercvt_typedef_to_implicitly_declared_pointer(self):
        results = kick_headercvt_and_get_results("""
        typedef struct clpy_struct_tag *    clpy_pointer_to_struct_t;
        """)
        self.assertTrue(contains(results["types"], "cdef struct clpy_struct_tag"))
        self.assertTrue(contains(results["types"], "ctypedef clpy_struct_tag * clpy_pointer_to_struct_t"))
        self.assertTrue(compile_with("cdef clpy_pointer_to_struct_t foo = <clpy_pointer_to_struct_t>0"))

    def test_headercvt_ignore_union_decl(self):
        results = kick_headercvt_and_get_results("""
        typedef union clpy_union_tag{
            int member1;
        } clpy_union_t;
        """)
        self.assertTrue(not contains(results["types"], "clpy_union_t"))

    def test_headercvt_ignore_union_reference(self):
        results = kick_headercvt_and_get_results("""
        typedef union clpy_union_tag{
            int member1;
        } clpy_union_t;
        typedef clpy_union_t clpy_typedefed_union_t;
        """)
        self.assertTrue(not contains(results["types"], "clpy_typedefed_union_t"))
