import unittest

import locale
import os
import subprocess
import tempfile


headercvt_abspath = os.path.join(
    os.path.dirname(__file__),
    "..",  # opencl_tests
    "..",  # clpy_tests
    "..",  # tests
    "..",  # clpy
    "headercvt",
    "headercvt")


def check_existence_of_headercvt():
    if not os.path.isfile(headercvt_abspath):
        raise FileNotFoundError("headercvt does not exist")


def exec_headercvt(workingdir, source):
    subprocess.run(
        [headercvt_abspath, "/dev/stdin", "--"],
        cwd=workingdir,
        input=source.encode(locale.getpreferredencoding()),
        timeout=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True
    )


def get_result_files(workingdir):
    with open(os.path.join(workingdir, "func_decl.pxi"), "r") as f:
        func_decl_str = f.read()
    with open(os.path.join(workingdir, "preprocessor_defines.pxi"), "r") as f:
        preprocessor_defines_str = f.read()
    with open(os.path.join(workingdir, "types.pxi"), "r") as f:
        types_str = f.read()
    print(func_decl_str)
    print(preprocessor_defines_str)
    print(types_str)
    return {
        "func_decl": func_decl_str,
        "preprocessor_defines": preprocessor_defines_str,
        "types": types_str
    }


def kick_headercvt_and_get_results(workingdir, source):
    exec_headercvt(workingdir, source)
    return get_result_files(workingdir)


def contains(result_string, match_string):
    return match_string in result_string


def compile_with(workingdir, source):
    source = """
include "func_decl.pxi"
include "preprocessor_defines.pxi"
include "types.pxi"


""" + source
    print(source)

    with open(os.path.join(workingdir, "test_case.pyx"), "w") as f:
        f.write(source)
        f.flush()
        os.fsync(f.fileno())
        try:
            subprocess.run(
                ["cython", f.name],
                cwd=workingdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode(locale.getpreferredencoding()))
            print(e.stderr.decode(locale.getpreferredencoding()))
            return False


def with_temp_wd(function):
    def impl(self):
        with tempfile.TemporaryDirectory() as wd:
            function(self, wd=wd)
    return impl


class TestHeadercvtWorking(unittest.TestCase):
    def setUp(self):
        check_existence_of_headercvt()

    @with_temp_wd
    def test_headercvt_working(self, wd):
        kick_headercvt_and_get_results(wd, "")
        # subprocess raises an exception if
        # headercvt returned non-zero exit code.
        self.assertTrue(compile_with(wd, ""))


class TestHeadercvtPreprocDefines(unittest.TestCase):
    def setUp(self):
        check_existence_of_headercvt()

    @with_temp_wd
    def test_headercvt_preproc_define_accept_case(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        #define CL_SOME_VALUE 1
        """)
        self.assertTrue(contains(
            results["preprocessor_defines"], "CL_SOME_VALUE"))
        self.assertTrue(compile_with(wd, "print(CL_SOME_VALUE)"))

    @with_temp_wd
    def test_headercvt_preproc_define_decline_case(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        #define SOME_VALUE 1
        """)
        self.assertTrue(not contains(
            results["preprocessor_defines"], "SOME_VALUE"))


class TestHeadercvtFuncDecl(unittest.TestCase):
    def setUp(self):
        check_existence_of_headercvt()

    @with_temp_wd
    def test_headercvt_funcdecl_accept_case(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        void clSomeFunction(int, void *);
        """)
        self.assertTrue(contains(
            results["func_decl"], "clSomeFunction(int, void *)"))
        self.assertTrue(compile_with(wd, "clSomeFunction(10, <void*>0)"))

    @with_temp_wd
    def test_headercvt_funcdecl_decline_case(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        void SomeFunction(int, void *);
        """)
        self.assertTrue(not contains(results["func_decl"], "SomeFunction"))


class TestHeadercvtTypes(unittest.TestCase):
    def setUp(self):
        check_existence_of_headercvt()

    @with_temp_wd
    def test_headercvt_typedef(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef int clpy_int;
        """)
        self.assertTrue(contains(results["types"], "ctypedef int clpy_int"))
        self.assertTrue(compile_with(wd, "cdef clpy_int foo = 0"))

    @with_temp_wd
    def test_headercvt_typedef_to_pointer(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef int* clpy_intptr;
        """)
        self.assertTrue(contains(
            results["types"], "ctypedef int * clpy_intptr"))
        self.assertTrue(compile_with(
            wd, "cdef clpy_intptr foo = <clpy_intptr>0"))

    @with_temp_wd
    def test_headercvt_typedef_to_tagged_struct(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef struct clpy_struct_tag{
            int member;
        } clpy_struct_t;
        """)
        self.assertTrue(contains(
            results["types"], "ctypedef struct clpy_struct_t:"))
        self.assertTrue(compile_with(
            wd, "cdef clpy_struct_t foo\nfoo.member = 0"))

    @with_temp_wd
    def test_headercvt_typedef_to_anonymous_struct(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef struct {
            int member;
        } clpy_struct_t;
        """)
        self.assertTrue(contains(
            results["types"], "ctypedef struct clpy_struct_t:"))
        self.assertTrue(compile_with(
            wd, "cdef clpy_struct_t foo\nfoo.member = 0"))

    @with_temp_wd
    def test_headercvt_typedef_to_empty_struct(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef struct {
        } clpy_empty_struct_t;
        """)
        self.assertTrue(contains(
            results["types"], "ctypedef struct clpy_empty_struct_t:"))
        self.assertTrue(compile_with(
            wd, "cdef clpy_empty_struct_t* foo"))

    @with_temp_wd
    def test_headercvt_typedef_to_struct_which_contains_an_array(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef struct clpy_struct_tag{
            int member[100];
        } clpy_struct_t;
        """)
        self.assertTrue(contains(
            results["types"], "int member[100]"))
        self.assertTrue(compile_with(
            wd, "cdef clpy_struct_t foo\nfoo.member[0] = 0"))

    @with_temp_wd
    def test_headercvt_typedef_to_struct_having_a_pointer_variable(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef struct clpy_struct_tag{
            int* ptr;
        } clpy_struct_t;
        """)
        self.assertTrue(contains(results["types"], "int *ptr"))
        self.assertTrue(compile_with(
            wd, "cdef clpy_struct_t foo\nfoo.ptr = <int*>0"))

    """
    @with_temp_wd
    def test_headercvt_typedef_to_discretely_tagged_struct(self, wd):
        kick_headercvt_and_get_results(wd, \"""
        struct clpy_struct_tag{
            int member;
        };
        typedef struct clpy_struct_tag clpy_struct_t;
        \""")
        self.assertTrue(compile_with(
            wd, "cdef clpy_struct_t foo\nfoo.member = 0"))
        # TODO(nsakabe-fixstars):
        # Make headercvt support this case of decl and update this testcase.
        # Current error:
        # cdef extern from "CL/cl.h":
        #    cdef struct clpy_struct_tag:
        #        int member
        #    ctypedef struct clpy_struct_tag clpy_struct_t
        #                                   ^
        # ------------------------------------------------------------
        #
        # types.pxi:4:36: Syntax error in struct or union definition
    """

    @with_temp_wd
    def test_headercvt_typedef_to_implicitly_declared_pointer(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef struct clpy_struct_tag *    clpy_pointer_to_struct_t;
        """)
        self.assertTrue(contains(
            results["types"], "cdef struct clpy_struct_tag"))
        self.assertTrue(contains(
            results["types"],
            "ctypedef clpy_struct_tag * clpy_pointer_to_struct_t"))
        self.assertTrue(compile_with(
            wd,
            "cdef clpy_pointer_to_struct_t foo = <clpy_pointer_to_struct_t>0"))

    @with_temp_wd
    def test_headercvt_ignore_typedef_to_function_pointer(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef void(*clpy_function_i_vstar_l_t)(int, void*, long);
        """)
        self.assertTrue(not contains(
            results["types"], "clpy_function_i_vstar_l_t"))
        self.assertTrue(compile_with(wd, ""))

    @with_temp_wd
    def test_headercvt_ignore_union_groupdecl(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef union clpy_union_tag{
            int member;
        } clpy_union_t;
        """)
        self.assertTrue(not contains(results["types"], "clpy_union_t"))
        self.assertTrue(compile_with(wd, ""))

    @with_temp_wd
    def test_headercvt_ignore_union_recorddecl(self, wd):
        kick_headercvt_and_get_results(wd, """
        union clpy_union_tag{
            int member;
        };
        """)
        self.assertTrue(compile_with(wd, ""))

    @with_temp_wd
    def test_headercvt_ignore_union_reference(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef union clpy_union_tag{
            int member;
        } clpy_union_t;
        typedef clpy_union_t clpy_typedefed_union_t;
        """)
        self.assertTrue(not contains(
            results["types"], "clpy_typedefed_union_t"))
        self.assertTrue(compile_with(wd, ""))

    @with_temp_wd
    def test_headercvt_ignore_pthread_related_groupdecl(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef struct{
            int* ptr;
        } clpy_pthread_struct_intptr;
        """)
        self.assertTrue(not contains(
            results["types"], "clpy_pthread_struct_intptr"))
        self.assertTrue(compile_with(wd, ""))

    @with_temp_wd
    def test_headercvt_ignore_pthread_related_recorddecl(self, wd):
        results = kick_headercvt_and_get_results(wd, """
        typedef struct clpy_struct_tag{
            int* ptr;
        } clpy_pthread_struct_intptr;
        """)
        self.assertTrue(not contains(
            results["types"], "clpy_pthread_struct_intptr"))
        self.assertTrue(compile_with(wd, ""))
