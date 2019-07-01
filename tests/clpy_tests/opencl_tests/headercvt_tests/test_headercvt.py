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

    def test_headercvt_funcdecl_decline_case(self):
        results = kick_headercvt_and_get_results("""
        void SomeFunction(int, void *);
        """)
        self.assertTrue(not contains(results["func_decl"], "SomeFunction"))

