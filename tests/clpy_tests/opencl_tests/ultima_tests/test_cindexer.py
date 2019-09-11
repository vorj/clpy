# flake8: noqa
# TODO(vorj): When we will meet flake8 3.7.0+,
#               we should ignore only W291 for whole file
#               using --per-file-ignores .

import unittest
import utility


class TestUltimaCIndexer(unittest.TestCase):

    def test_cindexer_argument_mutation(self):
        x = utility.exec_ultima('', _clpy_header='#include <cupy/carray.hpp>') + '''
void f(CIndexer_2 ind) 
{
}
'''[1:]
        y = utility.exec_ultima(
            '''
            void f(CIndexer<2> ind){}
            ''',
            _clpy_header='#include <cupy/carray.hpp>')
        self.maxDiff = None
        self.assertEqual(x, y)

    def test_cindexer_member_function(self):
        x = utility.exec_ultima('', _clpy_header='#include <cupy/carray.hpp>') + '''
void f(CIndexer_2 ind) 
{
    ind_size;
}
'''[1:]
        y = utility.exec_ultima(
            '''
            void f(CIndexer<2> ind){
              ind.size();
            }
            ''',
            _clpy_header='#include <cupy/carray.hpp>')
        self.maxDiff = None
        self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
