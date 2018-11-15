#!/bin/bash

# A filename to record stderrs
ERRORS_FILENAME=$WORKSPACE/erros.log


# Detailed output
set -x
# Exit immediately if an error has occurred
set -e

# Specify a python environment
export PATH="~/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell 3.6.5

# Set up a new temporary python modules environment
python -m venv venv
source venv/bin/activate


# Install dependencies
pip install -U pip
pip install Cython pytest

# Ignore occurred errors below
set +e
# Install clpy
python setup.py develop 2>&1 | tee build_log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  cat build_log >> $ERRORS_FILENAME
  exit 1
fi




# Run pytests
TEST_DIRS="
tests/clpy_tests/opencl_tests/
tests/clpy_tests/binary_tests/
tests/clpy_tests/ext_tests/
tests/clpy_tests/testing_tests/
tests/clpy_tests/io_tests/
tests/clpy_tests/padding_tests/
tests/clpy_tests/creation_tests/
tests/clpy_tests/manipulation_tests/
"

TEST_FILES="
tests/clpy_tests/core_tests/test_carray.py
tests/clpy_tests/core_tests/test_core.py
tests/clpy_tests/core_tests/test_cupy_aliased_ndarray.py
tests/clpy_tests/core_tests/test_elementwise.py
tests/clpy_tests/core_tests/test_flags.py
tests/clpy_tests/core_tests/test_function.py
tests/clpy_tests/core_tests/test_internal.py
tests/clpy_tests/core_tests/test_ndarray.py
tests/clpy_tests/core_tests/test_ndarray_contiguity.py
tests/clpy_tests/core_tests/test_ndarray_copy_and_view.py
tests/clpy_tests/core_tests/test_ndarray_elementwise_op.py
tests/clpy_tests/core_tests/test_ndarray_get.py
tests/clpy_tests/core_tests/test_ndarray_indexing.py
tests/clpy_tests/core_tests/test_ndarray_owndata.py
tests/clpy_tests/core_tests/test_ndarray_reduction.py
tests/clpy_tests/core_tests/test_ndarray_unary_op.py
tests/clpy_tests/core_tests/test_reduction.py
tests/clpy_tests/core_tests/test_scan.py
tests/clpy_tests/core_tests/test_userkernel.py
tests/clpy_tests/indexing_tests/test_insert.py
tests/clpy_tests/linalg_tests/test_product.py
tests/clpy_tests/logic_tests/test_comparison.py
tests/clpy_tests/logic_tests/test_ops.py
tests/clpy_tests/logic_tests/test_type_test.py
tests/clpy_tests/sorting_tests/test_count.py
tests/clpy_tests/statics_tests/test_correlation.py
tests/clpy_tests/statics_tests/test_meanvar.py
tests/clpy_tests/statics_tests/test_order.py
"

export CLPY_TEST_GPU_LIMIT=1

ERROR_HAS_OCCURRED=0

for d in $TEST_DIRS; do
  pushd $d
  python -m pytest  2>&1 | tee temporary_log
  if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    cat temporary_log >> $ERRORS_FILENAME
    ERROR_HAS_OCCURRED=1
  fi
  popd
done

for f in $TEST_FILES; do
  pushd $(dirname $f)
  python -m pytest $(basename $f) 2>&1 | tee temporary_log
  status=${PIPESTATUS[0]}
  if [[ status -ne 0 ]] && [[ status -ne 5 ]]; then
    cat temporary_log >> $ERRORS_FILENAME
    ERROR_HAS_OCCURRED=1
  fi
  popd
done

exit $ERROR_HAS_OCCURRED
