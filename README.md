# ClPy: OpenCL backend for CuPy

*ClPy* is an implementation of [CuPy](https://cupy.chainer.org/)'s OpenCL backend.
In other words, ClPy enables softwares written in CuPy to work also on OpenCL devices, not only on CUDA (NVIDIA) devices.

## Current status

Current ClPy is beta version, forked from [CuPy v2.1.0](https://github.com/cupy/cupy/releases/tag/v2.1.0).
ClPy is still under development and works on only limited APIs.

* Most of [ndarray](https://docs-cupy.chainer.org/en/v2.5.0/reference/ndarray.html) are supported, but not perfectly
* Most of [universal functions](https://docs-cupy.chainer.org/en/v2.5.0/reference/ufunc.html) are supported, but not perfectly
* All [custom kernel](https://docs-cupy.chainer.org/en/v2.5.0/reference/kernel.html)s are supported.
* All BLAS APIs used by ClPy itself are supported. Other types are currently not.
* Sparse matrix, dnn, rand libraries are not supported
* half and complex are not supported
* Works on only a single device
* No multiple command queue (Stream on CUDA)
* Dockerfile and some other files are just neglected thus don't work well

Many original CuPy's tests are passed but not perfectly. See current [CuPy's test and example results](https://github.com/fixstars/ClPy/wiki/cupy_test_example_results).

[Chainer](https://chainer.org/) works with limited situation.
Some examples are confirmed to work. See current [Chainer's test and example results](https://github.com/fixstars/ClPy/wiki/chainer_test_example_results).

## Recommended system

We develop and test ClPy in following environments.

* Primary machine
	* OS: Ubuntu 16.04.4 LTS
	* CPU: Core i7-7700
	* GPU: AMD Radeon Vega Frontier Edition (Air Cooled)
	* SDK: amdgpu-pro-18.20
* Secondary machine
	* OS: Ubuntu 16.04.4 LTS
	* CPU: Core i9-7900X
	* GPU: NVIDIA TITAN V
	* SDK: CUDA 9.2

We develop ClPy with Python 3.6.5. Currently, we do not check the behavior on other versions of Python.
We recommend those environments to all ClPy users. However, reports on other environments are welcome.

## Installation

### Setup OpenCL

Install and setup OpenCL environment.

`cl.h` and OpenCL libs (`libOpenCL.so`) must be able to be included and linked without any special path settings.

For example with AMD APP SDK, you should set following environment variables:

```sh
export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${AMDAPPSDKROOT}/include
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${AMDAPPSDKROOT}/include
export LIBRARY_PATH=${LIBRARY_PATH}:${AMDAPPSDKROOT}/lib/x86_64
```

and add ldconfig on `/etc/ldconf.so.d/` and `$ sudo ldconfig`.

### Install LLVM/Clang

Current ClPy requires LLVM/Clang 4.0, 5.0, 6.0, or 7.0.
We **strongly** recommend that you build LLVM/Clang from the source codes and install it.
However, at least in Ubuntu 16.04, you can use the LLVM/Clang from the Ubuntu official package repository.
In that case, you need to set `PATH` and `CPLUS_INCLUDE_PATH` environment variables like below.

```console
# apt install clang-6.0 libclang-6.0-dev
$ export PATH=/usr/lib/llvm-6.0/bin:${PATH}
$ export CPLUS_INCLUDE_PATH=/usr/lib/llvm-6.0/include:${CPLUS_INCLUDE_PATH}
```

### Install CLBlast

ClPy depends on [CLBlast 1.4.1](https://github.com/CNugteren/CLBlast/releases/tag/1.4.1) or newer.
Install it and set the paths if needed.

### Install ClPy

After OpenCL and LLVM/Clang is successfully installed, install ClPy.
ClPy uses `make` command in build process, so if you do not have `make` , please install it before install ClPy.

```console
$ pip install cython
$ python setup.py install
```

## How to use

Run your CuPy code with `-m clpy` option ( e.g. `python -m clpy /path/to/chainer/examples/mnist/train_mnist.py -g0`).
This option adds aliases to CuPy by hooking `import cupy` and call ClPy through `cupy.foobar`.
You don't need to modify any your codes.

If you don't want to run with `-m` option, you must add `import clpy` before `import cupy` in your codes.
`import clpy` adds the aliases same as `-m clpy`.

If you want to disable such aliases, set `export CLPY_NOT_HOOK_CUPY=1` before execution.
Then, you need to replace `cupy` to `clpy` in your all codes (e.g. `import cupy` -> `import clpy`).

### Woking with Chainer

It's confirmed that ClPy works with [Chainer v3.3.0](https://github.com/chainer/chainer/tree/v3.3.0).

### Tests

```console
$ pip install pytest
$ cd tests/you/want
$ python -m pytest test_you_want.py
```

## Development

1. All source codes (including comments) and commit messages should be written in English.
2. Issues and pull requests are welcome in any languages (recommended in English or Japanese).
3. Detailed coding styles are same as [CuPy's](https://docs-cupy.chainer.org/en/stable/contribution.html#coding-guidelines). Read and follow the guidelines before submitting PRs.

## Future plan

We are developing v0.2.1beta2 for next release.

* Support all BLAS APIs
* Accelerate chainer's example performance
* Multiple devices
* Multiple CommandQueue (Stream)
* -- and other functions and/or bug fixes that someone develops and/or requests..

We also plan to update CuPy's base version to v4 or v5 after beta release.

Check [github's issues and pull requests ](https://github.com/fixstars/clpy/issues) to get latest status.

## License

MIT License (see `LICENSE` file).
