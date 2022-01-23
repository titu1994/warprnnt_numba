# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging as pylogger
import operator
import os
from packaging import version
from typing import Optional

# Prevent Numba CUDA logs from showing at info level
cuda_logger = pylogger.getLogger('numba.cuda.cudadrv.driver')
cuda_logger.setLevel(pylogger.ERROR)  # only show error

__NUMBA_DEFAULT_MINIMUM_VERSION__ = "0.53.0"
__NUMBA_MINIMUM_VERSION__ = os.environ.get("NEMO_NUMBA_MINVER", __NUMBA_DEFAULT_MINIMUM_VERSION__)

NUMBA_INSTALLATION_MESSAGE = (
    "Could not import `numba`.\n"
    "Please install numba in one of the following ways."
    "1) If using conda, simply install it with conda using `conda install -c numba numba`\n"
    "2) If using pip (not recommended), `pip install --upgrade numba`\n"
    "followed by `export NUMBAPRO_LIBDEVICE='/usr/local/cuda/nvvm/libdevice/'` and \n"
    "`export NUMBAPRO_NVVM='/usr/local/cuda/nvvm/lib64/libnvvm.so'`.\n"
    "It is advised to always install numba using conda only, "
    "as pip installations might interfere with other libraries such as llvmlite.\n"
    "If pip install does not work, you can also try adding `--ignore-installed` to the pip command,\n"
    "but this is not advised."
)

STRICT_NUMBA_COMPAT_CHECK = True

# Get environment key if available
if 'STRICT_NUMBA_COMPAT_CHECK' in os.environ:
    check_str = os.environ.get('STRICT_NUMBA_COMPAT_CHECK')
    check_bool = str(check_str).lower() in ("yes", "true", "t", "1")
    STRICT_NUMBA_COMPAT_CHECK = check_bool


def import_class_by_path(path: str):
    """
    Recursive import of class by path string.
    """
    paths = path.split('.')
    path = ".".join(paths[:-1])
    class_name = paths[-1]
    mod = __import__(path, fromlist=[class_name])
    mod = getattr(mod, class_name)
    return mod


def check_lib_version(lib_name: str, checked_version: str, operator) -> (Optional[bool], str):
    """
    Checks if a library is installed, and if it is, checks the operator(lib.__version__, checked_version) as a result.
    This bool result along with a string analysis of result is returned.

    If the library is not installed at all, then returns None instead, along with a string explaining
    that the library is not installed

    Args:
        lib_name: lower case str name of the library that must be imported.
        checked_version: semver string that is compared against lib.__version__.
        operator: binary callable function func(a, b) -> bool; that compares lib.__version__ against version in
            some manner. Must return a boolean.

    Returns:
        A tuple of results:
        -   Bool or None. Bool if the library could be imported, and the result of
            operator(lib.__version__, checked_version) or False if __version__ is not implemented in lib.
            None is passed if the library is not installed at all.
        -   A string analysis of the check.
    """
    try:
        if '.' in lib_name:
            mod = import_class_by_path(lib_name)
        else:
            mod = __import__(lib_name)

        if hasattr(mod, '__version__'):
            lib_ver = version.Version(mod.__version__)
            match_ver = version.Version(checked_version)

            if operator(lib_ver, match_ver):
                msg = f"Lib {lib_name} version is satisfied !"
                return True, msg
            else:
                msg = (
                    f"Lib {lib_name} version ({lib_ver}) is not {operator.__name__} than required version {checked_version}.\n"
                    f"Please upgrade the lib using either pip or conda to the latest version."
                )
                return False, msg
        else:
            msg = (
                f"Lib {lib_name} does not implement __version__ in its init file. "
                f"Could not check version compatibility."
            )
            return False, msg
    except (ImportError, ModuleNotFoundError):
        pass

    msg = f"Lib {lib_name} has not been installed. Please use pip or conda to install this package."
    return None, msg


def is_numba_compat_strict() -> bool:
    """
    Returns strictness level of numba cuda compatibility checks.

    If value is true, numba cuda compatibility matrix must be satisfied.
    If value is false, only cuda availability is checked, not compatibility.
    Numba Cuda may still compile and run without issues in such a case, or it may fail.
    """
    return STRICT_NUMBA_COMPAT_CHECK


def set_numba_compat_strictness(strict: bool):
    """
    Sets the strictness level of numba cuda compatibility checks.

    If value is true, numba cuda compatibility matrix must be satisfied.
    If value is false, only cuda availability is checked, not compatibility.
    Numba Cuda may still compile and run without issues in such a case, or it may fail.

    Args:
        strict: bool value, whether to enforce strict compatibility checks or relax them.
    """
    global STRICT_NUMBA_COMPAT_CHECK
    STRICT_NUMBA_COMPAT_CHECK = strict


@contextlib.contextmanager
def with_numba_compat_strictness(strict: bool):
    initial_strictness = is_numba_compat_strict()
    set_numba_compat_strictness(strict=strict)
    yield
    set_numba_compat_strictness(strict=initial_strictness)


def numba_cpu_is_supported(min_version: str) -> bool:
    """
    Tests if an appropriate version of numba is installed.

    Args:
        min_version: The minimum version of numba that is required.

    Returns:
        bool, whether numba CPU supported with this current installation or not.
    """
    module_available, msg = check_lib_version('numba', checked_version=min_version, operator=operator.ge)

    # If numba is not installed
    if module_available is None:
        return False
    else:
        return True


def numba_cuda_is_supported(min_version: str) -> bool:
    """
    Tests if an appropriate version of numba is installed, and if it is,
    if cuda is supported properly within it.

    Args:
        min_version: The minimum version of numba that is required.

    Returns:
        bool, whether cuda is supported with this current installation or not.
    """
    module_available = numba_cpu_is_supported(min_version)

    # If numba is not installed
    if module_available is None:
        return False

    # If numba version is installed and available
    if module_available is True:
        from numba import cuda

        # this method first arrived in 0.53, and that's the minimum version required
        if hasattr(cuda, 'is_supported_version'):
            try:
                cuda_available = cuda.is_available()
                if cuda_available:
                    cuda_compatible = cuda.is_supported_version()
                else:
                    cuda_compatible = False

                if is_numba_compat_strict():
                    return cuda_available and cuda_compatible
                else:
                    return cuda_available

            except OSError:
                # dlopen(libcudart.dylib) might fail if CUDA was never installed in the first place.
                return False
        else:
            # assume cuda is supported, but it may fail due to CUDA incompatibility
            return False

    else:
        return False


def skip_numba_cuda_test_if_unsupported(min_version: str):
    """
    Helper method to skip pytest test case if numba cuda is not supported.

    Args:
        min_version: The minimum version of numba that is required.
    """
    numba_cuda_support = numba_cuda_is_supported(min_version)
    if not numba_cuda_support:
        import pytest

        pytest.skip(f"Numba cuda test is being skipped. Minimum version required : {min_version}")
