"""
Copyright (c) 2014, Samsung Electronics Co.,Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of Samsung Electronics Co.,Ltd..
"""

"""
opencl4py - OpenCL cffi bindings and helper classes.
URL: https://github.com/Samsung/opencl4py
Original author: Alexey Kazantsev <a.kazantsev@samsung.com>
"""

"""
Init module.
"""

# High-level interface.
from opencl4py._py import Platforms, Context, CLRuntimeError, skip

# Low-level interface.
from opencl4py._cffi import ffi, lib, initialize

# Constants.
from opencl4py._cffi import (CL_DEVICE_TYPE_CPU,
                             CL_DEVICE_TYPE_GPU,
                             CL_DEVICE_TYPE_ACCELERATOR,
                             CL_DEVICE_TYPE_CUSTOM,
                             CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                             CL_QUEUE_PROFILING_ENABLE,
                             CL_MAP_READ,
                             CL_MAP_WRITE,
                             CL_MAP_WRITE_INVALIDATE_REGION,
                             CL_MEM_READ_WRITE,
                             CL_MEM_WRITE_ONLY,
                             CL_MEM_READ_ONLY,
                             CL_MEM_USE_HOST_PTR,
                             CL_MEM_ALLOC_HOST_PTR,
                             CL_MEM_COPY_HOST_PTR,
                             CL_PROFILING_COMMAND_QUEUED,
                             CL_PROFILING_COMMAND_SUBMIT,
                             CL_PROFILING_COMMAND_START,
                             CL_PROFILING_COMMAND_END)


def realign_array(a, align, np):
    """Returns aligned copy of the numpy array
    (useful for CL_MEM_USE_HOST_PTR buffers).

    Parameters:
        a: numpy array to create aligned array from.
        align: alignment in bytes of the new array.
        np: reference to numpy module.
    """
    if a.__array_interface__["data"][0] % align == 0:
        return a
    b = np.empty(a.nbytes + align, dtype=np.byte)
    addr = b.__array_interface__["data"][0]
    offs = 0
    if addr % align != 0:
        offs += align - (addr % align)
    b = b[offs:offs + a.nbytes].view(dtype=a.dtype)
    b.shape = a.shape
    if b.__array_interface__["data"][0] % align != 0:
        raise ValueError("Could not realign numpy array with shape [%s]" %
                         ", ".join(str(x) for x in a.shape))
    b[:] = a[:]
    return b
