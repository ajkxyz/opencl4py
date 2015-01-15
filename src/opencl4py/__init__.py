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
                             CL_QUEUE_ON_DEVICE,
                             CL_QUEUE_ON_DEVICE_DEFAULT,
                             CL_QUEUE_PROPERTIES,
                             CL_QUEUE_SIZE,
                             CL_MAP_READ,
                             CL_MAP_WRITE,
                             CL_MAP_WRITE_INVALIDATE_REGION,
                             CL_MEM_READ_WRITE,
                             CL_MEM_WRITE_ONLY,
                             CL_MEM_READ_ONLY,
                             CL_MEM_USE_HOST_PTR,
                             CL_MEM_ALLOC_HOST_PTR,
                             CL_MEM_COPY_HOST_PTR,
                             CL_MEM_HOST_NO_ACCESS,
                             CL_MEM_SVM_FINE_GRAIN_BUFFER,
                             CL_MEM_SVM_ATOMICS,
                             CL_DEVICE_SVM_COARSE_GRAIN_BUFFER,
                             CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
                             CL_DEVICE_SVM_FINE_GRAIN_SYSTEM,
                             CL_DEVICE_SVM_ATOMICS,
                             CL_PROFILING_COMMAND_QUEUED,
                             CL_PROFILING_COMMAND_SUBMIT,
                             CL_PROFILING_COMMAND_START,
                             CL_PROFILING_COMMAND_END,

                             CL_SUCCESS,
                             CL_DEVICE_NOT_FOUND,
                             CL_DEVICE_NOT_AVAILABLE,
                             CL_COMPILER_NOT_AVAILABLE,
                             CL_MEM_OBJECT_ALLOCATION_FAILURE,
                             CL_OUT_OF_RESOURCES,
                             CL_OUT_OF_HOST_MEMORY,
                             CL_PROFILING_INFO_NOT_AVAILABLE,
                             CL_MEM_COPY_OVERLAP,
                             CL_IMAGE_FORMAT_MISMATCH,
                             CL_IMAGE_FORMAT_NOT_SUPPORTED,
                             CL_BUILD_PROGRAM_FAILURE,
                             CL_MAP_FAILURE,
                             CL_MISALIGNED_SUB_BUFFER_OFFSET,
                             CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
                             CL_COMPILE_PROGRAM_FAILURE,
                             CL_LINKER_NOT_AVAILABLE,
                             CL_LINK_PROGRAM_FAILURE,
                             CL_DEVICE_PARTITION_FAILED,
                             CL_KERNEL_ARG_INFO_NOT_AVAILABLE,

                             CL_INVALID_VALUE,
                             CL_INVALID_DEVICE_TYPE,
                             CL_INVALID_PLATFORM,
                             CL_INVALID_DEVICE,
                             CL_INVALID_CONTEXT,
                             CL_INVALID_QUEUE_PROPERTIES,
                             CL_INVALID_COMMAND_QUEUE,
                             CL_INVALID_HOST_PTR,
                             CL_INVALID_MEM_OBJECT,
                             CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
                             CL_INVALID_IMAGE_SIZE,
                             CL_INVALID_SAMPLER,
                             CL_INVALID_BINARY,
                             CL_INVALID_BUILD_OPTIONS,
                             CL_INVALID_PROGRAM,
                             CL_INVALID_PROGRAM_EXECUTABLE,
                             CL_INVALID_KERNEL_NAME,
                             CL_INVALID_KERNEL_DEFINITION,
                             CL_INVALID_KERNEL,
                             CL_INVALID_ARG_INDEX,
                             CL_INVALID_ARG_VALUE,
                             CL_INVALID_ARG_SIZE,
                             CL_INVALID_KERNEL_ARGS,
                             CL_INVALID_WORK_DIMENSION,
                             CL_INVALID_WORK_GROUP_SIZE,
                             CL_INVALID_WORK_ITEM_SIZE,
                             CL_INVALID_GLOBAL_OFFSET,
                             CL_INVALID_EVENT_WAIT_LIST,
                             CL_INVALID_EVENT,
                             CL_INVALID_OPERATION,
                             CL_INVALID_GL_OBJECT,
                             CL_INVALID_BUFFER_SIZE,
                             CL_INVALID_MIP_LEVEL,
                             CL_INVALID_GLOBAL_WORK_SIZE,
                             CL_INVALID_PROPERTY,
                             CL_INVALID_IMAGE_DESCRIPTOR,
                             CL_INVALID_COMPILER_OPTIONS,
                             CL_INVALID_LINKER_OPTIONS,
                             CL_INVALID_DEVICE_PARTITION_COUNT,
                             CL_INVALID_PIPE_SIZE,
                             CL_INVALID_DEVICE_QUEUE)


def eq_addr(a, b):
    """Compares addresses of the two numpy arrays.
    """
    return a.__array_interface__["data"][0] == b.__array_interface__["data"][0]


def realign_array(a, align, np):
    """Returns aligned copy of the numpy array with continuous memory layout.
    (useful for CL_MEM_USE_HOST_PTR buffers).

    Parameters:
        a: numpy array to create aligned array from.
        align: alignment in bytes of the new array.
        np: reference to numpy module.
    """
    if a.__array_interface__["data"][0] % align == 0 and eq_addr(a, a.ravel()):
        return a
    b = np.empty(a.nbytes + align, dtype=np.byte)
    addr = b.__array_interface__["data"][0]
    offs = 0
    if addr % align != 0:
        offs += align - (addr % align)
    b = b[offs:offs + a.nbytes].view(dtype=a.dtype)
    b.shape = a.shape
    if b.__array_interface__["data"][0] % align != 0:
        raise ValueError("Could not realign numpy array with shape %s" %
                         str(a.shape))
    b[:] = a[:]
    return b
