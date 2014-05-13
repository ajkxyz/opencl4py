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
URL: https://github.com/ajkxyz/opencl4py
Original author: Alexey Kazantsev <a.kazantsev@samsung.com>
"""

"""
OpenCL cffi bindings.
"""
import cffi
import threading


# Constants
CL_CONTEXT_PLATFORM = 0x1084
CL_DEVICE_TYPE_CPU = 2
CL_DEVICE_TYPE_GPU = 4
CL_DEVICE_TYPE_ACCELERATOR = 8
CL_DEVICE_TYPE_CUSTOM = 16
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_DEVICE_TYPE = 0x1000
CL_DEVICE_NAME = 0x102B
CL_DEVICE_OPENCL_C_VERSION = 0x103D
CL_PLATFORM_NAME = 0x0902
CL_DEVICE_VENDOR = 0x102C
CL_DEVICE_VENDOR_ID = 0x1001
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = 1
CL_QUEUE_PROFILING_ENABLE = 2
CL_PROGRAM_BUILD_LOG = 0x1183
CL_MAP_READ = 1
CL_MAP_WRITE = 2
CL_MAP_WRITE_INVALIDATE_REGION = 4
CL_MEM_READ_WRITE = 1
CL_MEM_WRITE_ONLY = 2
CL_MEM_READ_ONLY = 4
CL_MEM_USE_HOST_PTR = 8
CL_MEM_ALLOC_HOST_PTR = 16
CL_MEM_COPY_HOST_PTR = 32
CL_PROFILING_COMMAND_QUEUED = 0x1280
CL_PROFILING_COMMAND_SUBMIT = 0x1281
CL_PROFILING_COMMAND_START = 0x1282
CL_PROFILING_COMMAND_END = 0x1283


#: ffi parser
ffi = cffi.FFI()


#: Loaded shared library
lib = None


#: cffi NULL pointer
NULL = ffi.NULL


#: Lock
lock = threading.Lock()


def initialize(backends=("libOpenCL.so", "OpenCL.dll")):
    global lib
    if lib is not None:
        return
    global lock
    lock.acquire()
    if lib is not None:
        lock.release()
        return
    # C function definitions
    src = """
    typedef int32_t cl_int;
    typedef uint32_t cl_uint;
    typedef uint64_t cl_ulong;
    typedef uint64_t cl_device_type;
    typedef uint32_t cl_platform_info;
    typedef uint32_t cl_device_info;
    typedef uint32_t cl_program_build_info;
    typedef cl_uint  cl_program_info;
    typedef cl_uint  cl_kernel_info;
    typedef uint32_t cl_kernel_work_group_info;
    typedef uint64_t cl_command_queue_properties;
    typedef uint64_t cl_mem_flags;
    typedef uint32_t cl_bool;
    typedef uint64_t cl_map_flags;
    typedef uint32_t cl_profiling_info;

    typedef void* cl_platform_id;
    typedef void* cl_device_id;
    typedef void* cl_context;
    typedef void* cl_program;
    typedef void* cl_kernel;
    typedef void* cl_command_queue;
    typedef void* cl_mem;
    typedef void* cl_event;

    typedef intptr_t cl_context_properties;

    cl_int clGetPlatformIDs(cl_uint num_entries,
                            cl_platform_id *platforms,
                            cl_uint *num_platforms);
    cl_int clGetDeviceIDs(cl_platform_id  platform,
                          cl_device_type device_type,
                          cl_uint num_entries,
                          cl_device_id *devices,
                          cl_uint *num_devices);

    cl_int clGetPlatformInfo(cl_platform_id platform,
                             cl_platform_info param_name,
                             size_t param_value_size,
                             void *param_value,
                             size_t *param_value_size_ret);
    cl_int clGetDeviceInfo(cl_device_id device,
                           cl_device_info param_name,
                           size_t param_value_size,
                           void *param_value,
                           size_t *param_value_size_ret);

    cl_context clCreateContext(const cl_context_properties *properties,
                               cl_uint num_devices,
                               const cl_device_id *devices,
                               void *pfn_notify,
                               void *user_data,
                               cl_int *errcode_ret);
    cl_int clReleaseContext(cl_context context);

    cl_program clCreateProgramWithSource(cl_context context,
                                         cl_uint count,
                                         const char **strings,
                                         const size_t *lengths,
                                         cl_int *errcode_ret);

    cl_program clCreateProgramWithBinary(cl_context context,
                                         cl_uint num_devices,
                                         const cl_device_id *device_list,
                                         const size_t *lengths,
                                         const unsigned char **binaries,
                                         cl_int *binary_status,
                                         cl_int *errcode_ret);

    cl_int clReleaseProgram(cl_program program);
    cl_int clBuildProgram(cl_program program,
                          cl_uint num_devices,
                          const cl_device_id *device_list,
                          const char *options,
                          void *pfn_notify,
                          void *user_data);
    cl_int clGetProgramBuildInfo(cl_program program,
                                 cl_device_id device,
                                 cl_program_build_info param_name,
                                 size_t param_value_size,
                                 void *param_value,
                                 size_t *param_value_size_ret);

    cl_int clGetProgramInfo(cl_program program,
                            cl_program_info param_name,
                            size_t param_value_size,
                            void *param_value,
                            size_t *param_value_size_ret);

    cl_kernel clCreateKernel(cl_program program,
                             const char *kernel_name,
                             cl_int *errcode_ret);
    cl_int clReleaseKernel(cl_kernel kernel);
    cl_int clGetKernelInfo(cl_kernel kernel,
                           cl_kernel_info param_name,
                           size_t param_value_size,
                           void *param_value,
                           size_t *param_value_size_ret);

    cl_int clGetKernelWorkGroupInfo(cl_kernel kernel,
                                    cl_device_id device,
                                    cl_kernel_work_group_info param_name,
                                    size_t param_value_size,
                                    void *param_value,
                                    size_t *param_value_size_ret);
    cl_int clSetKernelArg(cl_kernel kernel,
                          cl_uint arg_index,
                          size_t arg_size,
                          const void *arg_value);

    cl_command_queue clCreateCommandQueue(
                                    cl_context context,
                                    cl_device_id device,
                                    cl_command_queue_properties properties,
                                    cl_int *errcode_ret);
    cl_int clReleaseCommandQueue(cl_command_queue command_queue);

    cl_mem clCreateBuffer(cl_context context,
                          cl_mem_flags flags,
                          size_t size,
                          void *host_ptr,
                          cl_int *errcode_ret);
    cl_int clReleaseMemObject(cl_mem memobj);
    void* clEnqueueMapBuffer(cl_command_queue command_queue,
                             cl_mem buffer,
                             cl_bool blocking_map,
                             cl_map_flags map_flags,
                             size_t offset,
                             size_t size,
                             cl_uint num_events_in_wait_list,
                             const cl_event *event_wait_list,
                             cl_event *event,
                             cl_int *errcode_ret);
    cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue,
                                   cl_mem memobj,
                                   void *mapped_ptr,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list,
                                   cl_event *event);
    cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
                               cl_mem buffer,
                               cl_bool blocking_read,
                               size_t offset,
                               size_t size,
                               void *ptr,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event);
    cl_int clEnqueueWriteBuffer(cl_command_queue command_queue,
                                cl_mem buffer,
                                cl_bool blocking_write,
                                size_t offset,
                                size_t size,
                                const void *ptr,
                                cl_uint num_events_in_wait_list,
                                const cl_event *event_wait_list,
                                cl_event *event);

    cl_int clWaitForEvents(cl_uint num_events,
                           const cl_event *event_list);
    cl_int clReleaseEvent(cl_event event);

    cl_int clFlush(cl_command_queue command_queue);
    cl_int clFinish(cl_command_queue command_queue);

    cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
                                  cl_kernel kernel,
                                  cl_uint work_dim,
                                  const size_t *global_work_offset,
                                  const size_t *global_work_size,
                                  const size_t *local_work_size,
                                  cl_uint num_events_in_wait_list,
                                  const cl_event *event_wait_list,
                                  cl_event *event);

    cl_int clGetEventProfilingInfo(cl_event event,
                                   cl_profiling_info param_name,
                                   size_t param_value_size,
                                   void *param_value,
                                   size_t *param_value_size_ret);
    """

    # Parse
    global ffi
    ffi.cdef(src)

    # Load library
    for libnme in backends:
        try:
            lib = ffi.dlopen(libnme)
            break
        except OSError:
            pass
    else:
        lock.release()
        raise OSError("Could not load OpenCL library")
    lock.release()
