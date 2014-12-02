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
OpenCL cffi bindings.
"""
import cffi
import threading


# Constants
CL_CONTEXT_PLATFORM = 0x1084
CL_PLATFORM_NAME = 0x0902
CL_DEVICE_TYPE_CPU = 2
CL_DEVICE_TYPE_GPU = 4
CL_DEVICE_TYPE_ACCELERATOR = 8
CL_DEVICE_TYPE_CUSTOM = 16
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_DEVICE_TYPE = 0x1000
CL_DEVICE_NAME = 0x102B
CL_DEVICE_OPENCL_C_VERSION = 0x103D
CL_DEVICE_VENDOR_ID = 0x1001
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003
CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004
CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005
CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006
CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x1007
CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x1008
CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x1009
CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100A
CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x100B
CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C
CL_DEVICE_ADDRESS_BITS = 0x100D
CL_DEVICE_MAX_READ_IMAGE_ARGS = 0x100E
CL_DEVICE_MAX_WRITE_IMAGE_ARGS = 0x100F
CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010
CL_DEVICE_IMAGE2D_MAX_WIDTH = 0x1011
CL_DEVICE_IMAGE2D_MAX_HEIGHT = 0x1012
CL_DEVICE_IMAGE3D_MAX_WIDTH = 0x1013
CL_DEVICE_IMAGE3D_MAX_HEIGHT = 0x1014
CL_DEVICE_IMAGE3D_MAX_DEPTH = 0x1015
CL_DEVICE_IMAGE_SUPPORT = 0x1016
CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017
CL_DEVICE_MAX_SAMPLERS = 0x1018
CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019
CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101A
CL_DEVICE_SINGLE_FP_CONFIG = 0x101B
CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101C
CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101D
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101E
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020
CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021
CL_DEVICE_LOCAL_MEM_TYPE = 0x1022
CL_DEVICE_LOCAL_MEM_SIZE = 0x1023
CL_DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024
CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025
CL_DEVICE_ENDIAN_LITTLE = 0x1026
CL_DEVICE_AVAILABLE = 0x1027
CL_DEVICE_COMPILER_AVAILABLE = 0x1028
CL_DEVICE_EXECUTION_CAPABILITIES = 0x1029
CL_DEVICE_QUEUE_PROPERTIES = 0x102A
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_PROFILE = 0x102E
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_PLATFORM = 0x1031
CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032
CL_DEVICE_HALF_FP_CONFIG = 0x1033
CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034
CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035
CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR = 0x1036
CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT = 0x1037
CL_DEVICE_NATIVE_VECTOR_WIDTH_INT = 0x1038
CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG = 0x1039
CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103A
CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103B
CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103C
CL_DEVICE_OPENCL_C_VERSION = 0x103D
CL_DEVICE_LINKER_AVAILABLE = 0x103E
CL_DEVICE_BUILT_IN_KERNELS = 0x103F
CL_DEVICE_IMAGE_MAX_BUFFER_SIZE = 0x1040
CL_DEVICE_IMAGE_MAX_ARRAY_SIZE = 0x1041
CL_DEVICE_PARENT_DEVICE = 0x1042
CL_DEVICE_PARTITION_MAX_SUB_DEVICES = 0x1043
CL_DEVICE_PARTITION_PROPERTIES = 0x1044
CL_DEVICE_PARTITION_AFFINITY_DOMAIN = 0x1045
CL_DEVICE_PARTITION_TYPE = 0x1046
CL_DEVICE_REFERENCE_COUNT = 0x1047
CL_DEVICE_PREFERRED_INTEROP_USER_SYNC = 0x1048
CL_DEVICE_PRINTF_BUFFER_SIZE = 0x1049
CL_DEVICE_IMAGE_PITCH_ALIGNMENT = 0x104A
CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT = 0x104B
CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS = 0x1056
CL_DEVICE_PIPE_MAX_PACKET_SIZE = 0x1057
CL_DEVICE_SVM_CAPABILITIES = 0x1053
CL_DEVICE_SVM_COARSE_GRAIN_BUFFER = 1
CL_DEVICE_SVM_FINE_GRAIN_BUFFER = 2
CL_DEVICE_SVM_FINE_GRAIN_SYSTEM = 4
CL_DEVICE_SVM_ATOMICS = 8
CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT = 0x1058
CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT = 0x1059
CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT = 0x105A
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = 1
CL_QUEUE_PROFILING_ENABLE = 2
CL_QUEUE_ON_DEVICE = 4
CL_QUEUE_ON_DEVICE_DEFAULT = 8
CL_QUEUE_PROPERTIES = 0x1093
CL_QUEUE_SIZE = 0x1094
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
CL_MEM_HOST_NO_ACCESS = 512
CL_MEM_SVM_FINE_GRAIN_BUFFER = 1024
CL_MEM_SVM_ATOMICS = 2048
CL_PROFILING_COMMAND_QUEUED = 0x1280
CL_PROFILING_COMMAND_SUBMIT = 0x1281
CL_PROFILING_COMMAND_START = 0x1282
CL_PROFILING_COMMAND_END = 0x1283
CL_PROGRAM_REFERENCE_COUNT = 0x1160
CL_PROGRAM_CONTEXT = 0x1161
CL_PROGRAM_NUM_DEVICES = 0x1162
CL_PROGRAM_DEVICES = 0x1163
CL_PROGRAM_SOURCE = 0x1164
CL_PROGRAM_BINARY_SIZES = 0x1165
CL_PROGRAM_BINARIES = 0x1166
CL_PROGRAM_NUM_KERNELS = 0x1167
CL_PROGRAM_KERNEL_NAMES = 0x1168
CL_KERNEL_FUNCTION_NAME = 0x1190
CL_KERNEL_NUM_ARGS = 0x1191
CL_KERNEL_REFERENCE_COUNT = 0x1192
CL_KERNEL_CONTEXT = 0x1193
CL_KERNEL_PROGRAM = 0x1194
CL_KERNEL_ATTRIBUTES = 0x1195
CL_BUFFER_CREATE_TYPE_REGION = 0x1220
CL_KERNEL_GLOBAL_WORK_SIZE = 0x11B5
CL_KERNEL_WORK_GROUP_SIZE = 0x11B0
CL_KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11B1
CL_KERNEL_LOCAL_MEM_SIZE = 0x11B2
CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3
CL_KERNEL_PRIVATE_MEM_SIZE = 0x11B4


# Error codes
CL_SUCCESS = 0
CL_DEVICE_NOT_FOUND = -1
CL_DEVICE_NOT_AVAILABLE = -2
CL_COMPILER_NOT_AVAILABLE = -3
CL_MEM_OBJECT_ALLOCATION_FAILURE = -4
CL_OUT_OF_RESOURCES = -5
CL_OUT_OF_HOST_MEMORY = -6
CL_PROFILING_INFO_NOT_AVAILABLE = -7
CL_MEM_COPY_OVERLAP = -8
CL_IMAGE_FORMAT_MISMATCH = -9
CL_IMAGE_FORMAT_NOT_SUPPORTED = -10
CL_BUILD_PROGRAM_FAILURE = -11
CL_MAP_FAILURE = -12
CL_MISALIGNED_SUB_BUFFER_OFFSET = -13
CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14
CL_COMPILE_PROGRAM_FAILURE = -15
CL_LINKER_NOT_AVAILABLE = -16
CL_LINK_PROGRAM_FAILURE = -17
CL_DEVICE_PARTITION_FAILED = -18
CL_KERNEL_ARG_INFO_NOT_AVAILABLE = -19

CL_INVALID_VALUE = -30
CL_INVALID_DEVICE_TYPE = -31
CL_INVALID_PLATFORM = -32
CL_INVALID_DEVICE = -33
CL_INVALID_CONTEXT = -34
CL_INVALID_QUEUE_PROPERTIES = -35
CL_INVALID_COMMAND_QUEUE = -36
CL_INVALID_HOST_PTR = -37
CL_INVALID_MEM_OBJECT = -38
CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39
CL_INVALID_IMAGE_SIZE = -40
CL_INVALID_SAMPLER = -41
CL_INVALID_BINARY = -42
CL_INVALID_BUILD_OPTIONS = -43
CL_INVALID_PROGRAM = -44
CL_INVALID_PROGRAM_EXECUTABLE = -45
CL_INVALID_KERNEL_NAME = -46
CL_INVALID_KERNEL_DEFINITION = -47
CL_INVALID_KERNEL = -48
CL_INVALID_ARG_INDEX = -49
CL_INVALID_ARG_VALUE = -50
CL_INVALID_ARG_SIZE = -51
CL_INVALID_KERNEL_ARGS = -52
CL_INVALID_WORK_DIMENSION = -53
CL_INVALID_WORK_GROUP_SIZE = -54
CL_INVALID_WORK_ITEM_SIZE = -55
CL_INVALID_GLOBAL_OFFSET = -56
CL_INVALID_EVENT_WAIT_LIST = -57
CL_INVALID_EVENT = -58
CL_INVALID_OPERATION = -59
CL_INVALID_GL_OBJECT = -60
CL_INVALID_BUFFER_SIZE = -61
CL_INVALID_MIP_LEVEL = -62
CL_INVALID_GLOBAL_WORK_SIZE = -63
CL_INVALID_PROPERTY = -64
CL_INVALID_IMAGE_DESCRIPTOR = -65
CL_INVALID_COMPILER_OPTIONS = -66
CL_INVALID_LINKER_OPTIONS = -67
CL_INVALID_DEVICE_PARTITION_COUNT = -68
CL_INVALID_PIPE_SIZE = -69
CL_INVALID_DEVICE_QUEUE = -70


#: ffi parser
ffi = cffi.FFI()


#: Loaded shared library
lib = None


#: cffi NULL pointer
NULL = ffi.NULL


#: Lock
lock = threading.Lock()


def _initialize(backends):
    global lib
    if lib is not None:
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
    typedef uint64_t cl_queue_properties;
    typedef uint64_t cl_mem_flags;
    typedef uint32_t cl_bool;
    typedef uint64_t cl_map_flags;
    typedef uint32_t cl_profiling_info;
    typedef uint32_t cl_buffer_create_type;
    typedef uint64_t cl_svm_mem_flags;

    typedef void* cl_platform_id;
    typedef void* cl_device_id;
    typedef void* cl_context;
    typedef void* cl_program;
    typedef void* cl_kernel;
    typedef void* cl_command_queue;
    typedef void* cl_mem;
    typedef void* cl_event;

    typedef intptr_t cl_context_properties;
    typedef intptr_t cl_pipe_properties;

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
    cl_command_queue clCreateCommandQueueWithProperties(
                                    cl_context context,
                                    cl_device_id device,
                                    const cl_queue_properties *properties,
                                    cl_int *errcode_ret);
    cl_int clReleaseCommandQueue(cl_command_queue command_queue);

    cl_mem clCreateBuffer(cl_context context,
                          cl_mem_flags flags,
                          size_t size,
                          void *host_ptr,
                          cl_int *errcode_ret);
    cl_mem clCreateSubBuffer(cl_mem buffer,
                             cl_mem_flags flags,
                             cl_buffer_create_type buffer_create_type,
                             const void *buffer_create_info,
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
    cl_int clEnqueueCopyBuffer(cl_command_queue command_queue,
                               cl_mem src_buffer,
                               cl_mem dst_buffer,
                               size_t src_offset,
                               size_t dst_offset,
                               size_t size,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event);
    cl_int clEnqueueCopyBufferRect(cl_command_queue command_queue,
                                   cl_mem src_buffer,
                                   cl_mem dst_buffer,
                                   const size_t *src_origin,
                                   const size_t *dst_origin,
                                   const size_t *region,
                                   size_t src_row_pitch,
                                   size_t src_slice_pitch,
                                   size_t dst_row_pitch,
                                   size_t dst_slice_pitch,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list,
                                   cl_event *event);
    cl_int clEnqueueFillBuffer(cl_command_queue command_queue,
                               cl_mem buffer,
                               const void *pattern,
                               size_t pattern_size,
                               size_t offset,
                               size_t size,
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

    cl_mem clCreatePipe(cl_context context,
                        cl_mem_flags flags,
                        cl_uint pipe_packet_size,
                        cl_uint pipe_max_packets,
                        const cl_pipe_properties *properties,
                        cl_int *errcode_ret);

    void *clSVMAlloc(cl_context context,
                     cl_svm_mem_flags flags,
                     size_t size,
                     unsigned int alignment);
    void clSVMFree(cl_context context,
                   void *svm_pointer);
    cl_int clEnqueueSVMMap(cl_command_queue command_queue,
                           cl_bool blocking_map,
                           cl_map_flags map_flags,
                           void *svm_ptr,
                           size_t size,
                           cl_uint num_events_in_wait_list,
                           const cl_event *event_wait_list,
                           cl_event *event);
    cl_int clEnqueueSVMUnmap(cl_command_queue command_queue,
                             void *svm_ptr,
                             cl_uint  num_events_in_wait_list,
                             const cl_event *event_wait_list,
                             cl_event *event);
    cl_int clSetKernelArgSVMPointer(cl_kernel kernel,
                                    cl_uint arg_index,
                                    const void *arg_value);
    cl_int clEnqueueSVMMemcpy(cl_command_queue command_queue,
                              cl_bool blocking_copy,
                              void *dst_ptr,
                              const void *src_ptr,
                              size_t size,
                              cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event *event);
    cl_int clEnqueueSVMMemFill(cl_command_queue command_queue,
                               void *svm_ptr,
                               const void *pattern,
                               size_t pattern_size,
                               size_t size,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event);
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
        ffi = cffi.FFI()  # reset before raise
        raise OSError("Could not load OpenCL library")


def initialize(backends=("libOpenCL.so", "OpenCL.dll")):
    global lib
    if lib is not None:
        return
    global lock
    with lock:
        _initialize(backends)
