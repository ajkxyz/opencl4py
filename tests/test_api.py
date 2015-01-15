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
Tests some of the api in opencl4py package.
"""
import unittest
import logging
import opencl4py as cl
import os


class Test(unittest.TestCase):
    def setUp(self):
        self.old_env = os.environ.get("PYOPENCL_CTX")
        if self.old_env is None:
            os.environ["PYOPENCL_CTX"] = "0:0"
        self.src_test = (
            """
            #include "test.cl"
            """)
        self.include_dirs = ("", os.path.dirname(__file__), ".")

    def tearDown(self):
        if self.old_env is None:
            del os.environ["PYOPENCL_CTX"]
        else:
            os.environ["PYOPENCL_CTX"] = self.old_env
        del self.old_env

    def test_constants(self):
        self.assertEqual(cl.CL_DEVICE_TYPE_CPU, 2)
        self.assertEqual(cl.CL_DEVICE_TYPE_GPU, 4)
        self.assertEqual(cl.CL_DEVICE_TYPE_ACCELERATOR, 8)
        self.assertEqual(cl.CL_DEVICE_TYPE_CUSTOM, 16)
        self.assertEqual(cl.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 1)
        self.assertEqual(cl.CL_QUEUE_PROFILING_ENABLE, 2)
        self.assertEqual(cl.CL_QUEUE_ON_DEVICE, 4)
        self.assertEqual(cl.CL_QUEUE_ON_DEVICE_DEFAULT, 8)
        self.assertEqual(cl.CL_QUEUE_PROPERTIES, 0x1093)
        self.assertEqual(cl.CL_QUEUE_SIZE, 0x1094)
        self.assertEqual(cl.CL_MAP_READ, 1)
        self.assertEqual(cl.CL_MAP_WRITE, 2)
        self.assertEqual(cl.CL_MAP_WRITE_INVALIDATE_REGION, 4)
        self.assertEqual(cl.CL_MEM_READ_WRITE, 1)
        self.assertEqual(cl.CL_MEM_WRITE_ONLY, 2)
        self.assertEqual(cl.CL_MEM_READ_ONLY, 4)
        self.assertEqual(cl.CL_MEM_USE_HOST_PTR, 8)
        self.assertEqual(cl.CL_MEM_ALLOC_HOST_PTR, 16)
        self.assertEqual(cl.CL_MEM_COPY_HOST_PTR, 32)
        self.assertEqual(cl.CL_MEM_HOST_NO_ACCESS, 512)
        self.assertEqual(cl.CL_MEM_SVM_FINE_GRAIN_BUFFER, 1024)
        self.assertEqual(cl.CL_MEM_SVM_ATOMICS, 2048)
        self.assertEqual(cl.CL_DEVICE_SVM_COARSE_GRAIN_BUFFER, 1)
        self.assertEqual(cl.CL_DEVICE_SVM_FINE_GRAIN_BUFFER, 2)
        self.assertEqual(cl.CL_DEVICE_SVM_FINE_GRAIN_SYSTEM, 4)
        self.assertEqual(cl.CL_DEVICE_SVM_ATOMICS, 8)
        self.assertEqual(cl.CL_PROFILING_COMMAND_QUEUED, 0x1280)
        self.assertEqual(cl.CL_PROFILING_COMMAND_SUBMIT, 0x1281)
        self.assertEqual(cl.CL_PROFILING_COMMAND_START, 0x1282)
        self.assertEqual(cl.CL_PROFILING_COMMAND_END, 0x1283)

    def test_error_codes(self):
        self.assertEqual(cl.CL_SUCCESS, 0)
        self.assertEqual(cl.CL_DEVICE_NOT_FOUND, -1)
        self.assertEqual(cl.CL_DEVICE_NOT_AVAILABLE, -2)
        self.assertEqual(cl.CL_COMPILER_NOT_AVAILABLE, -3)
        self.assertEqual(cl.CL_MEM_OBJECT_ALLOCATION_FAILURE, -4)
        self.assertEqual(cl.CL_OUT_OF_RESOURCES, -5)
        self.assertEqual(cl.CL_OUT_OF_HOST_MEMORY, -6)
        self.assertEqual(cl.CL_PROFILING_INFO_NOT_AVAILABLE, -7)
        self.assertEqual(cl.CL_MEM_COPY_OVERLAP, -8)
        self.assertEqual(cl.CL_IMAGE_FORMAT_MISMATCH, -9)
        self.assertEqual(cl.CL_IMAGE_FORMAT_NOT_SUPPORTED, -10)
        self.assertEqual(cl.CL_BUILD_PROGRAM_FAILURE, -11)
        self.assertEqual(cl.CL_MAP_FAILURE, -12)
        self.assertEqual(cl.CL_MISALIGNED_SUB_BUFFER_OFFSET, -13)
        self.assertEqual(cl.CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, -14)
        self.assertEqual(cl.CL_COMPILE_PROGRAM_FAILURE, -15)
        self.assertEqual(cl.CL_LINKER_NOT_AVAILABLE, -16)
        self.assertEqual(cl.CL_LINK_PROGRAM_FAILURE, -17)
        self.assertEqual(cl.CL_DEVICE_PARTITION_FAILED, -18)
        self.assertEqual(cl.CL_KERNEL_ARG_INFO_NOT_AVAILABLE, -19)

        self.assertEqual(cl.CL_INVALID_VALUE, -30)
        self.assertEqual(cl.CL_INVALID_DEVICE_TYPE, -31)
        self.assertEqual(cl.CL_INVALID_PLATFORM, -32)
        self.assertEqual(cl.CL_INVALID_DEVICE, -33)
        self.assertEqual(cl.CL_INVALID_CONTEXT, -34)
        self.assertEqual(cl.CL_INVALID_QUEUE_PROPERTIES, -35)
        self.assertEqual(cl.CL_INVALID_COMMAND_QUEUE, -36)
        self.assertEqual(cl.CL_INVALID_HOST_PTR, -37)
        self.assertEqual(cl.CL_INVALID_MEM_OBJECT, -38)
        self.assertEqual(cl.CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, -39)
        self.assertEqual(cl.CL_INVALID_IMAGE_SIZE, -40)
        self.assertEqual(cl.CL_INVALID_SAMPLER, -41)
        self.assertEqual(cl.CL_INVALID_BINARY, -42)
        self.assertEqual(cl.CL_INVALID_BUILD_OPTIONS, -43)
        self.assertEqual(cl.CL_INVALID_PROGRAM, -44)
        self.assertEqual(cl.CL_INVALID_PROGRAM_EXECUTABLE, -45)
        self.assertEqual(cl.CL_INVALID_KERNEL_NAME, -46)
        self.assertEqual(cl.CL_INVALID_KERNEL_DEFINITION, -47)
        self.assertEqual(cl.CL_INVALID_KERNEL, -48)
        self.assertEqual(cl.CL_INVALID_ARG_INDEX, -49)
        self.assertEqual(cl.CL_INVALID_ARG_VALUE, -50)
        self.assertEqual(cl.CL_INVALID_ARG_SIZE, -51)
        self.assertEqual(cl.CL_INVALID_KERNEL_ARGS, -52)
        self.assertEqual(cl.CL_INVALID_WORK_DIMENSION, -53)
        self.assertEqual(cl.CL_INVALID_WORK_GROUP_SIZE, -54)
        self.assertEqual(cl.CL_INVALID_WORK_ITEM_SIZE, -55)
        self.assertEqual(cl.CL_INVALID_GLOBAL_OFFSET, -56)
        self.assertEqual(cl.CL_INVALID_EVENT_WAIT_LIST, -57)
        self.assertEqual(cl.CL_INVALID_EVENT, -58)
        self.assertEqual(cl.CL_INVALID_OPERATION, -59)
        self.assertEqual(cl.CL_INVALID_GL_OBJECT, -60)
        self.assertEqual(cl.CL_INVALID_BUFFER_SIZE, -61)
        self.assertEqual(cl.CL_INVALID_MIP_LEVEL, -62)
        self.assertEqual(cl.CL_INVALID_GLOBAL_WORK_SIZE, -63)
        self.assertEqual(cl.CL_INVALID_PROPERTY, -64)
        self.assertEqual(cl.CL_INVALID_IMAGE_DESCRIPTOR, -65)
        self.assertEqual(cl.CL_INVALID_COMPILER_OPTIONS, -66)
        self.assertEqual(cl.CL_INVALID_LINKER_OPTIONS, -67)
        self.assertEqual(cl.CL_INVALID_DEVICE_PARTITION_COUNT, -68)
        self.assertEqual(cl.CL_INVALID_PIPE_SIZE, -69)
        self.assertEqual(cl.CL_INVALID_DEVICE_QUEUE, -70)

    def test_dump_devices(self):
        platforms = cl.Platforms()
        s = platforms.dump_devices()
        del s

    def test_create_context(self):
        platforms = cl.Platforms()
        ctx = cl.Context(platforms.platforms[0],
                         platforms.platforms[0].devices[0:1])
        del ctx

    def test_create_some_context(self):
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        del ctx

    def test_realign_numpy_array(self):
        import numpy
        a = numpy.empty(1000, dtype=numpy.float32)
        a = cl.realign_array(a, 1056, numpy)
        self.assertEqual(a.__array_interface__["data"][0] % 1056, 0)
        a = numpy.empty(1024, dtype=numpy.float32)
        a = cl.realign_array(a, 4096, numpy)
        self.assertEqual(a.__array_interface__["data"][0] % 4096, 0)

    def test_device_info(self):
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        dev = ctx.devices[0]
        self.assertGreater(dev.max_work_item_dimensions, 0)
        self.assertEqual(len(dev.max_work_item_sizes),
                         dev.max_work_item_dimensions)
        for size in dev.max_work_item_sizes:
            self.assertGreater(size, 0)
        self.assertIsInstance(dev.driver_version.encode("utf-8"), bytes)
        self.assertGreater(len(dev.driver_version), 0)
        try:
            self.assertIsInstance(dev.built_in_kernels, list)
            for krn in dev.built_in_kernels:
                self.assertIsInstance(krn, str)
                self.assertGreater(len(krn), 0)
        except cl.CLRuntimeError as e:
            if dev.version >= 1.2:
                raise
            self.assertEqual(e.code, -30)
        self.assertIsInstance(dev.extensions, list)
        for ext in dev.extensions:
            self.assertIsInstance(ext.encode("utf-8"), bytes)
            self.assertGreater(len(ext), 0)
        self.assertGreater(dev.preferred_vector_width_int, 0)
        self.assertGreater(dev.max_work_group_size, 1)
        self.assertTrue(dev.available)
        try:
            self.assertTrue(type(dev.pipe_max_active_reservations) == int)
            self.assertTrue(type(dev.pipe_max_packet_size) == int)
            self.assertTrue(type(dev.svm_capabilities) == int)
            self.assertTrue(
                type(dev.preferred_platform_atomic_alignment) == int)
            self.assertTrue(type(dev.preferred_global_atomic_alignment) == int)
            self.assertTrue(type(dev.preferred_local_atomic_alignment) == int)
        except cl.CLRuntimeError as e:
            if dev.version >= 2.0:
                raise
            self.assertEqual(e.code, -30)

    def test_program_info(self):
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        prg = ctx.create_program(self.src_test, self.include_dirs)
        self.assertGreater(prg.reference_count, 0)
        try:
            self.assertEqual(prg.num_kernels, 1)
            names = prg.kernel_names
            self.assertIsInstance(names, list)
            self.assertEqual(len(names), 1)
            self.assertEqual(names[0], "test")
        except cl.CLRuntimeError as e:
            if prg.devices[0].version >= 1.2:
                raise
            self.assertEqual(e.code, -30)
        bins = prg.binaries
        self.assertEqual(len(bins), 1)
        self.assertIsInstance(bins[0], bytes)
        self.assertGreater(len(bins[0]), 0)

    def test_kernel_info(self):
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        prg = ctx.create_program(self.src_test, self.include_dirs)
        krn = prg.get_kernel("test")
        self.assertGreater(krn.reference_count, 0)
        self.assertEqual(krn.num_args, 3)
        try:
            self.assertEqual(krn.attributes, "vec_type_hint(float4)")
        except cl.CLRuntimeError as e:
            self.assertEqual(e.code, -30)

    def test_binary(self):
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        prg = ctx.create_program(self.src_test, self.include_dirs)
        binary = prg.binaries[0]
        prg = ctx.create_program([binary], binary=True)
        krn = prg.get_kernel("test")
        del krn

    def test_set_kernel_args(self):
        import numpy

        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        prg = ctx.create_program(self.src_test, self.include_dirs)
        krn = prg.get_kernel("test")
        queue = ctx.create_queue(ctx.devices[0])
        global_size = [12345]
        local_size = None

        krn.set_args(cl.skip(3))
        self.assertRaises(cl.CLRuntimeError,
                          queue.execute_kernel, krn, global_size, local_size)
        krn.set_args(cl.skip, cl.skip, cl.skip)
        self.assertRaises(cl.CLRuntimeError,
                          queue.execute_kernel, krn, global_size, local_size)
        krn.set_args(cl.skip(1), cl.skip(1), cl.skip(1))
        self.assertRaises(cl.CLRuntimeError,
                          queue.execute_kernel, krn, global_size, local_size)
        krn.set_args(cl.skip(1000))
        self.assertRaises(cl.CLRuntimeError,
                          queue.execute_kernel, krn, global_size, local_size)
        self.assertRaises(ValueError, cl.skip, 0)
        self.assertRaises(ValueError, cl.skip, -1)

        c = numpy.array([1.2345], dtype=numpy.float32)
        krn.set_args(cl.skip(2), c)
        self.assertRaises(cl.CLRuntimeError,
                          queue.execute_kernel, krn, global_size, local_size)
        krn.set_args(cl.skip, cl.skip, c)
        self.assertRaises(cl.CLRuntimeError,
                          queue.execute_kernel, krn, global_size, local_size)

    def test_api_numpy(self):
        import numpy
        # Create platform, context, program, kernel and queue
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        prg = ctx.create_program(self.src_test, self.include_dirs)
        krn = prg.get_kernel("test")
        queue = ctx.create_queue(ctx.devices[0])

        # Create arrays with some values for testing
        a = numpy.arange(100000, dtype=numpy.float32)
        b = numpy.cos(a)
        a = numpy.sin(a)
        a_copy = a.copy()

        # Prepare arrays for use with map_buffer
        a = cl.realign_array(a, queue.device.memalign, numpy)
        b = cl.realign_array(b, queue.device.memalign, numpy)
        c = numpy.array([1.2345], dtype=numpy.float32)
        d = a + b * c[0]

        # Create buffers
        a_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR,
                               a)
        b_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR,
                               b)

        # Set kernel arguments
        krn.set_args(a_, b_, c[0:1])

        # Execute kernel
        global_size = [a.size]
        local_size = None
        queue.execute_kernel(krn, global_size, local_size, need_event=False)

        # Get results back from the device by map_buffer
        ev, ptr = queue.map_buffer(a_, cl.CL_MAP_READ, a.nbytes)
        del ev
        ev = queue.unmap_buffer(a_, ptr)
        ev.wait()
        self.assertLess(numpy.fabs(a - d).max(), 0.0001,
                        "Incorrect result after map_buffer")

        # Get results back from the device by read_buffer
        aa = numpy.zeros(a.shape, dtype=a.dtype)
        queue.read_buffer(a_, aa)
        self.assertLess(numpy.fabs(aa - d).max(), 0.0001,
                        "Incorrect result after read_buffer")

        # Refill buffer with stored copy by map_buffer with event
        ev, ptr = queue.map_buffer(
            a_, cl.CL_MAP_WRITE if queue.device.version < 1.1999
            else cl.CL_MAP_WRITE_INVALIDATE_REGION, a.nbytes,
            blocking=False, need_event=True)
        ev.wait()
        a[:] = a_copy[:]
        ev = queue.unmap_buffer(a_, ptr)

        # Execute kernel
        ev = queue.execute_kernel(krn, global_size, local_size, wait_for=(ev,))
        # Get results back from the device by map_buffer
        ev, ptr = queue.map_buffer(a_, cl.CL_MAP_READ, a.nbytes,
                                   wait_for=(ev,), need_event=True)
        ev.wait()
        ev = queue.unmap_buffer(a_, ptr)
        ev.wait()
        self.assertLess(numpy.fabs(a - d).max(), 0.0001,
                        "Incorrect result after map_buffer")

        # Refill buffer with stored copy by write_buffer
        ev = queue.write_buffer(a_, a_copy, blocking=False, need_event=True)

        # Execute kernel
        ev = queue.execute_kernel(krn, global_size, local_size, wait_for=(ev,))
        # Get results back from the device by map_buffer
        ev, ptr = queue.map_buffer(a_, cl.CL_MAP_READ, a.nbytes,
                                   wait_for=(ev,), need_event=True)
        ev.wait()
        ev = queue.unmap_buffer(a_, ptr)
        ev.wait()
        self.assertLess(numpy.fabs(a - d).max(), 0.0001,
                        "Incorrect result after map_buffer")

    def test_api_nonumpy(self):
        import math
        # Create platform, context, program, kernel and queue
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        prg = ctx.create_program(self.src_test, self.include_dirs)
        krn = prg.get_kernel("test")
        # Create command queue
        queue = ctx.create_queue(ctx.devices[0])

        # Create arrays with some values for testing
        N = 100000
        _a = cl.ffi.new("float[]", N + queue.device.memalign)
        sz = int(cl.ffi.cast("size_t", _a))
        if sz % queue.device.memalign != 0:
            sz += queue.device.memalign - (sz % queue.device.memalign)
            a = cl.ffi.cast("float*", sz)
        else:
            a = _a
        _b = cl.ffi.new("float[]", N + queue.device.memalign)
        sz = int(cl.ffi.cast("size_t", _b))
        if sz % queue.device.memalign != 0:
            sz += queue.device.memalign - (sz % queue.device.memalign)
            b = cl.ffi.cast("float*", sz)
        else:
            b = _b
        c = cl.ffi.new("float[]", 1)
        c[0] = 1.2345
        d = cl.ffi.new("float[]", N)
        sz = cl.ffi.sizeof(d)
        for i, t in enumerate(d):
            a[i] = math.sin(i)
            b[i] = math.cos(i)
            d[i] = a[i] + b[i] * c[0]
        a_copy = cl.ffi.new("float[]", N)
        a_copy[0:N] = a[0:N]

        # Create buffers
        a_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR,
                               a, size=sz)
        b_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR,
                               b, size=sz)

        # Set kernel arguments
        krn.set_arg(0, a_)
        krn.set_arg(1, b_)
        krn.set_arg(2, cl.ffi.cast("const void*", c), cl.ffi.sizeof(c))

        # Execute kernel
        global_size = [N]
        local_size = None
        queue.execute_kernel(krn, global_size, local_size, need_event=False)

        # Get results back from the device by map_buffer
        ev, ptr = queue.map_buffer(a_, cl.CL_MAP_READ, sz)
        del ev
        ev = queue.unmap_buffer(a_, ptr)
        ev.wait()
        mx = 0
        for i, t in enumerate(d):
            mx = max(mx, math.fabs(a[i] - t))
        self.assertLess(mx, 0.0001, "Incorrect result after map_buffer")

        # Get results back from the device by read_buffer
        aa = cl.ffi.new("float[]", N)
        queue.read_buffer(a_, aa, size=sz)
        mx = 0
        for i, t in enumerate(d):
            mx = max(mx, math.fabs(aa[i] - t))
        self.assertLess(mx, 0.0001, "Incorrect result after read_buffer")

        # Refill buffer with stored copy by map_buffer with event
        ev, ptr = queue.map_buffer(
            a_, cl.CL_MAP_WRITE if queue.device.version < 1.1999
            else cl.CL_MAP_WRITE_INVALIDATE_REGION, sz,
            blocking=False, need_event=True)
        ev.wait()
        a[0:N] = a_copy[0:N]
        ev = queue.unmap_buffer(a_, ptr)

        # Execute kernel
        ev = queue.execute_kernel(krn, global_size, local_size, wait_for=(ev,))
        # Get results back from the device by map_buffer
        ev, ptr = queue.map_buffer(a_, cl.CL_MAP_READ, sz,
                                   wait_for=(ev,), need_event=True)
        ev.wait()
        ev = queue.unmap_buffer(a_, ptr)
        ev.wait()
        mx = 0
        for i, t in enumerate(d):
            mx = max(mx, math.fabs(a[i] - t))
        self.assertLess(mx, 0.0001, "Incorrect result after map_buffer")

        # Refill buffer with stored copy by write_buffer
        ev = queue.write_buffer(a_, a_copy, size=sz,
                                blocking=False, need_event=True)

        # Execute kernel
        ev = queue.execute_kernel(krn, global_size, local_size, wait_for=(ev,))
        # Get results back from the device by map_buffer
        ev, ptr = queue.map_buffer(a_, cl.CL_MAP_READ, sz,
                                   wait_for=(ev,), need_event=True)
        ev.wait()
        ev = queue.unmap_buffer(a_, ptr)
        ev.wait()
        mx = 0
        for i, t in enumerate(d):
            mx = max(mx, math.fabs(a[i] - t))
        self.assertLess(mx, 0.0001, "Incorrect result after map_buffer")

        del _b
        del _a

    def test_event_profiling(self):
        import numpy
        # Create platform, context, program, kernel and queue
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        prg = ctx.create_program(self.src_test, self.include_dirs)
        krn = prg.get_kernel("test")
        queue = ctx.create_queue(ctx.devices[0], cl.CL_QUEUE_PROFILING_ENABLE)

        # Create arrays with some values for testing
        a = numpy.arange(100000, dtype=numpy.float32)
        b = numpy.cos(a)
        a = numpy.sin(a)
        c = numpy.array([1.2345], dtype=numpy.float32)

        # Create buffers
        a_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
                               a)
        b_ = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                               b)

        # Set kernel arguments
        krn.set_arg(0, a_)
        krn.set_arg(1, b_)
        krn.set_arg(2, c[0:1])

        # Execute kernel
        ev = queue.execute_kernel(krn, [a.size], None)
        ev.wait()

        try:
            vles, errs = ev.get_profiling_info()
            self.assertEqual(vles, ev.profiling_values)
            self.assertEqual(errs, ev.profiling_errors)
        except cl.CLRuntimeError:
            pass
        for name, vle in ev.profiling_values.items():
            err = ev.profiling_errors[name]
            self.assertTrue((vle and not err) or (not vle and err))
            self.assertEqual(type(vle), float)
            self.assertEqual(type(err), int)

    def test_copy_buffer(self):
        import numpy
        # Create platform, context and queue
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        queue = ctx.create_queue(ctx.devices[0])

        # Create arrays with some values for testing
        a = numpy.arange(10000, dtype=numpy.float32)
        b = a * 0.5
        c = numpy.empty_like(b)
        c[:] = 1.0e30

        # Create buffers
        a_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
                               a)
        b_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
                               b)

        # Copy some data from one buffer to another
        sz = a.itemsize
        ev = queue.copy_buffer(a_, b_, 1000 * sz, 2000 * sz, 3000 * sz)
        ev.wait()

        queue.read_buffer(b_, c)
        diff = numpy.fabs(c[2000:5000] - a[1000:4000]).max()
        self.assertEqual(diff, 0)

    def test_copy_buffer_rect(self):
        import numpy
        # Create platform, context and queue
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        queue = ctx.create_queue(ctx.devices[0])

        # Create arrays with some values for testing
        a = numpy.arange(35 * 25 * 15, dtype=numpy.float32).reshape(35, 25, 15)
        b = numpy.arange(37 * 27 * 17, dtype=numpy.float32).reshape(37, 27, 17)
        b *= 0.5
        c = numpy.empty_like(b)
        c[:] = 1.0e30

        # Create buffers
        a_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
                               a)
        b_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
                               b)

        # Copy 3D rect from one buffer to another
        sz = a.itemsize
        ev = queue.copy_buffer_rect(
            a_, b_, (3 * sz, 4, 5), (6 * sz, 7, 8), (5 * sz, 10, 20),
            a.shape[2] * sz, a.shape[1] * a.shape[2] * sz,
            b.shape[2] * sz, b.shape[1] * b.shape[2] * sz)
        ev.wait()

        queue.read_buffer(b_, c)
        diff = numpy.fabs(c[8:28, 7:17, 6:11] - a[5:25, 4:14, 3:8]).max()
        self.assertEqual(diff, 0)

    def test_fill_buffer(self):
        # Create platform, context and queue
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        if ctx.devices[0].version < 1.2:
            return
        queue = ctx.create_queue(ctx.devices[0])

        import numpy

        # Create array
        a = numpy.zeros(4096, dtype=numpy.int32)

        # Create buffer
        a_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
                               a)

        # Fill the buffer
        pattern = numpy.array([1, 2, 3, 4], dtype=numpy.int32)
        queue.fill_buffer(a_, pattern, pattern.nbytes, a.nbytes).wait()

        queue.read_buffer(a_, a)
        diff = 0
        for i in range(0, a.size, pattern.size):
            diff += numpy.fabs(a[i:i + pattern.size] - pattern).sum()
        self.assertEqual(diff, 0)

    def test_set_arg_None(self):
        import numpy
        # Create platform, context, program, kernel and queue
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        src = """
        __kernel void test(__global float *a, __global const float *b,
                           __global const float *c) {
            int idx = get_global_id(0);
            a[idx] += b[idx] + (c ? c[idx] : 0);
        }
        """
        prg = ctx.create_program(src)
        krn = prg.get_kernel("test")
        queue = ctx.create_queue(ctx.devices[0])

        # Create arrays with some values for testing
        a = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
        b = numpy.array([6, 7, 8, 9, 10], dtype=numpy.float32)
        c = numpy.array([11, 12, 13, 14, 15], dtype=numpy.float32)

        # Create buffers
        a_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
                               a)
        b_ = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                               b)
        c_ = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                               c)

        # Set kernel arguments
        krn.set_arg(0, a_)
        krn.set_arg(1, b_)
        krn.set_arg(2, c_)

        # Execute kernel
        ev = queue.execute_kernel(krn, [a.size], None)
        ev.wait()

        # Get results back
        d = numpy.zeros_like(a)
        queue.read_buffer(a_, d)
        t = a + b + c
        diff = numpy.fabs(d - t).max()
        self.assertEqual(diff, 0)

        # Set arg to None
        krn.set_arg(2, None)

        # Execute kernel
        ev = queue.execute_kernel(krn, [a.size], None)
        ev.wait()

        # Get results back
        queue.read_buffer(a_, d)
        t += b
        diff = numpy.fabs(d - t).max()
        self.assertEqual(diff, 0)

    def test_create_sub_buffer(self):
        import numpy
        # Create platform, context, program, kernel and queue
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        prg = ctx.create_program(self.src_test, self.include_dirs)
        krn = prg.get_kernel("test")
        queue = ctx.create_queue(ctx.devices[0])

        # Create arrays with some values for testing
        a = numpy.arange(100000, dtype=numpy.float32)
        b = numpy.cos(a)
        a = numpy.sin(a)

        # Prepare arrays for use with map_buffer
        a = cl.realign_array(a, queue.device.memalign, numpy)
        b = cl.realign_array(b, queue.device.memalign, numpy)
        c = numpy.array([1.2345], dtype=numpy.float32)
        d = a[1024:1024 + 4096] + b[2048:2048 + 4096] * c[0]

        # Create buffers
        a_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR,
                               a).create_sub_buffer(4096, 16384)
        b_ = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_USE_HOST_PTR,
                               b).create_sub_buffer(8192, 16384)

        # Set kernel arguments
        krn.set_args(a_, b_, c[0:1])

        # Execute kernel
        global_size = [4096]
        local_size = None
        queue.execute_kernel(krn, global_size, local_size, need_event=False)

        # Get results back from the device by map_buffer
        ev, ptr = queue.map_buffer(a_, cl.CL_MAP_READ, a_.size)
        del ev
        ev = queue.unmap_buffer(a_, ptr)
        ev.wait()
        self.assertLess(numpy.fabs(a[1024:1024 + 4096] - d).max(), 0.0001,
                        "Incorrect result after map_buffer")

        # Get results back from the device by read_buffer
        aa = numpy.zeros(4096, dtype=numpy.float32)
        queue.read_buffer(a_, aa)
        self.assertLess(numpy.fabs(aa - d).max(), 0.0001,
                        "Incorrect result after read_buffer")

    def test_create_queue_with_properties(self):
        ctx = cl.Platforms().create_some_context()
        try:
            queue = ctx.create_queue(
                ctx.devices[0],
                cl.CL_QUEUE_ON_DEVICE |
                cl.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                properties={cl.CL_QUEUE_SIZE: 64})
            del queue
        except cl.CLRuntimeError:
            if ctx.devices[0].version >= 2.0:
                raise
            return
        queue = ctx.create_queue(
            ctx.devices[0],
            properties={cl.CL_QUEUE_SIZE: 64,
                        cl.CL_QUEUE_PROPERTIES:
                        cl.CL_QUEUE_ON_DEVICE |
                        cl.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE})
        del queue

    def test_work_group_info(self):
        ctx = cl.Platforms().create_some_context()
        prg = ctx.create_program(self.src_test, self.include_dirs)
        krn = prg.get_kernel("test")
        info = krn.get_work_group_info(ctx.devices[0])

        self.assertRaises(cl.CLRuntimeError, getattr, info, "global_work_size")

        for vle in (info.compile_work_group_size,):
            self.assertIsInstance(vle, tuple)
            self.assertEqual(len(vle), 3)
            for x in vle:
                self.assertIsInstance(x, int)
                self.assertGreaterEqual(x, 0)

        for vle in (info.work_group_size, info.local_mem_size,
                    info.preferred_work_group_size_multiple,
                    info.private_mem_size):
            self.assertIsInstance(vle, int)
            self.assertGreaterEqual(vle, 0)

    def test_create_pipe(self):
        ctx = cl.Platforms().create_some_context()
        if ctx.devices[0].version < 2.0:
            return
        pipe = ctx.create_pipe(0, 8, 16)
        del pipe
        pipe = ctx.create_pipe(cl.CL_MEM_READ_WRITE, 8, 16)
        prg = ctx.create_program("""
            __kernel void test(__write_only pipe int p) {
                int x = 0;
                write_pipe(p, &x);
            }
            """, options="-cl-std=CL2.0")
        krn = prg.get_kernel("test")
        krn.set_arg(0, pipe)
        del krn
        del prg
        del pipe

    def test_svm_alloc(self):
        ctx = cl.Platforms().create_some_context()
        if ctx.devices[0].version < 2.0:
            return
        svm = ctx.svm_alloc(cl.CL_MEM_READ_WRITE, 4096)
        svm.release()
        self.assertIsNone(svm.handle)
        del svm
        svm = ctx.svm_alloc(cl.CL_MEM_READ_WRITE, 4096)
        prg = ctx.create_program("""
            __kernel void test(__global void *p) {
                __global int *ptr = (__global int*)p;
                *ptr += 1;
            }
            """, options="-cl-std=CL2.0")
        krn = prg.get_kernel("test")
        krn.set_arg(0, svm)
        krn.set_arg_svm(0, svm)
        queue = ctx.create_queue(ctx.devices[0])
        queue.svm_map(svm, cl.CL_MAP_WRITE_INVALIDATE_REGION, 4)
        p = cl.ffi.cast("int*", svm.handle)
        p[0] = 2
        queue.svm_unmap(svm)
        queue.execute_kernel(krn, [1], None)
        queue.svm_map(svm, cl.CL_MAP_READ, 4)
        self.assertEqual(p[0], 3)
        # always ensure that the last unmap had completed before
        # the svm destructor
        queue.svm_unmap(svm).wait()
        try:
            import numpy
            a = numpy.frombuffer(svm.buffer, dtype=numpy.int32)
            queue.execute_kernel(krn, [1], None)
            queue.svm_map(svm, cl.CL_MAP_READ, 4)
            self.assertEqual(a[0], 4)
            queue.svm_unmap(svm).wait()
        except ImportError:
            pass
        del svm  # svm destructor here

    def test_svm_memcpy(self):
        ctx = cl.Platforms().create_some_context()
        if ctx.devices[0].version < 2.0:
            return
        svm = ctx.svm_alloc(cl.CL_MEM_READ_WRITE, 4096)
        import numpy
        a = numpy.frombuffer(svm.buffer, dtype=numpy.int32)
        queue = ctx.create_queue(ctx.devices[0])
        queue.svm_map(svm, cl.CL_MAP_WRITE_INVALIDATE_REGION, svm.size)
        a[:] = numpy.arange(a.size, dtype=a.dtype)
        queue.svm_unmap(svm)
        n = a.size // 2
        queue.svm_memcpy(a[n:], a, n * a.itemsize)
        queue.svm_map(svm, cl.CL_MAP_READ, svm.size)
        self.assertEqual(numpy.fabs(a[n:] - a[:n]).max(), 0)
        queue.svm_unmap(svm).wait()
        del svm

    def test_svm_memfill(self):
        ctx = cl.Platforms().create_some_context()
        if ctx.devices[0].version < 2.0:
            return
        svm = ctx.svm_alloc(cl.CL_MEM_READ_WRITE, 4096)
        import numpy
        a = numpy.frombuffer(svm.buffer, dtype=numpy.int32)
        queue = ctx.create_queue(ctx.devices[0])
        pattern = numpy.array([1, 2, 3, 4], dtype=numpy.int32)
        queue.svm_memfill(a, pattern, pattern.nbytes, a.nbytes)
        queue.svm_map(svm, cl.CL_MAP_READ, svm.size)
        diff = 0
        for i in range(0, a.size, pattern.size):
            diff += numpy.fabs(a[i:i + pattern.size] - pattern).sum()
        self.assertEqual(diff, 0)
        queue.svm_unmap(svm).wait()
        del svm


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
