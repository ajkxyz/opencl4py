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
        logging.basicConfig(level=logging.DEBUG)
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
        self.assertEqual(cl.CL_MAP_READ, 1)
        self.assertEqual(cl.CL_MAP_WRITE, 2)
        self.assertEqual(cl.CL_MAP_WRITE_INVALIDATE_REGION, 4)
        self.assertEqual(cl.CL_MEM_READ_WRITE, 1)
        self.assertEqual(cl.CL_MEM_WRITE_ONLY, 2)
        self.assertEqual(cl.CL_MEM_READ_ONLY, 4)
        self.assertEqual(cl.CL_MEM_USE_HOST_PTR, 8)
        self.assertEqual(cl.CL_MEM_ALLOC_HOST_PTR, 16)
        self.assertEqual(cl.CL_MEM_COPY_HOST_PTR, 32)

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

    def set_kernel_args(self):
        platforms = cl.Platforms()
        ctx = platforms.create_some_context()
        prg = ctx.create_program(self.src_test, self.include_dirs)
        krn = prg.get_kernel("test")
        queue = ctx.create_queue(ctx.devices[0])
        global_size = [a.size]
        local_size = None

        krn.set_args(cl.skip(3))
        self.assertRaises(CLRuntimeError,
                          queue.execute_kernel(krn, global_size, local_size))
        krn.set_args(cl.skip. cl.skip, cl.skip)
        self.assertRaises(CLRuntimeError,
                          queue.execute_kernel(krn, global_size, local_size))
        krn.set_args(cl.skip(1). cl.skip(1), cl.skip(1))
        self.assertRaises(CLRuntimeError,
                          queue.execute_kernel(krn, global_size, local_size))
        krn.set_args(cl.skip(1000))
        self.assertRaises(CLRuntimeError,
                          queue.execute_kernel(krn, global_size, local_size))
        self.assertRaises(ValueError, cl.skip, 0)
        self.assertRaises(ValueError, cl.skip, -1)

        c = numpy.array([1.2345], dtype=numpy.float32)
        krn.set_args(cl.skip(2), c)
        self.assertRaises(CLRuntimeError,
                          queue.execute_kernel(krn, global_size, local_size))
        krn.set_args(cl.skip, cl.skip, c)
        self.assertRaises(CLRuntimeError,
                          queue.execute_kernel(krn, global_size, local_size))

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


if __name__ == "__main__":
    unittest.main()
