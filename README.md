opencl4py
=========

Python cffi OpenCL bindings and helper classes.

Tested with Python 2.7, Python 3.3, Python 3.4 and PyPy on Linux and Windows.

Covered OpenCL API:
```
clBuildProgram
clCreateBuffer
clCreateCommandQueue
clCreateContext
clCreateKernel
clCreateKernel
clCreateProgramWithSource
clCreateProgramWithBinary
clEnqueueCopyBuffer
clEnqueueCopyBufferRect
clEnqueueMapBuffer
clEnqueueNDRangeKernel
clEnqueueReadBuffer
clEnqueueUnmapMemObject
clEnqueueWriteBuffer
clFinish
clFlush
clGetDeviceIDs
clGetDeviceInfo
clGetEventProfilingInfo
clGetPlatformIDs
clGetPlatformInfo
clGetProgramInfo
clGetProgramBuildInfo
clGetKernelInfo
clReleaseCommandQueue
clReleaseContext
clReleaseEvent
clReleaseKernel
clReleaseKernel
clReleaseMemObject
clReleaseProgram
clSetKernelArg
clWaitForEvents
```

To install the module run:
```bash
python setup.py install
```
or just copy src/opencl4py to any place where python
interpreter will be able to find it.

To run the tests, execute:

for Python 2.7:
```bash
PYTHONPATH=src nosetests -w tests
```

for Python 3.3, 3.4:
```bash
PYTHONPATH=src nosetests3 -w tests
```

for PyPy:
```bash
PYTHONPATH=src pypy tests/test_api.py
```

Currently, PyPy numpy support may be incomplete,
so tests which use numpy arrays may fail.

Example usage:

```python
import opencl4py as cl
import logging
import numpy


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    platforms = cl.Platforms()
    logging.info("OpenCL devices available:\n\n%s\n",
                 platforms.dump_devices())
    ctx = platforms.create_some_context()
    queue = ctx.create_queue(ctx.devices[0])
    prg = ctx.create_program(
        """
        __kernel void test(__global const float *a, __global const float *b,
                           __global float *c, const float k) {
          size_t i = get_global_id(0);
          c[i] = (a[i] + b[i]) * k;
        }
        """)
    krn = prg.get_kernel("test")
    a = numpy.arange(1000000, dtype=numpy.float32)
    b = numpy.arange(1000000, dtype=numpy.float32)
    c = numpy.empty(1000000, dtype=numpy.float32)
    k = numpy.array([0.5], dtype=numpy.float32)
    a_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                              a)
    b_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                              b)
    c_buf = ctx.create_buffer(cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_ALLOC_HOST_PTR,
                              size=c.nbytes)
    krn.set_arg(0, a_buf)
    krn.set_arg(1, b_buf)
    krn.set_arg(2, c_buf)
    krn.set_arg(3, k[0:1])
    queue.execute_kernel(krn, [a.size], None)
    queue.read_buffer(c_buf, c)
    max_diff = numpy.fabs(c - (a + b) * k[0]).max()
    logging.info("max_diff = %.6f", max_diff)
```

Released under Simplified BSD License.
Copyright (c) 2014, Samsung Electronics Co.,Ltd.
