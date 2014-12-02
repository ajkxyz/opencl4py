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
Tests some of the api in opencl4py.blas._clBlas module.
"""


import unittest
import logging
import numpy
import opencl4py as cl
import opencl4py.blas as blas
import os


class Test(unittest.TestCase):
    def setUp(self):
        self.old_env = os.environ.get("PYOPENCL_CTX")
        if self.old_env is None:
            os.environ["PYOPENCL_CTX"] = "0:0"
        self.blas = blas.CLBLAS()

    def tearDown(self):
        if self.old_env is None:
            del os.environ["PYOPENCL_CTX"]
        else:
            os.environ["PYOPENCL_CTX"] = self.old_env
        del self.old_env

    def _test_gemm(self, gemm, dtype):
        ctx = cl.Platforms().create_some_context()
        queue = ctx.create_queue(ctx.devices[0])
        a = numpy.zeros([127, 353], dtype=dtype)
        b = numpy.zeros([135, a.shape[1]], dtype=dtype)
        c = numpy.zeros([a.shape[0], b.shape[0]], dtype=dtype)
        numpy.random.seed(123)
        a[:] = numpy.random.rand(a.size).astype(dtype).reshape(a.shape)
        b[:] = numpy.random.rand(b.size).astype(dtype).reshape(b.shape)
        gold_c = numpy.dot(a, b.transpose())
        a_buf = ctx.create_buffer(
            cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR, a)
        b_buf = ctx.create_buffer(
            cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR, b)
        c_buf = ctx.create_buffer(
            cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR, c)
        gemm([queue], blas.clblasRowMajor, blas.clblasNoTrans,
             blas.clblasTrans, a.shape[0], b.shape[0], a.shape[1],
             1.0, a_buf, b_buf, 0.0, c_buf)
        queue.flush()
        queue.read_buffer(c_buf, c)
        max_diff = numpy.fabs(c - gold_c).max()
        self.assertLess(max_diff, 0.0001)

    def test_sgemm(self):
        logging.debug("ENTER: test_sgemm")
        self._test_gemm(self.blas.sgemm, numpy.float32)
        logging.debug("EXIT: test_sgemm")

    def test_dgemm(self):
        logging.debug("ENTER: test_dgemm")
        self._test_gemm(self.blas.dgemm, numpy.float64)
        logging.debug("EXIT: test_dgemm")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
