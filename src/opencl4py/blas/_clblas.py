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
clBLAS cffi bindings and helper classes.
"""
import opencl4py._cffi as clffi
from opencl4py._py import CL, CLRuntimeError, Event
import cffi


#: ffi parser
ffi = cffi.FFI()


#: Loaded shared library
lib = None


#: Error codes
clblasSuccess = clffi.CL_SUCCESS
clblasInvalidValue = clffi.CL_INVALID_VALUE
clblasInvalidCommandQueue = clffi.CL_INVALID_COMMAND_QUEUE
clblasInvalidContext = clffi.CL_INVALID_CONTEXT
clblasInvalidMemObject = clffi.CL_INVALID_MEM_OBJECT
clblasInvalidDevice = clffi.CL_INVALID_DEVICE
clblasInvalidEventWaitList = clffi.CL_INVALID_EVENT_WAIT_LIST
clblasOutOfResources = clffi.CL_OUT_OF_RESOURCES
clblasOutOfHostMemory = clffi.CL_OUT_OF_HOST_MEMORY
clblasInvalidOperation = clffi.CL_INVALID_OPERATION
clblasCompilerNotAvailable = clffi.CL_COMPILER_NOT_AVAILABLE
clblasBuildProgramFailure = clffi.CL_BUILD_PROGRAM_FAILURE
clblasNotImplemented = -1024
clblasNotInitialized = -1023
clblasInvalidMatA = -1022
clblasInvalidMatB = -1021
clblasInvalidMatC = -1020
clblasInvalidVecX = -1019
clblasInvalidVecY = -1018
clblasInvalidDim = -1017
clblasInvalidLeadDimA = -1016
clblasInvalidLeadDimB = -1015
clblasInvalidLeadDimC = -1014
clblasInvalidIncX = -1013
clblasInvalidIncY = -1012
clblasInsufficientMemMatA = -1011
clblasInsufficientMemMatB = -1010
clblasInsufficientMemMatC = -1009
clblasInsufficientMemVecX = -1008
clblasInsufficientMemVecY = -1007


#: Error descriptions
ERRORS = {
    clblasNotImplemented: "Functionality is not implemented",
    clblasNotInitialized: "clblas library is not initialized yet",
    clblasInvalidMatA: "Matrix A is not a valid memory object",
    clblasInvalidMatB: "Matrix B is not a valid memory object",
    clblasInvalidMatC: "Matrix C is not a valid memory object",
    clblasInvalidVecX: "Vector X is not a valid memory object",
    clblasInvalidVecY: "Vector Y is not a valid memory object",
    clblasInvalidDim: "An input dimension (M,N,K) is invalid",
    clblasInvalidLeadDimA:
    "Leading dimension A must not be less "
    "than the size of the first dimension",
    clblasInvalidLeadDimB:
    "Leading dimension B must not be less "
    "than the size of the second dimension",
    clblasInvalidLeadDimC:
    "Leading dimension C must not be less "
    "than the size of the third dimension",
    clblasInvalidIncX: "The increment for a vector X must not be 0",
    clblasInvalidIncY: "The increment for a vector Y must not be 0",
    clblasInsufficientMemMatA: "The memory object for Matrix A is too small",
    clblasInsufficientMemMatB: "The memory object for Matrix B is too small",
    clblasInsufficientMemMatC: "The memory object for Matrix C is too small",
    clblasInsufficientMemVecX: "The memory object for Vector X is too small",
    clblasInsufficientMemVecY: "The memory object for Vector Y is too small"
}


#: clblasOrder
clblasRowMajor = 0
clblasColumnMajor = 1


#: clblasTranspose
clblasNoTrans = 0
clblasTrans = 1
clblasConjTrans = 2


def _initialize(backends):
    global lib
    if lib is not None:
        return
    # C function definitions
    src = """
    typedef int32_t cl_int;
    typedef uint32_t cl_uint;
    typedef float cl_float;
    typedef double cl_double;

    typedef void* cl_mem;
    typedef void* cl_command_queue;
    typedef void* cl_event;

    typedef int clblasStatus;
    typedef int clblasOrder;
    typedef int clblasTranspose;

    clblasStatus clblasSetup();
    void clblasTeardown();

    clblasStatus clblasSgemm(clblasOrder order,
                             clblasTranspose transA,
                             clblasTranspose transB,
                             size_t M,
                             size_t N,
                             size_t K,
                             cl_float alpha,
                             const cl_mem A,
                             size_t offA,
                             size_t lda,
                             const cl_mem B,
                             size_t offB,
                             size_t ldb,
                             cl_float beta,
                             cl_mem C,
                             size_t offC,
                             size_t ldc,
                             cl_uint numCommandQueues,
                             cl_command_queue *commandQueues,
                             cl_uint numEventsInWaitList,
                             const cl_event *eventWaitList,
                             cl_event *events);
    clblasStatus clblasDgemm(clblasOrder order,
                             clblasTranspose transA,
                             clblasTranspose transB,
                             size_t M,
                             size_t N,
                             size_t K,
                             cl_double alpha,
                             const cl_mem A,
                             size_t offA,
                             size_t lda,
                             const cl_mem B,
                             size_t offB,
                             size_t ldb,
                             cl_double beta,
                             cl_mem C,
                             size_t offC,
                             size_t ldc,
                             cl_uint numCommandQueues,
                             cl_command_queue *commandQueues,
                             cl_uint numEventsInWaitList,
                             const cl_event *eventWaitList,
                             cl_event *events);
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
        raise OSError("Could not load clBlas library")

    global ERRORS
    CL.ERRORS.update(ERRORS)


def initialize(backends=("libclBLAS.so", "clBLAS.dll")):
    clffi.initialize()
    global lib
    if lib is not None:
        return
    with clffi.lock:
        _initialize(backends)


class CLBLAS(object):
    """CLBLAS functions can be invoked from this class.
    """
    def __init__(self):
        self._lib = None
        initialize()
        err = lib.clblasSetup()
        if err:
            raise CLRuntimeError("clblasSetup() failed with error %s" %
                                 CL.get_error_description(err), err)
        self._lib = lib  # to hold the reference

    def sgemm(self, queues, order, transA, transB,
              rowsCountA, columnCountB, commonSideLength,
              alpha, A, B, beta, C,
              offsetA=0, strideA=0,
              offsetB=0, strideB=0,
              offsetC=0, strideC=0,
              wait_for=None, need_event=False):
        """Single precision (float) GEneral Matrix Multiplication.

        C = alpha * dot(A, B) + beta * C
        C = alpha * dot(A^T, B) + beta * C
        C = alpha * dot(A, B^T) + beta * C
        C = alpha * dot(A^T, B^T) + beta * C

        Parameters:
            queues: list of the Queue objects on which this operation
                    will be enqueued.
            order: row/column order (clblasRowMajor, clblasColumnMajor).
            transA: how matrix A is to be transposed
                    (clblasNoTrans, clblasTrans, clblasConjTrans).
            transB: how matrix B is to be transposed
                    (clblasNoTrans, clblasTrans, clblasConjTrans).
            rowsCountA: number of rows in matrix A.
            columnCountB: number of columns in matrix B.
            commonSideLength: length of the common side of the matrices.
            alpha: the factor of matrix A.
            A: Buffer object storing matrix A.
            B: Buffer object storing matrix B.
            beta: the factor of matrix C.
            C: Buffer object storing matrix C.
            offsetA: offset of the first element of the matrix A
                     in the buffer object, counted in elements.
            strideA: leading dimension of matrix A:
                     ((clblasNoTrans, clblasRowMajor) or
                      (clblasTrans, clblasColumnMajor)): >= commonSideLength,
                     else: >= rowsCountA.
            offsetB: offset of the first element of the matrix B
                     in the buffer object, counted in elements.
            strideB: leading dimension of matrix B:
                     ((clblasNoTrans, clblasRowMajor) or
                      (clblasTrans, clblasColumnMajor)): >= columnCountB,
                     else: >= commonSideLength.
            offsetC: offset of the first element of the matrix C
                     in the buffer object, counted in elements.
            strideC: leading dimension of matrix C:
                     clblasRowMajor: >= columnCountB,
                     else: >= rowsCountA.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = ffi.new("cl_event[]", 1) if need_event else clffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        _queues = ffi.new("cl_command_queue[]", len(queues))
        for i, q in enumerate(queues):
            _queues[i] = q.handle
        if not strideA:
            strideA = (
                commonSideLength
                if ((transA == clblasNoTrans and order == clblasRowMajor) or
                    (transA != clblasNoTrans and order == clblasColumnMajor))
                else rowsCountA)
        if not strideB:
            strideB = (
                columnCountB
                if ((transB == clblasNoTrans and order == clblasRowMajor) or
                    (transB != clblasNoTrans and order == clblasColumnMajor))
                else commonSideLength)
        if not strideC:
            strideC = columnCountB if order == clblasRowMajor else rowsCountA
        err = self._lib.clblasSgemm(
            order, transA, transB, rowsCountA, columnCountB, commonSideLength,
            alpha, A.handle, offsetA, strideA, B.handle, offsetB, strideB,
            beta, C.handle, offsetC, strideC, len(queues), _queues,
            n_events, wait_list, event)
        if err:
            raise CLRuntimeError("clblasSgemm() failed with error %s" %
                                 CL.get_error_description(err), err)
        return Event(event[0]) if event != clffi.NULL else None

    def dgemm(self, queues, order, transA, transB,
              rowsCountA, columnCountB, commonSideLength,
              alpha, A, B, beta, C,
              offsetA=0, strideA=0,
              offsetB=0, strideB=0,
              offsetC=0, strideC=0,
              wait_for=None, need_event=False):
        """Double precision (double) GEneral Matrix Multiplication.

        C = alpha * dot(A, B) + beta * C
        C = alpha * dot(A^T, B) + beta * C
        C = alpha * dot(A, B^T) + beta * C
        C = alpha * dot(A^T, B^T) + beta * C

        Parameters:
            queues: list of the Queue objects on which this operation
                    will be enqueued.
            order: row/column order (clblasRowMajor, clblasColumnMajor).
            transA: how matrix A is to be transposed
                    (clblasNoTrans, clblasTrans, clblasConjTrans).
            transB: how matrix B is to be transposed
                    (clblasNoTrans, clblasTrans, clblasConjTrans).
            rowsCountA: number of rows in matrix A.
            columnCountB: number of columns in matrix B.
            commonSideLength: length of the common side of the matrices.
            alpha: the factor of matrix A.
            A: Buffer object storing matrix A.
            B: Buffer object storing matrix B.
            beta: the factor of matrix C.
            C: Buffer object storing matrix C.
            offsetA: offset of the first element of the matrix A
                     in the buffer object, counted in elements.
            strideA: leading dimension of matrix A:
                     ((clblasNoTrans, clblasRowMajor) or
                      (clblasTrans, clblasColumnMajor)): >= commonSideLength,
                     else: >= rowsCountA.
            offsetB: offset of the first element of the matrix B
                     in the buffer object, counted in elements.
            strideB: leading dimension of matrix B:
                     ((clblasNoTrans, clblasRowMajor) or
                      (clblasTrans, clblasColumnMajor)): >= columnCountB,
                     else: >= commonSideLength.
            offsetC: offset of the first element of the matrix C
                     in the buffer object, counted in elements.
            strideC: leading dimension of matrix C:
                     clblasRowMajor: >= columnCountB,
                     else: >= rowsCountA.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = ffi.new("cl_event[]", 1) if need_event else clffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        _queues = ffi.new("cl_command_queue[]", len(queues))
        for i, q in enumerate(queues):
            _queues[i] = q.handle
        if not strideA:
            strideA = (
                commonSideLength
                if ((transA == clblasNoTrans and order == clblasRowMajor) or
                    (transA != clblasNoTrans and order == clblasColumnMajor))
                else rowsCountA)
        if not strideB:
            strideB = (
                columnCountB
                if ((transB == clblasNoTrans and order == clblasRowMajor) or
                    (transB != clblasNoTrans and order == clblasColumnMajor))
                else commonSideLength)
        if not strideC:
            strideC = columnCountB if order == clblasRowMajor else rowsCountA
        err = self._lib.clblasDgemm(
            order, transA, transB, rowsCountA, columnCountB, commonSideLength,
            alpha, A.handle, offsetA, strideA, B.handle, offsetB, strideB,
            beta, C.handle, offsetC, strideC, len(queues), _queues,
            n_events, wait_list, event)
        if err:
            raise CLRuntimeError("clblasDgemm() failed with error %s" %
                                 CL.get_error_description(err), err)
        return Event(event[0]) if event != clffi.NULL else None

    def __del__(self):
        if self._lib is not None:
            self._lib.clblasTeardown()
