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
Helper classes for OpenCL cffi bindings.
"""
import opencl4py._cffi as cl


class CLRuntimeError(RuntimeError):
    def __init__(self, msg, code):
        super(CLRuntimeError, self).__init__(msg)
        self.code = code


class CL(object):
    """Base OpenCL class.
    """
    def __init__(self):
        self.lib_ = cl.lib  # to hold the reference

    @staticmethod
    def extract_ptr_and_size(host_array, size):
        """Returns cffi pointer to host_array and its size.
        """
        if hasattr(host_array, "__array_interface__"):
            host_ptr = host_array.__array_interface__["data"][0]
            if size is None:
                size = host_array.nbytes
        else:
            host_ptr = host_array
            if size is None:
                raise ValueError("size should be set "
                                 "in case of non-numpy host_array")
        return (cl.NULL if host_ptr is None
                else cl.ffi.cast("void*", host_ptr), size)

    @staticmethod
    def get_wait_list(wait_for):
        """Returns cffi event list and number of events
        from list of Event objects, returns (None, 0) if wait_for is None.
        """
        if wait_for is not None:
            n_events = len(wait_for)
            wait_list = cl.ffi.new("cl_event[]", n_events)
            for i, ev in enumerate(wait_for):
                wait_list[i] = ev.event_
        else:
            n_events = 0
            wait_list = cl.NULL
        return (wait_list, n_events)


class Event(CL):
    """Holds OpenCL event.

    Attributes:
        event_: cffi OpenCL event handle.
        profiling_values:
            dictionary of profiling values
            if get_profiling_info was ever called;
            keys: CL_PROFILING_COMMAND_QUEUED,
                  CL_PROFILING_COMMAND_SUBMIT,
                  CL_PROFILING_COMMAND_START,
                  CL_PROFILING_COMMAND_END;
            values: the current device time counter in seconds (float),
                    or 0 if there was an error, in such case, corresponding
                    profile_errors will be set with the error code.
        profiling_errors: dictionary of profiling errors
                          if get_profiling_info was ever called.
    """
    def __init__(self, event_):
        super(Event, self).__init__()
        self.event_ = event_

    @staticmethod
    def wait_multi(wait_for, lib=cl.lib):
        """Wait on list of Event objects.
        """
        wait_list, n_events = CL.get_wait_list(wait_for)
        n = lib.clWaitForEvents(n_events, wait_list)
        if n:
            raise CLRuntimeError("clWaitForEvents() failed with "
                                 "error %d" % (n), n)

    def wait(self):
        """Waits on this event.
        """
        Event.wait_multi((self,), self.lib_)

    def get_profiling_info(self, raise_exception=True):
        """Get profiling info of the event.

        Queue should be created with CL_QUEUE_PROFILING_ENABLE flag,
        and event should be in complete state (wait completed).

        Parameters:
            raise_exception: raise exception on error or not,
                             self.profiling_values, self.profiling_errors
                             will be available anyway.

        Returns:
            tuple of (profiling_values, profiling_errors).
        """
        vle = cl.ffi.new("cl_ulong[]", 1)
        sz = cl.ffi.sizeof(vle)
        vles = {}
        errs = {}
        for name in (cl.CL_PROFILING_COMMAND_QUEUED,
                     cl.CL_PROFILING_COMMAND_SUBMIT,
                     cl.CL_PROFILING_COMMAND_START,
                     cl.CL_PROFILING_COMMAND_END):
            vle[0] = 0
            n = self.lib_.clGetEventProfilingInfo(
                self.event_, name, sz, vle, cl.NULL)
            vles[name] = 1.0e-9 * vle[0] if not n else 0.0
            errs[name] = n
        self.profiling_values = vles
        self.profiling_errors = errs
        if raise_exception:
            for err in errs.values():
                if not err:
                    continue
                raise CLRuntimeError("clGetEventProfilingInfo() failed with "
                                     "error %d" % (err), err)
        return (vles, errs)

    def release(self):
        if self.event_ is not None:
            self.lib_.clReleaseEvent(self.event_)
            self.event_ = None

    def __del__(self):
        self.release()


class Queue(CL):
    """Holds OpenCL command queue.

    Attributes:
        queue_: cffi OpenCL queue handle.
        context: context assosiated with this queue.
        device: device assosiated with this queue.
    """
    def __init__(self, context, device, flags):
        super(Queue, self).__init__()
        self.queue_ = None
        self.context = context
        self.device = device
        err = cl.ffi.new("cl_int[]", 1)
        self.queue_ = self.lib_.clCreateCommandQueue(
            context.context_, device.device_, flags, err)
        if err[0]:
            self.queue_ = None
            raise CLRuntimeError("clCreateCommandQueue() failed with "
                                 "error %d" % (err[0]), err[0])

    def execute_kernel(self, kernel, global_size, local_size,
                       global_offset=None, wait_for=None, need_event=True):
        """Executes OpenCL kernel (calls clEnqueueNDRangeKernel).

        Parameters:
            kernel: Kernel object.
            global_size: global size.
            local_size: local size.
            global_offset: global offset.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event[]", 1) if need_event else cl.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        n_dims = len(global_size)
        global_work_size = cl.ffi.new("size_t[]", n_dims)
        for i, sz in enumerate(global_size):
            global_work_size[i] = sz
        if local_size is None:
            local_work_size = cl.NULL
        else:
            if len(local_size) != n_dims:
                raise ValueError("local_size should be the same length "
                                 "as global_size")
            local_work_size = cl.ffi.new("size_t[]", n_dims)
            for i, sz in enumerate(local_size):
                local_work_size[i] = sz
        if global_offset is None:
            global_work_offset = cl.NULL
        else:
            if len(global_work_offset) != n_dims:
                raise ValueError("global_offset should be the same length "
                                 "as global_size")
            global_work_offset = cl.ffi.new("size_t[]", n_dims)
            for i, sz in enumerate(global_offset):
                global_work_offset[i] = sz
        n = self.lib_.clEnqueueNDRangeKernel(
            self.queue_, kernel.kernel_, n_dims, global_work_offset,
            global_work_size, local_work_size, n_events, wait_list, event)
        if n:
            raise CLRuntimeError("clEnqueueNDRangeKernel() failed with "
                                 "error %d" % (n), n)
        return Event(event[0]) if event != cl.NULL else None

    def map_buffer(self, buf, flags, size, blocking=True, offset=0,
                   wait_for=None, need_event=False):
        """Maps buffer.

        Parameters:
            buf: Buffer object.
            flags: mapping flags.
            size: mapping size.
            offset: mapping offset.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            (event, ptr): event - Event object or None if need_event == False,
                          ptr - pointer to the mapped buffer
                                (cffi void* converted to int).
        """
        err = cl.ffi.new("cl_int[]", 1)
        event = cl.ffi.new("cl_event[]", 1) if need_event else cl.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        ptr = self.lib_.clEnqueueMapBuffer(
            self.queue_, buf.buffer_, blocking, flags, offset, size,
            n_events, wait_list, event, err)
        if err[0]:
            raise CLRuntimeError("clEnqueueMapBuffer() failed with "
                                 "error %d" % (err[0]), err[0])
        return (None if event == cl.NULL else Event(event[0]),
                int(cl.ffi.cast("size_t", ptr)))

    def unmap_buffer(self, buf, ptr, wait_for=None, need_event=True):
        """Unmaps previously mapped buffer.

        Parameters:
            buf: Buffer object to unmap.
            ptr: pointer to the mapped buffer.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event[]", 1) if need_event else cl.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        n = self.lib_.clEnqueueUnmapMemObject(
            self.queue_, buf.buffer_, cl.ffi.cast("void*", ptr),
            n_events, wait_list, event)
        if n:
            raise CLRuntimeError("clEnqueueUnmapMemObject() failed with "
                                 "error %d" % (n), n)
        return Event(event[0]) if event != cl.NULL else None

    def read_buffer(self, buf, host_array, blocking=True, size=None, offset=0,
                    wait_for=None, need_event=False):
        """Copies from device buffer to host buffer.

        Parameters:
            buf: Buffer object.
            host_array: numpy array.
            blocking: if the read is blocking.
            size: size in bytes to copy (None for entire numpy array).
            offset: offset in the device buffer.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event[]", 1) if need_event else cl.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        host_ptr, size = CL.extract_ptr_and_size(host_array, size)
        n = self.lib_.clEnqueueReadBuffer(
            self.queue_, buf.buffer_, blocking, offset, size, host_ptr,
            n_events, wait_list, event)
        if n:
            raise CLRuntimeError("clEnqueueReadBuffer() failed with "
                                 "error %d" % (n), n)
        return Event(event[0]) if event != cl.NULL else None

    def write_buffer(self, buf, host_array, blocking=True, size=None, offset=0,
                     wait_for=None, need_event=False):
        """Copies from host buffer to device buffer.

        Parameters:
            buf: Buffer object.
            host_array: numpy array.
            blocking: if the read is blocking.
            size: size in bytes to copy (None for entire numpy array).
            offset: offset in the device buffer.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event[]", 1) if need_event else cl.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        host_ptr, size = CL.extract_ptr_and_size(host_array, size)
        n = self.lib_.clEnqueueWriteBuffer(
            self.queue_, buf.buffer_, blocking, offset, size, host_ptr,
            n_events, wait_list, event)
        if n:
            raise CLRuntimeError("clEnqueueReadBuffer() failed with "
                                 "error %d" % (n), n)
        return Event(event[0]) if event != cl.NULL else None

    def flush(self):
        """Flushes the queue.
        """
        n = self.lib_.clFlush(self.queue_)
        if n:
            raise CLRuntimeError("clFlush() failed with error %d" % (n), n)

    def finish(self):
        """Waits for all previous commands issued to this queue to end.
        """
        n = self.lib_.clFinish(self.queue_)
        if n:
            raise CLRuntimeError("clFinish() failed with error %d" % (n), n)

    def release(self):
        if self.queue_ is not None:
            self.lib_.clReleaseCommandQueue(self.queue_)
            self.queue_ = None

    def __del__(self):
        self.release()


class Buffer(CL):
    """Holds OpenCL buffer.

    Attributes:
        buffer_: cffi OpenCL buffer handle.
        context: Context object assosiated with this buffer.
        flags: flags supplied for the creation of this buffer.
        host_array: host array reference, such as numpy array,
                    will be stored only if flags include CL_MEM_USE_HOST_PTR.
        size: size of the host array.
    """
    def __init__(self, context, flags, host_array, size=None):
        super(Buffer, self).__init__()
        self.buffer_ = None
        self.context = context
        self.flags = flags
        self.host_array = (host_array if flags & cl.CL_MEM_USE_HOST_PTR != 0
                           else None)
        host_ptr, size = CL.extract_ptr_and_size(host_array, size)
        err = cl.ffi.new("cl_int[]", 1)
        self.buffer_ = self.lib_.clCreateBuffer(
            context.context_, flags, size, host_ptr, err)
        if err[0]:
            self.buffer_ = None
            raise CLRuntimeError("clCreateBuffer() failed with "
                                 "error %d" % (err[0]), err[0])

    def release(self):
        if self.buffer_ is not None:
            self.lib_.clReleaseMemObject(self.buffer_)
            self.buffer_ = None

    def __del__(self):
        self.release()


class Kernel(CL):
    """Holds OpenCL kernel.

    Attributes:
        kernel_: cffi OpenCL kernel handle.
        program: Program object assosiated with this kernel.
        name: kernel name in the program.
    """
    def __init__(self, program, name):
        super(Kernel, self).__init__()
        self.kernel_ = None
        self.program = program
        self.name = name
        err = cl.ffi.new("cl_int[]", 1)
        ss = cl.ffi.new("char[]", name.encode("utf-8"))
        self.kernel_ = self.lib_.clCreateKernel(program.program_, ss, err)
        if err[0]:
            self.kernel_ = None
            raise CLRuntimeError("clCreateKernel() failed with "
                                 "error %d" % (err[0]), err[0])

    def set_arg(self, idx, vle, size=None):
        """Sets kernel argument.

        Parameters:
            idx: index of the kernel argument (zero-based).
            vle: kernel argument:
                 - for buffers should be an instance of Buffer,
                 - for scalars should be a numpy array slice
                   (k[0:1] for example),
                 - for NULL should be None,
                 - may be cffi pointer also, in such case size should be set.
            size: size of the vle (may be None for buffers and scalars).
        """
        if isinstance(vle, Buffer):
            arg_value = cl.ffi.new("cl_mem[]", 1)
            arg_value[0] = vle.buffer_
            arg_size = cl.ffi.sizeof("cl_mem")
        elif hasattr(vle, "__array_interface__"):
            arg_value = cl.ffi.cast("const void*",
                                    vle.__array_interface__["data"][0])
            arg_size = vle.nbytes if size is None else size
        elif vle is None:
            arg_value = cl.NULL
            arg_size = 0
        elif type(vle) == type(cl.NULL):  # cffi pointer
            arg_value = cl.ffi.cast("const void*", vle)
            if size is None:
                raise ValueError("size should be set in case of cffi pointer")
            arg_size = size
        else:
            raise ValueError("vle should be of type Buffer, "
                             "numpy array, cffi pointer or None "
                             "in Kernel::set_arg()")
        n = self.lib_.clSetKernelArg(self.kernel_, idx, arg_size, arg_value)
        if n:
            raise CLRuntimeError("clSetKernelArg() failed with "
                                 "error %d" % (n), n)

    def release(self):
        if self.kernel_ is not None:
            self.lib_.clReleaseKernel(self.kernel_)
            self.kernel_ = None

    def __del__(self):
        self.release()


class Program(CL):
    """Holds OpenCL program.

    Attributes:
        program_: cffi OpenCL program handle.
        context: Context object assosiated with this program.
        devices: list of Device objects assosiated with this program.
        build_logs: list of program build logs (same length as devices list).
        src: program source.
        include_dirs: list of include dirs.
        options: additional build options.
    """
    def __init__(self, context, devices, src, include_dirs=(), options=""):
        super(Program, self).__init__()
        self.program_ = None
        self.context = context
        self.devices = devices
        self.src = src
        self.include_dirs = list(include_dirs)
        self.options = options
        self.build_logs = []
        err = cl.ffi.new("cl_int[]", 1)
        ss = cl.ffi.new("char[]", src.encode("utf-8"))
        strings = cl.ffi.new("char*[]", 1)
        strings[0] = cl.ffi.cast("char*", ss)
        self.program_ = self.lib_.clCreateProgramWithSource(
            context.context_, 1, strings, cl.NULL, err)
        if err[0]:
            self.program_ = None
            raise CLRuntimeError("clCreateProgramWithSource() failed with "
                                 "error %d" % (err[0]), err[0])
        for dirnme in include_dirs:
            if not len(dirnme):
                continue
            options += " -I " + (dirnme if dirnme.find(" ") < 0
                                 else "\'%s\'" % (dirnme))
        self._build_program(devices, options.strip().encode("utf-8"))

    def get_kernel(self, name):
        """Returns Kernel object from its name.
        """
        return Kernel(self, name)

    def _build_program(self, devices, options):
        n_devices = len(devices)
        device_list = cl.ffi.new("cl_device_id[]", n_devices)
        for i, dev in enumerate(devices):
            device_list[i] = dev.device_
        n = self.lib_.clBuildProgram(self.program_, n_devices, device_list,
                                     options, cl.NULL, cl.NULL)
        del self.build_logs[:]
        log = cl.ffi.new("char[]", 65536)
        sz = cl.ffi.new("size_t[]", 1)
        for dev in device_list:
            m = self.lib_.clGetProgramBuildInfo(
                self.program_, dev, cl.CL_PROGRAM_BUILD_LOG, 65536, log, sz)
            if m or sz[0] <= 0:
                self.build_logs.append("")
                continue
            self.build_logs.append(
                (b"".join(log[0:sz[0] - 1])).decode("utf-8", "replace"))
        if n:
            raise CLRuntimeError(
                "clBuildProgram() failed with error %d\n"
                "Logs are:\n%s\nSource was:\n%s\n" %
                (n, "\n".join(self.build_logs), self.src), n)

    def release(self):
        if self.program_ is not None:
            self.lib_.clReleaseProgram(self.program_)
            self.program_ = None

    def __del__(self):
        self.release()


class Context(CL):
    """Holds OpenCL context.

    Attributes:
        context_: cffi OpenCL context handle.
        platform: Platform object assosiated with this context.
        devices: list of Device object assosiated with this context.
    """
    def __init__(self, platform, devices):
        super(Context, self).__init__()
        self.context_ = None
        self.platform = platform
        self.devices = devices
        props = cl.ffi.new("cl_context_properties[]", 3)
        props[0] = cl.CL_CONTEXT_PLATFORM
        props[1] = cl.ffi.cast("cl_context_properties", platform.platform_)
        props[2] = 0
        err = cl.ffi.new("cl_int[]", 1)
        n_devices = len(devices)
        device_list = cl.ffi.new("cl_device_id[]", n_devices)
        for i, dev in enumerate(devices):
            device_list[i] = dev.device_
        self.context_ = self.lib_.clCreateContext(
            props, n_devices, device_list, cl.NULL, cl.NULL, err)
        if err[0]:
            self.context_ = None
            raise CLRuntimeError("clCreateContext() failed with "
                                 "error %d" % (err[0]), err[0])

    def create_queue(self, device, flags=0):
        """Creates Queue object for the supplied device.

        Parameters:
            device: Device object.
            flags: queue flags (for example
                                CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE).

        Returns:
            Queue object.
        """
        return Queue(self, device, flags)

    def create_buffer(self, flags, host_array=None, size=None):
        """Creates Buffer object based on host_array.

        Parameters:
            host_array: numpy array of None.
            size: size if host_array is not a numpy array.

        Returns:
            Buffer object.
        """
        return Buffer(self, flags, host_array, size)

    def create_program(self, src, include_dirs=(), options="", devices=None):
        """Creates and builds OpenCL program from source
           for the supplied devices assosiated with this context.

        Parameters:
            src: program source.
            include_dirs: list of include directories.
            options: additional build options.
            devices: list of devices on which to build the program
                     (if None will build on all devices).
        Returns:
            Program object.
        """
        return Program(self, self.devices if devices is None else devices,
                       src, include_dirs, options)

    def release(self):
        if self.context_ is not None:
            self.lib_.clReleaseContext(self.context_)
            self.context_ = None

    def __del__(self):
        self.release()


class Device(CL):
    """OpenCL device.

    Attributes:
        device_: cffi OpenCL device id.
        platform: Platform object assosiated with this device.
        type: OpenCL type of the device (integer).
        name: OpenCL name of the device.
        version: OpenCL version number of the device (float).
        version_string: OpenCL version string of the device.
        vendor: OpenCL vendor name of the device.
        vendor_id: OpenCL vendor id of the device (integer).
        memsize: global memory size of the device.
        memalign: align in bytes, required for clMapBuffer.
    """
    def __init__(self, device_, platform):
        super(Device, self).__init__()
        self.device_ = device_
        self.platform = platform

        sz = cl.ffi.new("size_t[]", 1)
        tpe = cl.ffi.new("cl_device_type[]", 1)
        n = self.lib_.clGetDeviceInfo(device_, cl.CL_DEVICE_TYPE,
                                      8, tpe, sz)
        self.type = tpe[0] if not n else None

        nme = cl.ffi.new("char[]", 256)
        n = self.lib_.clGetDeviceInfo(device_, cl.CL_DEVICE_NAME,
                                      256, nme, sz)
        self.name = ((b"".join(nme[0:sz[0] - 1])).decode("utf-8")
                     if not n else None)

        n = self.lib_.clGetDeviceInfo(device_, cl.CL_DEVICE_OPENCL_C_VERSION,
                                      256, nme, sz)
        s = (b"".join(nme[0:sz[0] - 1])).decode("utf-8") if not n else None
        self.version_string = s
        n = len("OpenCL C ")
        m = s.find(" ", n)
        try:
            self.version = float(s[n:m])
        except ValueError:
            self.version = 0.0

        n = self.lib_.clGetDeviceInfo(device_, cl.CL_DEVICE_VENDOR,
                                      256, nme, sz)
        self.vendor = ((b"".join(nme[0:sz[0] - 1])).decode("utf-8")
                       if not n else None)

        vendor_id = cl.ffi.new("cl_uint[]", 1)
        n = self.lib_.clGetDeviceInfo(device_, cl.CL_DEVICE_VENDOR_ID,
                                      4, vendor_id, sz)
        self.vendor_id = vendor_id[0] if not n else None

        memsize = cl.ffi.new("cl_ulong[]", 1)
        n = self.lib_.clGetDeviceInfo(device_, cl.CL_DEVICE_GLOBAL_MEM_SIZE,
                                      8, memsize, sz)
        self.memsize = memsize[0] if not n else None

        memalign = cl.ffi.new("cl_uint[]", 1)
        n = self.lib_.clGetDeviceInfo(
            device_, cl.CL_DEVICE_MEM_BASE_ADDR_ALIGN, 4, memalign, sz)
        self.memalign = memalign[0] if not n else None


class Platform(CL):
    """OpenCL platform.

    Attributes:
        platform_: cffi OpenCL platform id.
        devices: list of Device objects available on this platform.
        name: OpenCL name of the platform.
    """
    def __init__(self, platform_):
        super(Platform, self).__init__()
        self.platform_ = platform_

        sz = cl.ffi.new("size_t[]", 1)
        nme = cl.ffi.new("char[]", 256)
        n = self.lib_.clGetPlatformInfo(platform_, cl.CL_PLATFORM_NAME,
                                        256, nme, sz)
        self.name = ((b"".join(nme[0:sz[0] - 1])).decode("utf-8")
                     if not n else None)

        nn = cl.ffi.new("cl_uint[]", 1)
        n = self.lib_.clGetDeviceIDs(platform_, cl.CL_DEVICE_TYPE_ALL,
                                     0, cl.NULL, nn)
        if n:
            raise CLRuntimeError("clGetDeviceIDs() failed "
                                 "with error %d" % (n), n)
        ids = cl.ffi.new("cl_device_id[]", nn[0])
        n = self.lib_.clGetDeviceIDs(platform_, cl.CL_DEVICE_TYPE_ALL,
                                     nn[0], ids, nn)
        if n:
            raise CLRuntimeError("clGetDeviceIDs() failed "
                                 "with error %d" % (n), n)
        self.devices = list(Device(dev_id, self) for dev_id in ids)

    def create_context(self, devices):
        """Creates OpenCL context on this platform and selected devices.

        Parameters:
            devices: list of Device objects.

        Returns:
            Context object.
        """
        return Context(self, devices)


class Platforms(CL):
    """List of OpenCL plaforms.

    Attributes:
        platforms: list of Platform objects.
    """
    def __init__(self):
        cl.initialize()
        super(Platforms, self).__init__()
        nn = cl.ffi.new("cl_uint[]", 1)
        n = self.lib_.clGetPlatformIDs(0, cl.NULL, nn)
        if n:
            raise CLRuntimeError("clGetPlatformIDs() failed "
                                 "with error %d" % (n), n)
        ids = cl.ffi.new("cl_platform_id[]", nn[0])
        n = self.lib_.clGetPlatformIDs(nn[0], ids, nn)
        if n:
            raise CLRuntimeError("clGetPlatformIDs() failed "
                                 "with error %d" % (n), n)
        self.platforms = list(Platform(p_id) for p_id in ids)

    def dump_devices(self):
        """Returns string with information about OpenCL platforms and devices.
        """
        if not len(self.platforms):
            return "No OpenCL devices available."
        lines = []
        for i, platform in enumerate(self.platforms):
            lines.append("Platform %d: %s" % (i, platform.name.strip()))
            for j, device in enumerate(platform.devices):
                lines.append("\tDevice %d: %s (%d Mb, %d align, %s)" % (
                    j, device.name.strip(), device.memsize // (1024 * 1024),
                    device.memalign, device.version_string.strip()))
        return "\n".join(lines)

    def create_some_context(self):
        """Returns Context object with some OpenCL platform, devices attached.

        If environment variable PYOPENCL_CTX is set and not empty,
        gets context based on it, format is:
        <platform number>:<comma separated device numbers>
        (Examples: 0:0 - first platform, first device,
                   1:0,2 - second platform, first and third devices).

        If PYOPENCL_CTX is not set and os.isatty(0) == True, then
        displays available devices and reads line from stdin in the same
        format as PYOPENCL_CTX.

        Else chooses first platform and device.
        """
        if len(self.platforms) == 1 and len(self.platforms[0].devices) == 1:
            return self.platforms[0].create_context(self.platforms[0].devices)
        import os
        ctx = os.environ.get("PYOPENCL_CTX")
        if ctx is None or not len(ctx):
            if os.isatty(0):
                import sys
                sys.stdout.write(
                    "\nEnter "
                    "<platform number>:<comma separated device numbers> or "
                    "set PYOPENCL_CTX environment variable.\n"
                    "Examples: 0:0 - first platform, first device;\n"
                    "          1:0,2 - second platform, first and third "
                    "devices.\n"
                    "\nOpenCL devices available:\n\n%s\n\n" %
                    (self.dump_devices()))
                sys.stdout.flush()
                ctx = sys.stdin.readline().strip()
            else:
                ctx = ""
        idx = ctx.find(":")
        if idx >= 0:
            try:
                platform_number = int(ctx[:idx]) if len(ctx[:idx]) else 0
            except ValueError:
                raise ValueError("Incorrect platform number")
            ctx = ctx[idx + 1:]
        else:
            platform_number = 0
        device_strings = ctx.split(",")
        device_numbers = []
        try:
            for s in device_strings:
                device_numbers.append(int(s) if len(s) else 0)
        except ValueError:
            raise ValueError("Incorrect device number")
        try:
            platform = self.platforms[platform_number]
        except IndexError:
            raise IndexError("Platform index is out of range")
        devices = []
        try:
            for i in device_numbers:
                devices.append(platform.devices[i])
        except IndexError:
            raise IndexError("Devicve index is out of range")
        return platform.create_context(devices)
