from __future__ import annotations
import functools
import pyopencl as cl
import numpy as np
from enum import IntEnum, auto
from monograd.dtype import DType
from monograd.utils import flat_mv

# Lazy context
class OpenCLContext:
  @classmethod
  @functools.lru_cache(maxsize=1)
  def cl_ctx(cls) -> cl.Buffer: return cl.create_some_context()
  @classmethod
  @functools.lru_cache(maxsize=1)
  def cl_queue(cls) -> cl.CommandQueue: return cl.CommandQueue(cls.cl_ctx())
CL_CTX, CL_QUEUE = OpenCLContext.cl_ctx(), OpenCLContext.cl_queue() # NOTE: this still executes even if we only use CPU tensors, yes suboptimal

class Device(IntEnum):
  CPU = auto()
  GPU = auto() # GPU = OPENCL

  def __str__(self): return self.name.lower()
  def __repr__(self): return str(self)

DeviceLike = str | Device
def to_device(device: DeviceLike) -> Device: return device if isinstance(device, Device) else getattr(Device, device.upper())

# TODO: copyin and copyout should have dtype assertions that cause byte math to fail
class Buffer:
  def __init__(self, device:Device, size:int, dtype:DType,
               base:Buffer|None=None, offset:int=0):
    self.device, self.size, self.dtype, self._base, self.offset = device, size, dtype, base, offset
    if base is None:
      assert offset == 0, "base buffers can't have offset"
    else:
      assert base._base is None, "base can't have a base"
      assert device == base.device, f"base must have the same device {device} vs {base.device}"
  @property
  def base(self) -> Buffer: return self._base if self._base is not None else self
  @property
  def nbytes(self) -> int: return self.size*self.dtype.itemsize
  def is_allocated(self) -> bool: return self.base.is_allocated() if self._base is not None else hasattr(self, "_buf")
  def ensure_allocated(self) -> Buffer: return self.allocate() if not self.is_allocated() else self
  def allocate(self, initial_value=None) -> Buffer:
    assert not self.is_allocated(), "can't allocate already allocated buffer"
    if self.base is not self:
      self.base.ensure_allocated()
      self._buf = self.base._buf # NOTE: idk about this because self might have an offset and not fit base _buf.size
      return self
    if self.device == Device.CPU:
      if initial_value is not None: self._buf = np.array(initial_value, dtype=self.dtype.np_dtype)
      else: self._buf = np.empty(self.nbytes, dtype=np.uint8)
    elif self.device == Device.GPU:
      if initial_value is not None:
        flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
        hostbuf = np.ascontiguousarray(initial_value, dtype=self.dtype.np_dtype)
      else: hostbuf = None; flags = cl.mem_flags.READ_WRITE
      self._buf = cl.Buffer(CL_CTX, flags, self.nbytes, hostbuf=hostbuf)
    assert self.is_allocated(), f"couldn't allocate on device {self.device}"
    return self
  def copyin(self, mv: memoryview): # Copy from *mv* to this buffer
    assert self.is_allocated(), "buffer should be allocated in order to do copy ops"
    mv = flat_mv(memoryview(mv))
    byte_offset = self.offset * self.dtype.itemsize
    if self.device == Device.CPU:
      dest_view = self._buf.view(np.uint8)[byte_offset:byte_offset+self.nbytes]
      np.copyto(dest_view, np.frombuffer(mv, dtype=np.uint8))
    elif self.device == Device.GPU:
      cl.enqueue_copy(CL_QUEUE, self._buf, mv, device_offset=byte_offset, is_blocking=True)
  def copyout(self, mv: memoryview) -> memoryview: # Copy from this buffer to *mv*
    assert self.is_allocated(), "buffer should be allocated in order to do copy ops"
    mv = flat_mv(memoryview(mv))
    byte_offset = self.offset * self.dtype.itemsize
    if self.device == Device.CPU:
      src_view = self._buf.view(np.uint8)[byte_offset:byte_offset+self.nbytes]
      np.copyto(np.frombuffer(mv, dtype=np.uint8), src_view)
    elif self.device == Device.GPU:
      cl.enqueue_copy(CL_QUEUE, mv, self._buf, device_offset=byte_offset, is_blocking=True)
    assert isinstance(mv, memoryview), f"this didn't work as expected: mv({mv}) is type {type(mv)}, need to convert to type memoryview"
    return mv
  def as_buffer(self) -> memoryview: return self.copyout(memoryview(np.empty(self.size, dtype=self.dtype.np_dtype)))
  def __repr__(self):
    prefix = "Base" if self.base is self else f"View(base={hex(id(self.base))}, offset={self.offset})"
    ptr_info = f"ptr={type(self._buf)}" if hasattr(self, "_buf") else "ptr=None"
    return f"<{prefix}Buffer on {self.device} {self.size}x{self.dtype.name} {ptr_info}>"
  def __del__(self):
    if not hasattr(self, "_buf") and self.base is not self: return
    try:
      if hasattr(self._buf, "release"): self._buf.release() # CL gpu release
    except AttributeError: pass # dumb python __del__ stuff
    self._buf = None