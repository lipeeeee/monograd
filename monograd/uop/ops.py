from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import weakref
from monograd.device import Buffer, Device
from monograd.dtype import DType, dtypes
from monograd.uop import GroupOp, Ops

# only allow unique UOps
class UOpMetaClass(type):
  ucache: dict[tuple, weakref.ReferenceType[UOp]] = weakref.WeakValueDictionary()
  def __call__(self, *args, **kwds):
    if (ret:=UOpMetaClass.ucache.get(args, None)) is not None: return ret
    UOpMetaClass.ucache[args] = ret = super().__call__(*args)
    return ret

# NOTE: storing buffers as pointers outside UOp is kinda sad but it has to be done like this
# tinygrad and pytorch also solve this issue like this :/
# monograd's solution could've been to make an OP specifically to store buffers and send them through UOp.src
_uop_buffers:weakref.WeakKeyDictionary[UOp, Buffer] = weakref.WeakKeyDictionary()

@dataclass(eq=True, slots=True, weakref_slot=True)
class UOp(metaclass=UOpMetaClass): 
  op: Ops
  dtype: DType = dtypes.void
  src: tuple[UOp, ...] = tuple()
  arg: Any = None

  @property
  def device(self) -> Device:
    if self.op is Ops.LOAD: return _uop_buffers[self].device
    if self.op is Ops.COPY: return self.arg
    raise NotImplementedError("unkown op")
  @property
  def shape(self) -> tuple:
    if self.op is Ops.LOAD: return self.arg
    raise NotImplementedError("unkown op")
  @property
  def buffer(self) -> Buffer: return _uop_buffers[self]
  def assign_buffer(self, device:Device, size:int, initial_value=None) -> Buffer:
    assert self.op in GroupOp.Buffer, f"op {self.op} can't have a buffer attatched"
    if (dret:=_uop_buffers.get(self, None)) is not None: return dret
    ret = Buffer(device, size, self.dtype)
    ret.allocate(initial_value)
    _uop_buffers[self] = ret
    return ret
  def __del__(self): 
    try:
      del _uop_buffers[self]
      del UOpMetaClass.ucache[(self.op, self.dtype, self.src, self.arg)]
    except AttributeError: pass
  def __repr__(self):
    return f"<UOp {self.op} dtype={self.dtype.name} arg={self.arg}>"