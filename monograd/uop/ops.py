from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import weakref
from monograd.device import Device
from monograd.dtype import DType, dtypes
from monograd.uop import Ops

# only allow unique UOps
class UOpMetaClass(type):
  ucache: dict[tuple, weakref.ReferenceType[UOp]] = weakref.WeakValueDictionary()
  def __call__(self, *args, **kwds):
    if (ret:=UOpMetaClass.ucache.get(args, None)) is not None: return ret
    UOpMetaClass.ucache[args] = ret = super().__call__(*args)
    return ret

@dataclass(eq=True, slots=True, weakref_slot=True)
class UOp(metaclass=UOpMetaClass): 
  op: Ops
  dtype: DType = dtypes.void
  src: tuple[UOp, ...] = tuple()
  arg: Any = None

  @property
  def device(self) -> Device:
    if self.op is Ops.LOAD: return self.arg.device
    if self.op is Ops.COPY: return self.arg
    raise NotImplementedError("unkown op")
  @property
  def shape(self) -> tuple:
    pass
  def __repr__(self):
    return f"<UOp {self.op} dtype={self.dtype.name} arg={self.arg}>"