from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from monograd.dtype import DType, dtypes
from monograd.uop import Ops

# only allow unique UOps
class UOpMetaClass(type):
  ucache: dict[tuple, UOp] = {}
  def __call__(self, *args, **kwds):
    if (ret:=UOpMetaClass.ucache.get(args, None)) is not None: return ret
    UOpMetaClass.ucache[args] = ret = super().__call__(*args)
    return ret

@dataclass(eq=True, slots=True)
class UOp(metaclass=UOpMetaClass): 
  op: Ops
  dtype: DType = dtypes.void
  src: tuple[UOp, ...] = tuple()
