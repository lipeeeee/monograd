from __future__ import annotations
import numpy as np
from typing import Self
from monograd.uop import Ops
from monograd.dtype import ConstType

class MathMixin:
  def _reduceop(self, op:Ops, axis:int|tuple[int, ...]|None=None, keepdim:bool=False) -> Self:
    raise NotImplementedError
  def _unop(self, op:Ops) -> Self:
    raise NotImplementedError
  def _binop(self, op:Ops, x, reverse:bool=False) -> Self:
    raise NotImplementedError
  def matmul(self, x) -> Self:
    raise NotImplementedError
  def const_like(self, x) -> Self:
    raise NotImplementedError
  @property
  def device(self):
    raise NotImplementedError
  def ufix(self, x:Self|ConstType|list|np.ndarray):
    return self.const_like(x) if not isinstance(x, MathMixin) else x

  # bin
  def add(self, x:Self|ConstType, reverse:bool=False) -> Self:  return self._binop(Ops.ADD, self.ufix(x), reverse)
  def mul(self, x:Self|ConstType, reverse:bool=False) -> Self:  return self._binop(Ops.MUL, self.ufix(x), reverse)
  def sub(self, x:Self|ConstType, reverse:bool=False) -> Self:  return self.add(-self.ufix(x), reverse) # decomposed
  def div(self, x:Self|ConstType, reverse:bool=False) -> Self:  return self.mul(self.ufix(x).recip(), reverse) # decomposed
  def pow(self, x, reverse:bool=False) -> Self:                 return self._binop(Ops.POW, self.ufix(x), reverse)
  # un
  def log(self) -> Self:   return self._unop(Ops.LOG)
  def exp(self) -> Self:   return self._unop(Ops.EXP)
  def sqrt(self) -> Self:  return self._unop(Ops.SQRT)
  def sin(self) -> Self:   return self._unop(Ops.SIN)
  def relu(self) -> Self:  return self._unop(Ops.RELU)
  def recip(self) -> Self: return self._unop(Ops.RECIP)
  # red
  def sum(self, axis:int|tuple[int, ...]|None=None, keepdim:bool=False): return self._reduceop(Ops.SUM, axis, keepdim)
  def maxreduce(self): raise NotImplementedError # proably needs axis and keepdim and keep it the same as sum

  def __add__(self, x:Self|ConstType):  return self.add(x)
  def __mul__(self, x:Self|ConstType):  return self.mul(x)
  def __sub__(self, x:Self|ConstType):  return self.sub(x)
  def __radd__(self, x:Self|ConstType): return self.add(x, True)
  def __rmul__(self, x:Self|ConstType): return self.mul(x, True)
  def __rsub__(self, x:Self|ConstType): return self.sub(x, True)
  def __matmul__(self, x):              return self.matmul(x)

  def __truediv__(self, x):         return self.div(x)
  def __rtruediv__(self, x):        return self.div(x, True)
  def __pow__(self, x):             return self.pow(x)
  def __rpow__(self, x):            return self.pow(x, True)
  def __neg__(self):                return self.mul(-1) # decomposed
