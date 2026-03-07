from __future__ import annotations
from typing import Self
from monograd.uop import Ops
from monograd.dtype import ConstType

class MathMixin:
  def _binop(self, op:Ops, x, reverse:bool=False) -> Self:
    raise NotImplementedError
  def const_like(self, x) -> Self:
    raise NotImplementedError
  def ufix(self, x:Self|ConstType):
    return self.const_like(x) if not isinstance(x, MathMixin) else x

  def add(self, x:Self|ConstType, reverse:bool=False) -> Self:
    return self._binop(Ops.ADD, self.ufix(x), reverse)
  def mul(self, x:Self|ConstType, reverse:bool=False) -> Self:
    return self._binop(Ops.MUL, self.ufix(x), reverse)
  def sub(self, x:Self|ConstType, reverse:bool=False) -> Self:
    return self._binop(Ops.SUB, self.ufix(x), reverse)

  def __add__(self, x:Self|ConstType):
    return self.add(x)
  def __mul__(self, x:Self|ConstType):
    return self.mul(x)
  def __sub__(self, x:Self|ConstType):
    return self.sub(x)
  def __radd__(self, x:Self|ConstType):
    return self.add(x, True)
  def __rmul__(self, x:Self|ConstType):
    return self.mul(x, True)
  def __rsub__(self, x:Self|ConstType):
    return self.sub(x, True)
