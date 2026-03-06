from __future__ import annotations
from typing import Self
from math import prod
from monograd.utils import argfix
from monograd.uop import Ops

# taken from tinygrad/movementmixin.py
def _align_left(*shapes:tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
  # unsqueeze left to make every shape same length
  max_dim = max(len(shape) for shape in shapes)
  return tuple((1,) * (max_dim - len(shape)) + shape for shape in shapes)

class MovementMixin:
  def _mop(self, op:Ops, arg) -> Self:
    raise NotImplementedError
  @property
  def shape(self) -> tuple[int, ...]:
    raise NotImplementedError
  @property
  def ndim(self) -> int:
    raise NotImplementedError

  def _broadcast_to(self, new_shape:tuple) -> Self:
    if self.shape == new_shape: return self
    assert self.ndim >= len(new_shape), f"cannot broadcast tensor to fewer dimensions tried {self.shape}->{new_shape}"
    shape, _ = _align_left(self.shape, new_shape)
    # for each dimension, check either dim is 1, or it does not change
    assert all(s == ns or s == 1 for s, ns in zip(shape, new_shape)), "cannot broadcast {self.shape}->{new_shape}"
    reshaped = self.reshape(shape)
    ret = reshaped._mop(Ops.EXPAND, new_shape)
    return reshaped if ret.shape == reshaped.shape else ret

  def expand(self, shape, *args) -> Self:
    new_shape = tuple(from_ if to == -1 or to is None else to for from_, to in zip(*(_align_left(self.shape, argfix(shape, *args)))))
    return self._broadcast_to(new_shape)
  def reshape(self, shape, *args) -> Self:
    new_shape = tuple([s if s is not None else self.shape[i] for i, s in enumerate(argfix(shape, *args))])
    # resolve -1
    assert (c := new_shape.count(-1)) <= 1, f"only one dimension can be inferred using -1, getting {new_shape}"
    if c:
      new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else s for s in new_shape])
    assert prod(self.shape) == prod(new_shape), "size mismatch, can't reshape ({self.shape}) -> ({new_shape})"
    ret = self._mop(Ops.RESHAPE, new_shape)
    return self if ret.shape == self.shape else ret
