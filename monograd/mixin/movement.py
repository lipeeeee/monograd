from __future__ import annotations
from typing import Self
from monograd.uop import Ops

# taken from tinygrad/movementmixin.py
def _align_left(*shapes:tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
  # unsqueeze left to make every shape same length
  max_dim = max(len(shape) for shape in shapes)
  return tuple((1,) * (max_dim - len(shape)) + shape for shape in shapes)

class MovementMixin:
  def _mop(self, op:Ops, arg) -> Self:
    raise NotImplementedError

  def expand(self, shape) -> Self: ...
  def reshape(self, shape) -> Self: ...
