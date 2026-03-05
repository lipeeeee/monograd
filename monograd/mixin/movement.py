from __future__ import annotations
import math
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
  @property
  def shape(self) -> tuple[int, ...]:
    raise NotImplementedError

  def expand(self, shape:tuple) -> Self:
    assert len(self.shape) == len(shape), f"expand needs lengths to be the same: len({self.shape}) != len({shape})"

  def reshape(self, shape:tuple) -> Self: ...
