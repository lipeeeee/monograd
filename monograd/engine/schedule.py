from __future__ import annotations
from math import prod
from enum import auto, IntEnum
from dataclasses import dataclass
from monograd.device import Device
from monograd.dtype import DType
from monograd.uop import GroupOp, Ops
from monograd.uop.ops import UOp

def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
  if len(shape) == 0: return ()
  strides = [1] * len(shape)
  for i in range(len(shape) - 2, -1, -1):
    strides[i] = strides[i+1] * shape[i+1]
  return tuple(strides)

# uop utils
def is_scalar(uop:UOp , strides=(1,)) -> bool: return uop.op is Ops.CONST or all(s == 0 for s in strides)
def is_fusable(uop:UOp) -> bool: return uop.op in GroupOp.Unary | GroupOp.Binary # NOTE: Not checking CAST because it is in Unary
def is_invisible(uop:UOp) -> bool: return uop.op in GroupOp.Movement
def is_boundary(uop:UOp) -> bool: return uop.op in GroupOp.BLAS | {Ops.COPY}

@dataclass
class KernelTask: # what holds scheduled graph (list)
  kind: TaskKind
  ops: list[UOp]
  inputs: list[BufferRef]

  # output metadata is always gotten from last uop
  @property
  def output_dtype(self) -> DType: return self.ops[-1].dtype
  @property
  def output_shape(self) -> tuple[int, ...]: return self.ops[-1].shape
  @property
  def output_device(self) -> Device: return self.ops[-1].device

@dataclass  
class BufferRef:
  uop: UOp
  shape: tuple
  strides: tuple[int, ...]

  @staticmethod
  def from_uop(uop:UOp) -> BufferRef: # Main job is to compute strides for a given input uop
    movement_chain: list[UOp] = []
    cur = uop
    while cur.op in GroupOp.Movement:
      movement_chain.append(cur)
      cur = cur.src[0]
    shape = cur.shape
    strides = _row_major_strides(shape)
    # go through the movop chain and compute final stride
    for op in reversed(movement_chain):
      if op.op is Ops.RESHAPE:
        shape = op.shape
        strides = _row_major_strides(shape)
      elif op.op is Ops.EXPAND:
        new_shape = op.shape
        strides = tuple(
          0 if old == 1 and new > 1 else st
          for old, new, st in zip(shape, new_shape, strides)
        )
        shape = new_shape 
      elif op.op is Ops.PERMUTE:
        order = op.arg
        shape = op.shape
        strides = tuple(strides[i] for i in order)
    return BufferRef(cur, shape, strides)
  def index_expr(self, gid:str, output_shape:tuple[int, ...]) -> str: # generates C index expr
    assert self.shape == output_shape, f"is this a bug? shape mismatch generating C index {self.shape} != {output_shape}"
    if is_scalar(self.uop, self.strides): return "0"
    # build per-dim coordinate expressions from flat gid
    # e.g. for output_shape=(2,3):
    #   dim 0: coord = gid / 3
    #   dim 1: coord = gid % 3
    coords: list[str] = []
    remaining = gid
    for i, dim_size in enumerate(output_shape):
      below = prod(output_shape[i+1:])  # product of all dims below this one
      if below == 1:
        coords.append(remaining)
      else:
        coords.append(f"({remaining} / {below})")
        remaining = f"({remaining} % {below})"
    # build flat index from strides
    # skip dims where stride is 0 (broadcast — contributes nothing)
    terms: list[str] = []
    for coord, stride in zip(coords, self.strides):
      if stride == 0: continue            # broadcast dim, skip
      elif stride == 1: terms.append(coord)
      else: terms.append(f"{coord} * {stride}")
    if not terms: return "0"
    return " + ".join(terms)
  def __repr__(self):
    return (f"BufferRef(op={self.uop.op}, shape={self.shape}, strides={self.strides})")

class TaskKind(IntEnum):
  ELEMENTWISE = auto(); REDUCE = auto(); BLAS = auto(); COPY = auto()
