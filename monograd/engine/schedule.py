from __future__ import annotations
import weakref
from math import prod
from enum import auto, IntEnum
from dataclasses import dataclass
from monograd.device import Device
from monograd.dtype import DType
from monograd.uop import GroupOp, Ops
from monograd.uop.ops import UOp
from monograd.utils import DEBUG, toposort

def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
  # computes strides[i] = strides[i-1] * shape[i-1]  |||| strides[len(shape) - 1] = 1
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

_bufferref_cache: weakref.WeakKeyDictionary[UOp, BufferRef] = weakref.WeakKeyDictionary()
@dataclass  
class BufferRef:
  uop: UOp
  shape: tuple
  strides: tuple[int, ...]

  @staticmethod
  def from_uop(uop:UOp) -> BufferRef: # Main job is to compute strides for a given root input uop
    if uop in _bufferref_cache: return _bufferref_cache[uop] 
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
          for old, new, st in zip(shape, new_shape, strides))
        shape = new_shape 
      elif op.op is Ops.PERMUTE:
        order = op.arg
        shape = op.shape
        strides = tuple(strides[i] for i in order)
    ret = BufferRef(cur, shape, strides)
    _bufferref_cache[uop] = ret # NOTE: is it `uop` or `cur` key?
    if DEBUG >= 4: print(f"BufferRef.from_uop creating reference: {ret}")
    return ret
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
      if stride == 0: continue # broadcast dim, skip
      elif stride == 1: terms.append(coord)
      else: terms.append(f"{coord} * {stride}")
    if not terms: return "0"
    return " + ".join(terms)
  def __repr__(self):
    return f"BufferRef(op={self.uop.op}, shape={self.shape}, strides={self.strides})"

class Scheduler:
  scheduled_kernels: list[KernelTask]
  def __init__(self):
    self.scheduled_kernels: list[KernelTask] = []
    self._current_group: list[UOp] = []
  def run_scheduler(self, root:UOp) -> list[KernelTask]:
    nodes = toposort(root, lambda u: u.src)
    for node in nodes:
      if node.op in GroupOp.Input:    continue
      if node.op is Ops.SINK:         continue
      if node.op in GroupOp.Movement: continue
      elif is_fusable(node): self._current_group.append(node)
      elif is_boundary(node):
        self._flush(TaskKind.ELEMENTWISE)
        kind = TaskKind.BLAS if node.op in GroupOp.BLAS else TaskKind.REDUCE if node.op in GroupOp.Reduce else TaskKind.COPY
        self.scheduled_kernels.append(KernelTask(kind, [node], self._collect_inputs([node])))
    self._flush(TaskKind.ELEMENTWISE)  # flush any remaining ops
    return self.scheduled_kernels
  def _flush(self, kind:TaskKind):
    if not self._current_group: return
    self.scheduled_kernels.append(KernelTask(kind, self._current_group, self._collect_inputs(self._current_group)))
    self._current_group = []
  def _collect_inputs(self, ops:list[UOp]) -> list[BufferRef]:
    group_ids = {id(u) for u in ops}
    seen: set[int] = set()
    refs: list[BufferRef] = []
    for op in ops:
      for src in op.src:
        if id(src) in seen: continue
        ref = BufferRef.from_uop(src)
        if ref.uop.op is Ops.CONST: continue   # inline, skip
        if id(ref.uop) in group_ids: continue  # internal to group, skip
        seen.add(id(src))
        refs.append(ref)
    return refs

class TaskKind(IntEnum):
  ELEMENTWISE = auto(); REDUCE = auto(); BLAS = auto(); COPY = auto()
