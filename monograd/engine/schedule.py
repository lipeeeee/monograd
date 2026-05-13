from __future__ import annotations
import weakref
from math import prod
from enum import auto, IntEnum
from dataclasses import dataclass
from collections import defaultdict
from monograd.device import Device
from monograd.dtype import ConstType, DType
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
def is_fusable(uop:UOp) -> bool: return uop.op in GroupOp.ALU # NOTE: Not checking CAST because it is in Unary
def is_invisible(uop:UOp) -> bool: return uop.op in GroupOp.Movement | GroupOp.Input | {Ops.SINK}
def is_boundary(uop:UOp) -> bool: return uop.op in GroupOp.BLAS | GroupOp.Reduce | {Ops.COPY, Ops.CONTIGUOUS}

class TaskKind(IntEnum):
  ELEMENTWISE = auto(); REDUCE_FULL = auto(); REDUCE_STRIDED = auto(); BLAS = auto(); COPY = auto()

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
  @property
  def output_uop(self) -> UOp: return self.ops[-1]

  def __repr__(self) -> str:
    op_chain = " → ".join(o.op.name for o in self.ops)
    inputs_str = "\n".join(f"    {ref}" for ref in self.inputs)
    return (
      f"KernelTask({self.kind.name})\n"
      f"  ops    : {op_chain}\n"
      f"  output : shape={self.output_shape} dtype={self.output_dtype} device={self.output_device}\n"
      f"  inputs ({len(self.inputs)}):\n{inputs_str}"
    )

_bufferref_cache: weakref.WeakKeyDictionary[UOp, BufferRef] = weakref.WeakKeyDictionary()
@dataclass  
class BufferRef:
  uop: UOp # root uop, most cases is just LOAD
  shape: tuple
  strides: tuple[int, ...]
  padding_op: UOp|None

  @staticmethod
  def from_uop(uop:UOp) -> BufferRef: # Main job is to compute strides for a given root input uop
    # given MUL op will return itself as a BufferRef
    # given Movement op will compute strides until leaf uop, returning leaf with correct strides
    if uop in _bufferref_cache: return _bufferref_cache[uop] 
    movement_chain:list[UOp] = []
    cur = uop
    while cur.op in GroupOp.Movement:
      movement_chain.append(cur)
      cur = cur.src[0]
    shape:tuple[int, ...] = cur.shape
    strides:tuple[int, ...] = _row_major_strides(shape)
    padding_op:UOp|None = None
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
      elif op.op is Ops.PAD:
        assert padding_op is None, f"monograd only supports 1 padding OP per buffer(if u need more than 1 maybe smth is wrong)"
        padding_op = op
        shape = op.shape
    ret = BufferRef(cur, shape, strides, padding_op)
    _bufferref_cache[uop] = ret
    return ret
  def index_expr(self, gid: str = "gid") -> tuple[str, str]:
    if is_scalar(self.uop, self.strides): return "0", "true"
    coords: list[str] = []
    remaining = gid
    for i, dim_size in enumerate(self.shape):
      if dim_size == 1:
        coords.append("0")
        continue
      below = prod(self.shape[i+1:])
      if below == 1: 
        coords.append(remaining)
      else:
        coords.append(f"({remaining} / {below})")
        remaining = f"({remaining} % {below})"
    if not self.is_padded:
      terms = [f"{c}" if s == 1 else f"({c}) * {s}" for c, s in zip(coords, self.strides) if s != 0]
      return " + ".join(terms) if terms else "0", "true"
    terms: list[str] = []
    conditions: list[str] = []
    valid_mask: tuple[tuple[int, int], ...] = self.padding_op.arg[1] # type: ignore
    for i, (coord, stride, (pad_before, pad_after)) in enumerate(zip(coords, self.strides, valid_mask)):
      # Build Mask
      if pad_before > 0:
        conditions.append(f"({coord} >= {pad_before})")
      if pad_after > 0:
        conditions.append(f"({coord} < {self.shape[i] - pad_after})")
      # Build Index (Shift coordinate back to physical memory)
      if stride == 0: continue
      shifted_coord = f"({coord} - {pad_before})" if pad_before > 0 else coord
      if stride == 1: 
        terms.append(shifted_coord)
      else: 
        terms.append(f"{shifted_coord} * {stride}")
    index_math = " + ".join(terms) if terms else "0"
    mask_math = " && ".join(conditions) if conditions else "true"
    return index_math, mask_math
  def reduce_index_expr_multi(self, reduce_axes:tuple[int, ...], gid:str="gid", k:str="k") -> tuple[str, str]: # multi-axis reduction
    if is_scalar(self.uop, self.strides): return "0", "true"
    reduce_axes = tuple(sorted(reduce_axes))
    output_shape = [s for i, s in enumerate(self.shape) if i not in reduce_axes]
    reduce_shape = [self.shape[i] for i in reduce_axes]
    # Unpacks a flat 1D index into N-dimensional coordinates based on a target shape
    def unpack_index(flat_var: str, shape: list[int]) -> list[str]:
      if not shape: return []
      coords: list[str] = []
      remaining = flat_var
      for i in range(len(shape)):
        below = prod(shape[i+1:])
        if below == 1: 
          coords.append(remaining)
        else:
          coords.append(f"({remaining} / {below})")
          remaining = f"({remaining} % {below})"
      return coords
    gid_coords = unpack_index(gid, output_shape)
    k_coords = unpack_index(k, reduce_shape)
    # Merge output and reduction coordinates back into the original memory layout
    input_coords: list[str] = []
    gid_idx, k_idx = 0, 0
    for i in range(len(self.shape)):
      if i in reduce_axes:
        input_coords.append(k_coords[k_idx])
        k_idx += 1
      else:
        input_coords.append(gid_coords[gid_idx])
        gid_idx += 1
    # fast path
    if not self.is_padded:
      terms = [f"{c}" if s == 1 else f"({c}) * {s}" for c, s in zip(input_coords, self.strides) if s != 0]
      return " + ".join(terms) if terms else "0", "true"
    # Apply physical strides and padding guards
    terms: list[str] = []
    mask_conditions: list[str] = []
    valid_mask: tuple[tuple[int, int], ...] = self.padding_op.arg[1] # type: ignore
    for i, (coord, stride, (start, end)) in enumerate(zip(input_coords, self.strides, valid_mask)):
      if start > 0:
        mask_conditions.append(f"({coord} >= {start})")
      if end < self.shape[i]: 
        mask_conditions.append(f"({coord} < {end})")
      if stride == 0: continue
      shifted_coord = f"({coord} - {start})" if start > 0 else coord
      if stride == 1: 
        terms.append(shifted_coord)
      else: 
        terms.append(f"{shifted_coord} * {stride}")
    index_math = " + ".join(terms) if terms else "0"
    mask_math = " && ".join(mask_conditions) if mask_conditions else "true"
    return index_math, mask_math
  def load_expr(self, buf_name:str, gid:str="gid") -> str:
    index_math, mask_math = self.index_expr(gid)
    if not self.is_padded and mask_math == "true": return f"{buf_name}[{index_math}]"
    from monograd.engine.codegen import cl_const
    assert self.padding_op is not None, f"we are padded but have no padding op?: {self}"
    formatted_pad = cl_const(self.padding_op.arg[2], self.uop.dtype)
    return f"({mask_math} ? {buf_name}[{index_math}] : {formatted_pad})"
  def reduce_load_expr_multi(self, reduce_axes:tuple[int, ...], buf_name:str, gid:str="gid", k:str="k") -> str:
    index_math, mask_math = self.reduce_index_expr_multi(reduce_axes, gid, k)
    if not self.is_padded and mask_math == "true": return f"{buf_name}[{index_math}]"
    from monograd.engine.codegen import cl_const
    assert self.padding_op is not None, f"we are padded but have no padding op?: {self}"
    formatted_pad = cl_const(self.padding_op.arg[2], self.uop.dtype)
    return f"({mask_math} ? {buf_name}[{index_math}] : {formatted_pad})"
  @property
  def is_padded(self): return not self.padding_op is None
  def __repr__(self):
    return f"BufferRef(op={self.uop.op}, shape={self.shape}, strides={self.strides})"

# **** main scheduler compute ****
def run_scheduler(root: UOp) -> list[KernelTask]:
  # lean pre-pass: Count UNIQUE consumers for each node
  consumers: dict[int, int] = defaultdict(int)
  def _count(n: UOp, visited: set[int]):
    if id(n) in visited: return
    visited.add(id(n))
    for src in set(n.src): 
      consumers[id(src)] += 1
      _count(src, visited)
  _count(root, set())
  scheduled_kernels: list[KernelTask] = []
  scheduled_nodes: set[int] = set()
  def _schedule(node: UOp):
    if id(node) in scheduled_nodes: return
    if is_invisible(node):
      for src in node.src: _schedule(src)
      scheduled_nodes.add(id(node))
      return
    def _pull(n: UOp):
      if id(n) in scheduled_nodes: return
      if n.op in GroupOp.Input | GroupOp.Movement:
        for src in n.src: _pull(src)
        return
      if is_fusable(n):
        if consumers[id(n)] > 1: 
          _schedule(n) # fanout detected! force to VRAM
        else:
          scheduled_nodes.add(id(n))
          for src in n.src: _pull(src)
          group_ops.append(n)
      elif is_boundary(n):
        _schedule(n)
    # force kernel creation if it's a boundary, root, or multi-consumer
    if is_boundary(node) or node is root or consumers[id(node)] > 1:
      group_ops: list[UOp] = []
      for src in node.src: _pull(src)
      group_ops.append(node)
      scheduled_nodes.add(id(node))
      if node.op in GroupOp.BLAS: kind = TaskKind.BLAS
      elif node.op in GroupOp.Reduce:
        kind = TaskKind.REDUCE_FULL if len(node.shape) == len(node.arg[0]) else TaskKind.REDUCE_STRIDED
      elif node.op in {Ops.COPY, Ops.CONTIGUOUS}: kind = TaskKind.COPY
      else: kind = TaskKind.ELEMENTWISE
      scheduled_kernels.append(KernelTask(kind, group_ops, _collect_inputs(group_ops)))
  _schedule(root)
  return scheduled_kernels
def _collect_inputs(ops:list[UOp]) -> list[BufferRef]:
  # given *ops*, will return leaf/input nodes as buffer refs
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

def pprint_schedule(tasks:list[KernelTask]) -> None:
  print(f"Schedule: {len(tasks)} kernel(s)")
  print("─" * 50)
  for i, task in enumerate(tasks):
    print(f"[{i}] {task}")
    print("─" * 50)
