from __future__ import annotations
import weakref
from math import prod
from enum import auto, IntEnum
from dataclasses import dataclass
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
  def index_expr(self, gid:str="gid") -> str: # generates C index expr
    if is_scalar(self.uop, self.strides): return "0"
    # build per-dim coordinate expressions from flat gid
    # e.g. for output_shape=(2,3):
    #   dim 0: coord = gid / 3
    #   dim 1: coord = gid % 3
    coords:list[str] = []
    remaining = gid
    for i, dim_size in enumerate(self.shape):
      below = prod(self.shape[i+1:])  # product of all dims below this one
      if below == 1: coords.append(remaining)
      else:
        coords.append(f"({remaining} / {below})")
        remaining = f"({remaining} % {below})"
    # build flat index from strides
    # skip dims where stride is 0 (broadcast — contributes nothing)
    terms:list[str] = []
    for i, (coord, stride) in enumerate(zip(coords, self.strides)):
      if stride == 0: continue # broadcast dim, skip
      if self.is_padded:
        if (valid_mask := self.padding_op.arg[1][i][0]) > 0: coord = f"({coord} - {valid_mask})" # type: ignore
      if stride == 1: terms.append(coord)
      else: terms.append(f"{coord} * {stride}")
    if not terms: return "0"
    return " + ".join(terms)
  def reduce_index_expr(self, reduce_axis:int, gid:str="gid") -> str:
    # get the shape of the output (gid domain) by removing the reduced axis
    output_shape = list(self.shape)
    output_shape.pop(reduce_axis)
    # unpack 'gid' based strictly on the OUTPUT shape
    gid_coords:list[str] = []
    remaining = gid
    for i in range(len(output_shape)):
      below = prod(output_shape[i+1:])
      if below == 1: 
        gid_coords.append(remaining)
      else:
        gid_coords.append(f"({remaining} / {below})")
        remaining = f"({remaining} % {below})"
    # re-insert the loop variable 'k' at the reduced axis position
    input_coords = list(gid_coords)
    input_coords.insert(reduce_axis, "k")
    # multiply by the INPUT strides to get the flat physical address
    terms: list[str] = []
    for i, (coord, stride) in enumerate(zip(input_coords, self.strides)):
      if stride == 0: continue
      if self.is_padded:
        if (valid_mask := self.padding_op.arg[1][i][0]) > 0: coord = f"({coord} - {valid_mask})" # type: ignore
      if stride == 1: terms.append(coord)
      else: terms.append(f"{coord} * {stride}")
    return " + ".join(terms) if terms else "0"
  def load_expr(self, buf_name:str, gid:str="gid") -> str:
    idx_str:str = self.index_expr(gid)
    if not self.is_padded: return f"{buf_name}[{idx_str}]"
    from monograd.engine.codegen import cl_const
    mask_str:str = self.mask_expr(gid)
    val:ConstType = self.padding_op.arg[2] # type: ignore
    pad_val_str:str = cl_const(val, self.uop.dtype)
    return f"({mask_str} ? {buf_name}[{idx_str}] : {pad_val_str})"
  def reduce_load_expr(self, reduce_axis:int, buf_name:str, gid:str="gid") -> str:
    idx_str = self.reduce_index_expr(reduce_axis, gid)
    if not self.is_padded: return f"{buf_name}[{idx_str}]"
    from monograd.engine.codegen import cl_const
    mask_str:str = self.reduce_mask_expr(reduce_axis, gid)
    val:ConstType = self.padding_op.arg[2] # type: ignore
    pad_val_str:str = cl_const(val, self.uop.dtype)
    return f"({mask_str} ? {buf_name}[{idx_str}] : {pad_val_str})"
  def mask_expr(self, gid:str="gid") -> str:
    if not self.is_padded: return "1"
    coords:list[str] = []
    remaining = gid
    for i in range(len(self.shape)):
      below = prod(self.shape[i+1:])
      if below == 1: coords.append(remaining)
      else:
        coords.append(f"({remaining} / {below})")
        remaining = f"({remaining} % {below})"
    conditions:list[str] = []
    valid_mask:tuple[tuple[int, ...]] = self.padding_op.arg[1] # type: ignore
    for i, (coord, (start, end)) in enumerate(zip(coords, valid_mask)):
      if start > 0 or end < self.shape[i]:
        conditions.append(f"({coord} >= {start} && {coord} < {end})")
    return " && ".join(conditions) if conditions else "1"
  def reduce_mask_expr(self, reduce_axis:int, gid:str="gid") -> str:
    if not self.is_padded: return "1"
    output_shape = list(self.shape)
    output_shape.pop(reduce_axis)
    gid_coords:list[str] = []
    remaining = gid
    for i in range(len(output_shape)):
      below = prod(output_shape[i+1:])
      if below == 1: gid_coords.append(remaining)
      else:
        gid_coords.append(f"({remaining} / {below})")
        remaining = f"({remaining} % {below})"
    # Insert loop variable 'k'
    input_coords = list(gid_coords)
    input_coords.insert(reduce_axis, "k")
    # Check boundaries against valid_mask
    conditions:list[str] = []
    valid_mask:tuple[tuple[int, ...]] = self.padding_op.arg[1] # type: ignore
    for i, (coord, (start, end)) in enumerate(zip(input_coords, valid_mask)):
      if start > 0 or end < self.shape[i]:
        conditions.append(f"({coord} >= {start} && {coord} < {end})")
    return " && ".join(conditions) if conditions else "1"
  @property
  def is_padded(self): return not self.padding_op is None
  def __repr__(self):
    return f"BufferRef(op={self.uop.op}, shape={self.shape}, strides={self.strides})"

# **** main scheduler compute ****
def run_scheduler(root: UOp) -> list[KernelTask]:
  # Go from the *root* uop and group srcs using *_pull* for each boundary
  scheduled_kernels:list[KernelTask] = []
  scheduled_nodes:set[int] = set()
  def _schedule(node: UOp):
    if id(node) in scheduled_nodes: return
    if is_invisible(node): # ignore invisible nodes but make new *root* of their children
      for src in node.src: _schedule(src)
      scheduled_nodes.add(id(node))
      return
    if is_boundary(node) or node is root: # initiating kernel
      group_ops: list[UOp] = []
      def _pull(n: UOp):
        # pull ALL ALU nodes into *group_ops* until boundary
        # when boundary found: recursively make boundary the next root for a new kernel 
        if id(n) in scheduled_nodes: return
        if n.op in GroupOp.Input | GroupOp.Movement:
          for src in n.src: _pull(src)
          return
        if is_fusable(n):
          scheduled_nodes.add(id(n))
          for src in n.src: _pull(src)
          group_ops.append(n)
        elif is_boundary(n):
          _schedule(n)
      if is_boundary(node):
        for src in node.src: _pull(src) # pull every ALU until boundaries
        group_ops.append(node) # add the boundary op at the very end
        scheduled_nodes.add(id(node))
        if node.op in GroupOp.BLAS: kind = TaskKind.BLAS
        elif node.op in GroupOp.Reduce:
          kind = TaskKind.REDUCE_FULL if len(node.shape) == len(node.arg[0]) else TaskKind.REDUCE_STRIDED
        elif node.op in {Ops.COPY, Ops.CONTIGUOUS}: kind = TaskKind.COPY
        else: raise RuntimeError(f"unknown boundary kind: {node}")
      else: # If the root is just an ALU op (c = a + b)
        _pull(node)
        kind = TaskKind.ELEMENTWISE
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
