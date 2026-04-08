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
def is_fusable(uop:UOp) -> bool: return uop.op in GroupOp.Unary | GroupOp.Binary # NOTE: Not checking CAST because it is in Unary
def is_invisible(uop:UOp) -> bool: return uop.op in GroupOp.Movement
def is_boundary(uop:UOp) -> bool: return uop.op in GroupOp.BLAS | GroupOp.Reduce | {Ops.COPY}

class TaskKind(IntEnum):
  ELEMENTWISE = auto(); REDUCE = auto(); BLAS = auto(); COPY = auto()

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
    for coord, stride in zip(coords, self.strides):
      if stride == 0: continue # broadcast dim, skip
      elif stride == 1: terms.append(coord)
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
    for coord, stride in zip(input_coords, self.strides):
      if stride == 0: continue
      elif stride == 1: terms.append(coord)
      else: terms.append(f"{coord} * {stride}")
    return " + ".join(terms) if terms else "0"
  @property
  def is_padded(self): return hasattr(self, "padding_op") # TODO: test because name can change
  def __repr__(self):
    return f"BufferRef(op={self.uop.op}, shape={self.shape}, strides={self.strides})"

# **** main scheduler compute ****
def run_scheduler(root:UOp) -> list[KernelTask]:
  nodes = toposort(root, lambda u: u.src)
  current_group: list[UOp] = []
  scheduled_kernels: list[KernelTask] = []
  for node in nodes:
    if node.op in GroupOp.Input | GroupOp.Movement | {Ops.SINK}: continue # invisible/free ops
    elif is_fusable(node): current_group.append(node)
    elif is_boundary(node):
      _flush(TaskKind.ELEMENTWISE, current_group, scheduled_kernels)
      kind = TaskKind.BLAS if node.op in GroupOp.BLAS else TaskKind.REDUCE if node.op in GroupOp.Reduce else TaskKind.COPY if node.op is Ops.COPY else None
      assert kind is not None, "could not determine boundary node kind {node}"
      scheduled_kernels.append(KernelTask(kind, [node], _collect_inputs([node]))) # manual flush
  _flush(TaskKind.ELEMENTWISE, current_group, scheduled_kernels) # flush any remaining ops
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
def _flush(kind:TaskKind, current_group:list[UOp], scheduled_kernels:list[KernelTask]):
  if not current_group: return
  scheduled_kernels.append(KernelTask(kind, current_group.copy(), _collect_inputs(current_group)))
  current_group.clear()

def pprint_schedule(tasks:list[KernelTask]) -> None:
  print(f"Schedule: {len(tasks)} kernel(s)")
  print("─" * 50)
  for i, task in enumerate(tasks):
    print(f"[{i}] {task}")
    print("─" * 50)
