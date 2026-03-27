from __future__ import annotations
import functools
from monograd.utils import DEBUG 
from monograd.mixin import OpMixin
from monograd.mixin.movement import _align_left
from monograd.device import Device, DeviceLike, to_device
from monograd.dtype import ConstType, DType, DTypeLike, dtypes, from_np_dtype, most_upper_dtype, to_dtype
from monograd.uop import Ops
from monograd.uop.ops import UOp
import numpy as np

class Tensor(OpMixin):
  def __init__(self, data: ConstType|UOp|list|tuple|np.ndarray|None, requires_grad:bool = True,
               device:DeviceLike = Device.CPU, dtype:DTypeLike|None = None, name:str|None = None):
    if dtype is None and isinstance(data, np.ndarray): _dtype = from_np_dtype(data.dtype)
    else: _dtype = to_dtype(dtype) if dtype is not None else dtypes.default_float
    _device = to_device(device) if device is not Device.CPU else device
    del dtype, device
    self.grad:Tensor|None = None
    self.requires_grad:bool = requires_grad
    self.name:str|None = name

    # Create UOp from different types of inputs
    if isinstance(data, UOp):
      assert _dtype is None or _dtype == data.dtype, f"dtype mismatch: {_dtype} vs {data.dtype}"
    elif isinstance(data, ConstType):
      data = UOp(Ops.CONST, _dtype, (), (data, _device))
    elif isinstance(data, list|tuple|np.ndarray):
      buf = np.array(data)
      data = UOp(Ops.LOAD, _dtype, (), (buf.shape, _device))
      data.assign_buffer(buf.size, buf)

    # atp, data NEEDS to be a UOp
    assert isinstance(data, UOp), f"couldn't create Tensor from {data} with type {(type(data))}"
    self.uop:UOp = data

  @property
  def device(self) -> Device: return self.uop.device
  @property
  def dtype(self) -> DType: return self.uop.dtype
  @property
  def parents(self) -> tuple: return self.uop.src
  @property
  def shape(self) -> tuple: return self.uop.shape
  @property
  def ndim(self) -> int: return len(self.shape) # NOTE: maybe this goes in movement mixin (tinygrad does it)
  def _broadcasted(self, y:Tensor, reverse:bool=False) -> tuple[Tensor, Tensor]:
    # uses EXPAND and RESHAPE ops to broadcast 2 tensors
    target_shape, pad_x, pad_y = get_broadcasted_shape(self.shape, y.shape)
    if (x:=self).shape != pad_x: x = x.reshape(pad_x)
    if x.shape != target_shape: x = x.expand(target_shape)
    if y.shape != pad_y: y = y.reshape(pad_y)
    if y.shape != target_shape: y = y.expand(target_shape)
    return (y, x) if reverse else(x, y)

  def const_like(self, x:ConstType) -> Tensor: return Tensor(x, self.requires_grad, self.device, self.dtype)
  def _reduceop(self, op:Ops, axis:int|tuple[int, ...]|None=None, keepdim:bool=False) -> Tensor:
    if axis is None: resolved_axis = tuple(range(self.ndim))
    elif isinstance(axis, int): resolved_axis = (axis if axis >= 0 else axis + self.ndim,)
    elif isinstance(axis, tuple): resolved_axis = tuple(x if x >= 0 else x + self.ndim for x in axis)
    else: raise ValueError(f"unsupported axis: {axis}")
    if self.ndim == 0: resolved_axis = () # 0D scalars
    # compute reduced shape & create op
    reduced_shape = tuple(1 if i in resolved_axis else s for i, s in enumerate(self.shape))
    ret = Tensor.__new__(Tensor)
    ret.uop = UOp(op, self.dtype, (self.uop,), (resolved_axis, reduced_shape))
    ret.requires_grad = self.requires_grad
    # handle keepdim
    if not keepdim:
      final_shape = tuple(s for i, s in enumerate(self.shape) if i not in resolved_axis)
      return ret.reshape(final_shape if final_shape else (1,))
    return ret
  def _mop(self, op:Ops, arg) -> Tensor:
    ret = Tensor.__new__(Tensor)
    ret.uop = UOp(op, self.dtype, (self.uop,), arg)
    ret.requires_grad = self.requires_grad
    return ret
  def _unop(self, op:Ops) -> Tensor:
    ret = Tensor.__new__(Tensor)
    ret.uop = UOp(op, self.dtype, (self.uop,), self.device)
    ret.requires_grad = self.requires_grad
    return ret
  def _binop(self, op:Ops, x:Tensor, reverse:bool=False) -> Tensor:
    lhs, rhs = self._broadcasted(x, reverse)
    assert lhs.device == rhs.device, f"device {lhs.device} doesn't match {rhs.device}"
    target_dtype = most_upper_dtype(lhs.dtype, rhs.dtype)
    lhs, rhs = lhs.cast(target_dtype), rhs.cast(target_dtype)
    ret = Tensor.__new__(Tensor)
    ret.uop = UOp(op, lhs.dtype, (lhs.uop, rhs.uop), lhs.device) 
    ret.requires_grad = lhs.requires_grad or rhs.requires_grad
    return ret
  def _ternop(self, op:Ops, x:Tensor, y:Tensor, reverse:bool=False) -> Tensor:
    raise NotImplementedError("need 3-way broadcast functio")

  def matmul(self, x:Tensor) -> Tensor:
    assert self.shape[-1] == x.shape[-2], f"matmul shape mismatch: {self.shape} x {x.shape}"
    assert self.ndim == 2 and x.ndim == 2, "only 2D matmul for now" # NOTE: handle batched matmul later — for now assert 2D
    target_dtype = most_upper_dtype(self.dtype, x.dtype)
    self, x = self.cast(target_dtype), x.cast(target_dtype)
    ret = Tensor.__new__(Tensor)
    ret.uop = UOp(Ops.MATMUL, self.dtype, (self.uop, x.uop), self.device)
    ret.requires_grad = self.requires_grad or x.requires_grad
    return ret
  def cast(self, dtype:DTypeLike) -> Tensor:
    dtype = to_dtype(dtype)
    if self.dtype == dtype: return self # noop
    ret = Tensor.__new__(Tensor)
    ret.uop = self.uop.cast(dtype)
    ret.requires_grad = self.requires_grad
    return ret
  def cast_upwards(self, *tensors:Tensor) -> tuple[Tensor, ...]|Tensor:
    target = most_upper_dtype(*[t.dtype for t in tensors])
    return tuple(t.cast(target) for t in tensors) if isinstance(tensors, tuple) else tensors.cast(target)
  def to(self, device:DeviceLike) -> Tensor: # NOTE does this handle changing the device of grad???     !!!
    if (device:=to_device(device)) == self.device: return self
    copy_device_uop = UOp(Ops.COPY, self.dtype, (self.uop,), device)
    ret = Tensor(copy_device_uop, self.requires_grad, dtype=self.dtype)
    return ret

  @property
  def T(self) -> Tensor: return self.permute(tuple(range(self.ndim - 1, -1, -1))) # reverses all axis
  def __repr__(self):
    return f"<Tensor {self.uop} requires_grad={self.requires_grad}>"

@functools.cache
def get_broadcasted_shape(s1:tuple, s2:tuple) -> tuple[tuple, tuple, tuple]: # this can probably be re-done for to support *shapes
  if s1 == s2: return s1, s1, s2
  pad1, pad2 = _align_left(s1, s2)
  assert all(d1 == d2 or d1 == 1 or d2 == 1 for d1, d2 in zip(pad1, pad2)), f"cannot broadcast {s1} to {s2}"
  target_shape = tuple(max(d1, d2) for d1, d2 in zip(pad1, pad2))
  return target_shape, pad1, pad2
def pprint_graph(uop:Tensor|UOp, prefix:str="", is_last:bool=True, visited:set|None=None):
  if isinstance(uop, Tensor): uop = uop.uop
  if visited is None: visited = set()
  marker = "└── " if is_last else "├── "
  op_name = uop.op.name if hasattr(uop.op, "name") else str(uop.op)
  shape_str = f" {uop.shape}" if hasattr(uop, 'shape') else ""
  arg_str = f" arg={uop.arg}" if uop.arg is not None else ""
  # handle seens
  node_id = id(uop)
  if node_id in visited:
    print(f"{prefix}{marker}{op_name}{shape_str}{arg_str} [SEEN]")
    return
  visited.add(node_id)
  # current
  print(f"{prefix}{marker}{op_name}{shape_str}{arg_str}")
  # sources
  if hasattr(uop, 'src') and uop.src:
    next_prefix = prefix + ("    " if is_last else "│   ")
    for i, src_uop in enumerate(uop.src):
      is_last_src = (i == len(uop.src) - 1)
      pprint_graph(src_uop, next_prefix, is_last_src, visited)

if __name__ == "__main__":
  a = Tensor([[1, 2, 3], [4, 5, 6]], device="gpu", dtype="float64")
  b = Tensor([3, 2, 1], device="gpu")
  c = ((a * 2) + b) * 5
  pprint_graph(c)

  from monograd.engine.schedule import run_scheduler, pprint_schedule
  s = run_scheduler(c.uop)
  pprint_schedule(s)
  
  from monograd.engine.codegen import codegen
  [codegen(si) for si in s]