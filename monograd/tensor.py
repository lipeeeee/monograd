from __future__ import annotations
from monograd.mixin import OpMixin
from monograd.mixin.movement import _align_left
from monograd.device import Buffer, Device, DeviceLike, to_device
from monograd.dtype import ConstType, DType, DTypeLike, dtypes, to_dtype
from monograd.uop import Ops
from monograd.uop.ops import UOp
import numpy as np 

class Tensor(OpMixin):
  def __init__(self, data: ConstType|UOp|list|tuple|np.ndarray|None, requires_grad:bool = True,
               device:DeviceLike = Device.CPU, dtype:DTypeLike|None = None, name:str|None = None):
    _dtype = to_dtype(dtype) if dtype is not None else dtypes.default_int
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
      data = UOp(Ops.LOAD, _dtype, (), buf.shape)
      data.assign_buffer(_device, len(buf), buf)
    # elif isinstance(data, np.ndarray):
    #   pass

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
    return (y,x) if reverse else(x, y)

  def const_like(self, x:ConstType) -> Tensor: return Tensor(x, self.requires_grad, self.device, self.dtype)
  def _mop(self, op:Ops, arg) -> Tensor:
    raise NotImplementedError             ################ DO THIS
  def _unop(self, op:Ops) -> Tensor:
    ret = Tensor.__new__(Tensor)
    ret.uop = UOp(op, self.dtype, (self.uop,), self.device)
    return ret
  def _binop(self, op:Ops, x:Tensor, reverse:bool=False) -> Tensor:
    lhs, rhs = self._broadcasted(x, reverse)
    assert lhs.dtype == rhs.dtype, f"type {lhs.dtype} doesn't match {rhs.dtype}"
    assert lhs.device == rhs.device, f"device {lhs.device} doesn't match {rhs.device}"
    ret = Tensor.__new__(Tensor)
    ret.uop = UOp(op, lhs.dtype, (lhs.uop, rhs.uop), lhs.device) 
    ret.requires_grad = True if lhs.requires_grad or rhs.requires_grad else False
    return ret
  def _ternop(self, op:Ops, x:Tensor, y:Tensor, reverse:bool=False) -> Tensor:
    raise NotImplementedError("need 3-way broadcast functio")
  def to(self, device:DeviceLike) -> Tensor: # NOTE does this handle changing the device of grad???     !!!
    if (device:=to_device(device)) == self.device: return self
    copy_device_uop = UOp(Ops.COPY, self.dtype, (self.uop,), device)
    ret = Tensor(copy_device_uop, self.requires_grad, self.device, self.dtype)
    return ret
  def __repr__(self):
    return f"<Tensor {self.uop} requires_grad={self.requires_grad}>"

def get_broadcasted_shape(s1:tuple, s2:tuple) -> tuple[tuple, tuple, tuple]: # this can probably be re-done for to support *shapes
  if s1 == s2: return s1, s1, s2
  pad1, pad2 = _align_left(s1, s2)
  assert all(d1 == d2 or d1 == 1 or d2 == 1 for d1, d2 in zip(pad1, pad2)), f"cannot broadcast {s1} to {s2}"
  target_shape = tuple(max(d1, d2) for d1, d2 in zip(pad1, pad2))
  return target_shape, pad1, pad2

if __name__ == "__main__":
  a = Tensor(1)
  b = a + 1
  print(b.parents)
  # a = Tensor([1, 2, 3, 4], device="gpu")
  # t1 = memoryview(np.array([1,2,3,4], dtype=dtypes.int32.np_dtype))
  # a.uop.buffer.copyin(t1)
  # t2 = memoryview(np.zeros(4, dtype=dtypes.int32.np_dtype))
  # print(np.frombuffer(a.uop.buffer.copyout(t2), dtype=dtypes.int32.np_dtype))
  # print(a.uop)
  # a = Tensor([1,2,3,4]).to("gpu").to("cpu").to("gpu")

# def build_str_graph(root:UOp):
#   visited:set = set()
#   stack:list[tuple] = []
#   def dfs(u:UOp, tab=0):
#     visited.add(u)
#     parents = u.src if hasattr(u, "src") else False 
#     if parents:
#       for parent in parents:
#         if parent not in visited: dfs(parent, tab+1)
#     stack.append((u, tab))
#   dfs(root)
#   return stack[::-1]

# class Context(): # Tensor context
#   saved_data:List

#   def __init__(self):
#     self.saved_data = []

#   def save_for_backward(self, *args): # MEMIMPROVEMENT: only save x.data when x is Tensor :: its runtime 4 memory payoff
#     self.saved_data.extend(args)

# class Tensor():
#   op:type|None
#   data:np.ndarray # for now we save it as ndarray but slow
#   parents:Tuple|None
#   device:Device
#   requires_grad:bool
#   parents:Tuple|None
#   name:str|None
#   ctx:Context|None

#   def __init__(self, data:List|np.ndarray|int|float, op:type|None=None, parents:Tuple|None = None, requires_grad:bool=True, _dtype=np.float32, name=None):
#     self.name:str|None = name 
#     self.op = op
#     assert not self.op or issubclass(self.op, Ops.OP)

#     # data & grad
#     if isinstance(data, (List, Tuple, int, float, np.number)):
#         self.data = np.array(data, dtype=_dtype)
#     elif isinstance(data, np.ndarray): self.data = data
#     elif isinstance(data, Tensor): dbg(f"[TENSOR] Got type {type(data)} as *data* in Tensor")
#     assert isinstance(self.data, np.ndarray)
#     self.grad:Tensor|None = None
#     self.requires_grad = requires_grad

#     if not parents: self.parents = tuple()
#     else: self.parents = tuple(parents)
#     self.shape = self.data.shape
#     self.strides = self.data.strides
#     self.offset = 0
#     self._dtype = _dtype

#     self.device = Device.CPU
#     self.ctx = None

#   def backward(self):
#     graph: List[Tensor] = _toposort(self)
#     if not self.grad: 
#       self.grad = Tensor(np.ones(self.shape, dtype=self._dtype), requires_grad=False)

#     for node in graph:
#       # check if we can actually call op.backward
#       if not node.op or isinstance(node.op, Ops.LOADOP):
#         continue

#       grads = node.op.backward(node.ctx, node.grad)
#       if not isinstance(grads, (tuple, list)): grads = (grads,)

#       if not node.parents:
#         continue

#       for parent, grad in zip(node.parents, grads):
#         if not parent.requires_grad:
#           continue

#         if not parent.grad: parent.grad = grad
#         else: parent.grad += grad

#   def relu(self):
#     return Ops.RELU.apply(self)

#   def leaky_relu(self, slope=0.01):
#     return Ops.LEAKYRELU.apply(self, slope)

#   def sum(self, axis=None):
#     return Ops.SUM.apply(self, axis)

#   def matmul(self, other):
#     if not isinstance(other, Tensor):
#       other = Tensor(other)
#     return Ops.MATMUL.apply(self, other)

#   def transpose(self, order=None):
#     return Ops.TRANSPOSE.apply(self, order)

#   def reshape(self, shape):
#     return Ops.RESHAPE.apply(self, shape)

#   def exp(self):
#     return Ops.EXP.apply(self)

#   def log(self):
#     return Ops.LOG.apply(self)

#   @property
#   def T(self):
#     return self.transpose()
    
#   def toposort(self): # REMOVE: used rn for testing 
#     return _toposort(self)

#   def __neg__(self):
#     return self * -1

#   def __rsub__(self, other):
#     if not isinstance(other, Tensor):
#       other = Tensor(other)
#     return other + (-self)

#   def __add__(self, other):
#     if not isinstance(other, Tensor):
#       other = Tensor(other)
#     return Ops.ADD.apply(self, other)

#   def __sub__(self, other):
#     if not isinstance(other, Tensor):
#       other = Tensor(other)
#     return Ops.SUB.apply(self, other)

#   def __mul__(self, other):
#     if not isinstance(other, Tensor):
#       other = Tensor(other)
#     return Ops.MUL.apply(self, other)
    
#   def __matmul__(self, other):
#     return self.matmul(other)

#   def __repr__(self):
#     return f"<Tensor name={self.name} op={self.op} data={self.data} device={self.device} requires_grad={self.requires_grad}>"

# def _toposort(leaf:Tensor) -> List[Tensor]: # TODO: make tests
#   # topological sort algo to order DAG
#   visited:set = set()
#   stack:List = []

#   def dfs(t:Tensor):
#     visited.add(t)
#     parents = t.parents
#     if parents:
#       for parent in parents:
#         if parent not in visited: dfs(parent)
#     stack.append(t)

#   dfs(leaf)
#   return stack[::-1]

