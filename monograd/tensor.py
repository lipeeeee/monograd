from __future__ import annotations
from monograd.device import Buffer, Device, DeviceLike, to_device
from monograd.dtype import ConstType, DType, DTypeLike, dtypes, to_dtype
from monograd.uop import Ops
from monograd.uop.ops import UOp
import numpy as np 

class OpMixin: pass
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
      data = UOp(Ops.CONST, _dtype, (), data)
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
  def to(self, device:DeviceLike) -> Tensor:
    if (device:=to_device(device)) == self.device: return
    copy_device_uop = UOp(Ops.COPY, self.dtype, (self,), device)
    ret = Tensor(copy_device_uop, self.requires_grad, self.device, self.dtype)
    return ret
  def __repr__(self):
    return f"<Tensor {self.uop} requires_grad={self.requires_grad}>"

if __name__ == "__main__":
  # TODO: SHAPE ;  do we only need shape on load? for sure need shape on store
  a = Tensor([1, 2, 3, 4], device="gpu")
  a.uop.arg.allocate()
  t1 = memoryview(np.array([1,2,3,4], dtype=dtypes.int32.np_dtype))
  a.uop.arg.copyin(t1)
  t2 = memoryview(np.zeros(4, dtype=dtypes.int32.np_dtype))
  print(np.frombuffer(a.uop.arg.copyout(t2), dtype=dtypes.int32.np_dtype))
  print(a.uop.arg)
  # a = Tensor([1,2,3,4]).to("gpu").to("cpu").to("gpu")
  # for e, t in build_str_graph(a.uop):
  #   print("\t"*t, e)

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

