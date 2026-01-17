from typing import List, Tuple
from monograd.utils import dbg, Device
import monograd.ops as Ops
import numpy as np 

class Context(): # Tensor context
    saved_data:List

    def __init__(self):
        self.saved_data = []

    def save_for_backward(self, *args): # MEMIMPROVEMENT: only save x.data when x is Tensor :: its runtime 4 memory payoff
        self.saved_data.extend(args)

class Tensor():
    op:type|None
    data:np.ndarray # for now we save it as ndarray but slow
    parents:Tuple|None
    device:Device
    requires_grad:bool
    parents:Tuple|None
    name:str|None
    ctx:Context|None

    def __init__(self, data:List|np.ndarray|int|float, op:type|None=None, parents:Tuple|None = None, requires_grad:bool=True, _dtype=np.float32, name=None):
        self.name:str|None = name 
        self.op = op
        assert not self.op or issubclass(self.op, Ops.OP)

        # data & grad
        if isinstance(data, (List, Tuple, int, float, np.number)):
            self.data = np.array(data, dtype=_dtype)
        elif isinstance(data, np.ndarray): self.data = data
        elif isinstance(data, Tensor): dbg(f"[TENSOR] Got type {type(data)} as *data* in Tensor")
        assert isinstance(self.data, np.ndarray)
        self.grad:Tensor|None = None
        self.requires_grad = requires_grad

        if not parents: self.parents = tuple()
        else: self.parents = tuple(parents)
        self.shape = self.data.shape
        self.strides = self.data.strides
        self.offset = 0
        self._dtype = _dtype

        self.device = Device.CPU
        self.ctx = None

    def backward(self):
        graph: List[Tensor] = _toposort(self)
        if not self.grad: 
            self.grad = Tensor(np.ones(self.shape, dtype=self._dtype), requires_grad=False)

        for node in graph:
            # check if we can actually call op.backward
            if not node.op or isinstance(node.op, Ops.LOADOP):
                continue

            grads = node.op.backward(node.ctx, node.grad)
            if not isinstance(grads, (tuple, list)): grads = (grads,)

            if not node.parents:
                continue

            for parent, grad in zip(node.parents, grads):
                if not parent.requires_grad:
                    continue

                if not parent.grad: parent.grad = grad
                else: parent.grad += grad

    def relu(self):
        return Ops.RELU.apply(self)

    def leaky_relu(self, slope=0.01):
        return Ops.LEAKYRELU.apply(self, slope)

    def sum(self, axis=None):
        return Ops.SUM.apply(self, axis)

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Ops.MATMUL.apply(self, other)

    def transpose(self, order=None):
        return Ops.TRANSPOSE.apply(self, order)

    @property
    def T(self):
        return self.transpose()
      
    def toposort(self): # REMOVE: used rn for testing 
        return _toposort(self)

    def __neg__(self):
        return self * -1

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other + (-self)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Ops.ADD.apply(self, other)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Ops.SUB.apply(self, other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Ops.MUL.apply(self, other)
    
    def __matmul__(self, other):
        return self.matmul(other)

    def __repr__(self):
        return f"<Tensor name={self.name} op={self.op} data={self.data} device={self.device} requires_grad={self.requires_grad}>"

def _toposort(leaf:Tensor) -> List[Tensor]: # TODO: make tests
    # topological sort algo to order DAG
    visited:set = set()
    stack:List = []

    def dfs(t:Tensor):
        visited.add(t)
        parents = t.parents
        if parents:
            for parent in parents:
                if parent not in visited: dfs(parent)
        stack.append(t)

    dfs(leaf)
    return stack[::-1]

