
#.data   A flat array (Buffer)   The physical numbers for the forward pass.
#.grad   A flat array (Buffer)   The physical numbers for the backward pass.
#.metadata   Shape, Strides, Offset  The "glasses" that tell you how to read both buffers.
#.op Add, Mul, Conv, etc.    Remembers how this tensor was created.
#.parents    List of Tensors Tells the Autograd where to send the gradients next.

from typing import List, Tuple
from monograd.utils import dbg, Device
import monograd.ops as Ops 
import numpy as np 

class Context(): # Tensor context
    saved_tensors:List # can save anything not just tensors

    def __init__(self):
        self.saved_tensors = []

    def save_for_backward(self, *args): # TODO: memory improvement: only save x.data when x is Tensor
        self.saved_tensors.extend(args)

# TODO: for tensor, impl magic methods for things like (1 + tensor)
# TODO: Lazy ops like tinygrad https://docs.tinygrad.org/quickstart/#tensors
# TODO: mytype everything
class Tensor():
    op:Ops.OP|None
    data:np.ndarray # for now we save it as ndarray but slow
    parents:Tuple|None
    device:Device
    requires_grad:bool
    parents:Tuple|None
    name:str|None
    ctx:Context|None

    def __init__(self, data:List|np.ndarray, op:Ops.OP|None=None, parents:Tuple|None = None, requires_grad:bool=False, _dtype=np.float32):
        self.name:str|None = None
        self.op = op

        # data & grad
        if isinstance(data, List) or isinstance(data, Tuple):
            self.data = np.array(data, dtype=_dtype)
        assert isinstance(self.data, np.ndarray)
        self.grad:Tensor|None = None
        self.requires_grad = requires_grad

        if not parents: self.parents = tuple()
        else: self.parents = tuple(parents)
        self.shape = self.data.shape
        self.strides = self.data.strides
        self.offset = 0

        self.device = Device.CPU
        self.ctx = None
