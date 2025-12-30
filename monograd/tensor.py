
#.data   A flat array (Buffer)   The physical numbers for the forward pass.
#.grad   A flat array (Buffer)   The physical numbers for the backward pass.
#.metadata   Shape, Strides, Offset  The "glasses" that tell you how to read both buffers.
#.op Add, Mul, Conv, etc.    Remembers how this tensor was created.
#.parents    List of Tensors Tells the Autograd where to send the gradients next.

from typing import Any, List, Tuple
from monograd.utils import dbg
import monograd.ops as Ops 

# TODO: for tensor, impl magic methods for things like (1 + tensor)
class Tensor():
    op:Ops.OP
    data:None
    metadata:Tuple
    parents:List|None

    def __init__(self, op:Ops.OP, data:Any, metadata:Any, parents:List|None):
        assert op

