import numpy as np
from monograd.utils import dbg


class OP():
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args):
        # TODO: Might need to assert / transfer args into Tensor!!!
        from monograd.tensor import Context, Tensor

        # 1. create context
        ctx = Context()

        # 2. forward pass
        result_data = cls.forward(ctx, *[x.data for x in args])
        result_tensor = Tensor(result_data, op=cls, parents=args) # pyright: ignore cls is not recognized as OP

        # 3. attatch
        result_tensor.ctx = ctx
        # dbg("saved", result_tensor.ctx.saved_data)
        # dbg("op", result_tensor.op)
        # dbg("data", result_tensor.data)
        return result_tensor

class ADD(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        y:np.ndarray = args[1]
        return x + y # numpy op

    @staticmethod
    def backward(ctx, grad_output):
        # dL/dx = grad_output * 1
        # dL/dy = grad_output * 1
        return grad_output, grad_output

class MUL(OP):
    @staticmethod
    def forward(ctx, *args):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

class LOADOP(OP):
    @staticmethod
    def forward(ctx, *args):
        raise TypeError

    @staticmethod
    def backward(ctx, grad_output):
        raise TypeError

