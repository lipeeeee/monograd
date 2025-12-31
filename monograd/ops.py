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
    def __call__(cls, *args):
        # TODO: Might need to assert / transfer args into Tensor!!!
        from monograd.tensor import Context, Tensor

        # 1. create context
        ctx = Context()

        # 2. forward pass
        result_data = cls.forward(ctx, *[x.data for x in args])
        result_tensor = Tensor(result_data, op=cls, parents=args) # pyright: ignore cls is not recognized as OP

        # 3. attatch
        result_tensor.ctx = ctx
        dbg("saved", result_tensor.ctx.saved_tensors)
        dbg("op", result_tensor.op)
        dbg("data", result_tensor.data)
        return result_tensor

class ADD():
    @staticmethod
    def forward(ctx, *args):
        ctx.saved_tensors()
        pass
        return np.add(args)

    @staticmethod
    def backward(ctx, grad_output):
        pass

class MUL():
    @staticmethod
    def forward(ctx, *args):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

class LOADOP(OP):
    @staticmethod
    def forward(ctx, *args):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

