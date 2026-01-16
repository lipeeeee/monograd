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
        # PERFIMPROVEMENT: do we need to have requires_grad=True in *result_tensor*
        from monograd.tensor import Context, Tensor

        # 1. create context
        ctx = Context()

        # 2. forward pass
        forward_args = [x.data if hasattr(x, "data") else x for x in args] # basically tensors
        result_data = cls.forward(ctx, *forward_args)
        result_tensor = Tensor(result_data, op=cls, parents=[x for x in args if hasattr(x, "op")]) # pyright: ignore cls is not recognized as OP

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
        x:np.ndarray = args[0]
        y:np.ndarray = args[1]
        ctx.save_for_backward(x, y)
        return x * y 

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_data
        return grad_output * y, grad_output * x 

class TRANSPOSE(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        order:list|tuple|None = args[1]
        ctx.save_for_backward(order)
        return np.transpose(x, order)
    
    @staticmethod
    def backward(ctx, grad_output):
        order, = ctx.saved_data
        return grad_output.transpose(order)

class MATMUL(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        y:np.ndarray = args[1]
        ctx.save_for_backward(x, y)
        return x @ y

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        x, y = ctx.saved_data
        
        x_t = Tensor(x, requires_grad=False)
        y_t = Tensor(y, requires_grad=False)
        
        # grad_x = grad_output @ y.T
        grad_x = grad_output.matmul(y_t.transpose()) # !
        # grad_y = x.T @ grad_output
        grad_y = x_t.transpose().matmul(grad_output)
        
        return grad_x, grad_y

class LOADOP(OP):
    @staticmethod
    def forward(ctx, *args):
        raise TypeError

    @staticmethod
    def backward(ctx, grad_output):
        raise TypeError

