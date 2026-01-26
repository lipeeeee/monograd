from typing import Any
import numpy as np
from monograd.utils import dbg, im2col_indices, col2im_indices

def unbroadcast(grad:np.ndarray, original_shape:tuple) -> np.ndarray:
    # Collapse leading dimensions (e.g. (32, 10) -> (10,))
    while grad.ndim > len(original_shape):
        grad = np.sum(grad, axis=0)

    # Collapse broadcasted dimensions (e.g. (4, 4) -> (4, 1) -> (4,))
    for i, dim in enumerate(original_shape):
        if dim == 1:
            grad = np.sum(grad, axis=i, keepdims=True)
    return grad

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
        return result_tensor

class ADD(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        y:np.ndarray = args[1]
        ctx.save_for_backward(x.shape, y.shape)
        return x + y # numpy op

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        x_shape, y_shape = ctx.saved_data
        grad_x = grad_output
        grad_y = grad_output

        # unbroadcast nparrays
        if x_shape != grad_x.shape:
            grad_x = Tensor(unbroadcast(grad_x.data, x_shape), requires_grad=False)
        if y_shape != grad_y.shape:
            grad_y = Tensor(unbroadcast(grad_y.data, y_shape), requires_grad=False)

        return grad_x, grad_y 

class SUB(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        y:np.ndarray = args[1]
        ctx.save_for_backward(x.shape, y.shape)
        return x - y # numpy op

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        x_shape, y_shape = ctx.saved_data
        grad_x = grad_output
        grad_y = -grad_output

        # unbroadcast nparrays
        if x_shape != grad_x.shape:
            grad_x = Tensor(unbroadcast(grad_x.data, x_shape), requires_grad=False)
        if y_shape != grad_y.shape:
            grad_y = Tensor(unbroadcast(grad_y.data, y_shape), requires_grad=False)

        return grad_x, grad_y 

class MUL(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        y:np.ndarray = args[1]
        ctx.save_for_backward(x, y)
        return x * y  # numpy op

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        x, y = ctx.saved_data
        grad_x = grad_output * y
        grad_y = grad_output * x

        # unbroadcast nparrays
        if x.shape != grad_x.shape:
            grad_x = Tensor(unbroadcast(grad_x.data, x.shape), requires_grad=False)
        if y.shape != grad_y.shape:
            grad_y = Tensor(unbroadcast(grad_y.data, y.shape), requires_grad=False)

        return grad_x, grad_y 

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
        if order is None:
            return grad_output.transpose()
        else:
            return grad_output.transpose(np.argsort(order))

class MATMUL(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        y:np.ndarray = args[1]
        ctx.save_for_backward(x, y)
        return x @ y # numpy matmul

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

class RELU(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        ctx.save_for_backward(x)
        return np.maximum(x, 0)

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        x, = ctx.saved_data
        a = grad_output.data * (x > 0)
        return Tensor(a)

class LEAKYRELU(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        slope:float = args[1]
        ctx.save_for_backward(x, slope)
        return np.maximum(x, x * slope)

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        x, slope = ctx.saved_data
        grad = grad_output.data.copy()
        grad[x <= 0] *= slope
        return Tensor(grad)

class SUM(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        axis:int|None = args[1]
        ctx.save_for_backward(x.shape, axis)
        return np.sum(x, axis=axis)

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        shape, axis = ctx.saved_data

        grad = grad_output.data
        if axis is not None:
            grad = np.expand_dims(grad, axis)

        return Tensor(np.ones(shape) * grad)

class RESHAPE(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        shape:tuple = args[1]
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        original_shape, = ctx.saved_data
        return Tensor(grad_output.data.reshape(original_shape), requires_grad=False)

class EXP(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        result = np.exp(x)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        result, = ctx.saved_data
        return Tensor(grad_output.data * result, requires_grad=False)

class LOG(OP):
    @staticmethod
    def forward(ctx, *args):
        x:np.ndarray = args[0]
        ctx.save_for_backward(x)
        return np.log(x) 

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        x, = ctx.saved_data
        return Tensor(grad_output.data / x, requires_grad=False)

class CONV2D(OP):
    @staticmethod
    def forward(ctx, *args):
        """
        x: (N, C_in, H, W)
        w: (C_out, C_in, KH, KW)
        """
        x = args[0]
        w = args[1]
        stride = args[2]
        padding = args[3]

        N, C, H, W = x.shape
        F, C, HH, WW = w.shape # F = Filters (C_out)
        
        # 1. Math for output dimensions
        H_out = (H + 2 * padding - HH) // stride + 1
        W_out = (W + 2 * padding - WW) // stride + 1
        
        # 2. Im2Col (Turn image into a matrix)
        # Shape: (C_in * KH * KW, N * H_out * W_out)
        cols = im2col_indices(x, HH, WW, padding, stride)
        
        # 3. Flatten Weights
        # Shape: (F, C_in * KH * KW)
        w_col = w.reshape(F, -1)
        
        # 4. Matrix Multiplication (The actual convolution)
        # Shape: (F, N * H_out * W_out)
        out = w_col @ cols
        
        # 5. Reshape back to (N, F, H_out, W_out)
        out = out.reshape(F, H_out, W_out, N)
        out = out.transpose(3, 0, 1, 2)
        
        ctx.save_for_backward(x, cols, w, stride, padding)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        x, cols, w, stride, padding = ctx.saved_data
        
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        
        # Reshape Gradient for Matrix Mult
        # (N, F, H_out, W_out) -> (F, N * H_out * W_out)
        grad_reshaped = grad_output.data.transpose(1, 2, 3, 0).reshape(F, -1)
        
        # Compute Weight Gradient
        # dW = grad * cols.T
        grad_w_col = grad_reshaped @ cols.T
        grad_w = grad_w_col.reshape(w.shape)
        
        # Compute Input Gradient (This needs Col2Im)
        # dX = W.T * grad
        w_reshape = w.reshape(F, -1)
        grad_cols = w_reshape.T @ grad_reshaped
        
        # Turn columns back into image
        grad_x = col2im_indices(grad_cols, x.shape, HH, WW, padding, stride)
        
        return Tensor(grad_x), Tensor(grad_w)

class MAXPOOL2D(OP):
    @staticmethod
    def forward(ctx, *args):
        x = args[0]
        kernel_size = args[1]
        stride = args[2]
        padding = args[3]

        # Default stride to kernel_size (standard for pooling)
        if stride is None: stride = kernel_size
            
        N, C, H, W = x.shape
        H_out = (H + 2 * padding - kernel_size) // stride + 1
        W_out = (W + 2 * padding - kernel_size) // stride + 1

        # shape: (C * kernel_size * kernel_size, N * H_out * W_out)
        cols = im2col_indices(x, kernel_size, kernel_size, padding, stride)
        
        # New shape: (kernel_size * kernel_size, C, N * H_out * W_out)
        cols_reshaped = cols.reshape(C, kernel_size * kernel_size, -1)
        
        out = np.max(cols_reshaped, axis=1)
        
        # Argmax gives the index of the max value (0 to k*k-1)
        argmax = np.argmax(cols_reshaped, axis=1)
        
        # Reshape Output
        out = out.reshape(C, H_out, W_out, N)
        out = out.transpose(3, 0, 1, 2)

        ctx.save_for_backward(x.shape, argmax, kernel_size, stride, padding)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from monograd.tensor import Tensor
        x_shape, argmax, kernel_size, stride, padding = ctx.saved_data
        N, C, H, W = x_shape
        
        # Flatten gradient to match the column shape
        grad = grad_output.data.transpose(1, 2, 3, 0).reshape(C, -1)
        
        # Create a zero matrix for the columns
        cols_grad = np.zeros((C, kernel_size * kernel_size, grad.shape[1]))
        
        # Route the gradient ONLY to the max indices
        rows = np.arange(C)[:, None]
        cols_indices = np.arange(grad.shape[1])[None, :]
        cols_grad[rows, argmax, cols_indices] = grad
        
        # Flatten back to im2col shape
        cols_grad = cols_grad.reshape(C * kernel_size * kernel_size, -1)
        
        # Col2Im
        grad_x = col2im_indices(cols_grad, x_shape, kernel_size, kernel_size, padding, stride)
        
        return Tensor(grad_x)

class LOADOP(OP):
    @staticmethod
    def forward(ctx, *args):
        raise TypeError

    @staticmethod
    def backward(ctx, grad_output):
        raise TypeError

