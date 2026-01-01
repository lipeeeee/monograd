import numpy as np
from monograd import tensor, ops

def addition():
    x = np.random.rand(5)
    y = np.random.rand(5)
    z = np.random.rand(1)

    x_tensor = tensor.Tensor(x, op=ops.LOADOP)
    y_tensor = tensor.Tensor(y, op=ops.LOADOP)

    _ = x_tensor + y_tensor # same shape
    _ = x_tensor + z # scalar
    _ = x_tensor + 5 # python int

def multiplication():
    x = np.random.rand(5)
    y = np.random.rand(5)
    z = np.random.rand(1)

    x_tensor = tensor.Tensor(x, op=ops.LOADOP)
    y_tensor = tensor.Tensor(y, op=ops.LOADOP)

    i = x_tensor * y_tensor # same shape
    i = x_tensor * z # scalar
    i = x_tensor * 5 # python int

addition()
multiplication()
