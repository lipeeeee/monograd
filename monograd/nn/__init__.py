import inspect
import numpy as np
from monograd.tensor import Tensor
import monograd.ops as Ops 

class Module():
    def parameters(self) -> list:
        params:list = []
        for _, attr in inspect.getmembers(self):
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())

        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs) # pyright: ignore dont know .forward exists

class Linear(Module):
    def __init__(self, in_features:int, out_features:int, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        # Random Gaussian weight initialization
        std_dev = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * std_dev,
            requires_grad=True,
            name="Linear Weight"
        )

        if bias:
            self.bias = Tensor(np.zeros((out_features,)), requires_grad=True, name="Linear Bias")
        else:
            self.bias = None
    
    def forward(self, x:Tensor):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        # Kaiming He Initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale,
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        out = Ops.CONV2D.apply(x, self.weight, self.stride, self.padding)
        out = out + self.bias.reshape((1, -1, 1, 1))
        
        return out

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return Ops.MAXPOOL2D.apply(x, self.kernel_size, self.stride, self.padding)
