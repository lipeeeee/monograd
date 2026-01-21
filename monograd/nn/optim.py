# Unfortunatly no good way to mytype this cuz of redundancy imports
import numpy as np

class Optimizer():
    def __init__(self, params, lr):
        self.params:list = list(params) # params can be Tensor or nn.Module?
        self.lr:float = lr
    
    def step(self):
        for tensor in self.params:
            tensor.data -= self.lr * tensor.grad.data

    def zero_grad(self):
        for tensor in self.params:
            tensor.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = []

        for p in self.params:
            self.velocities.append(np.zeros_like(p.data))

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            # v = (momentum * v) + grad
            self.velocities[i] = (self.momentum * self.velocities[i]) + p.grad.data
            # w = w - (lr * v)
            p.data -= self.lr * self.velocities[i]

class Adam(Optimizer):
    pass
