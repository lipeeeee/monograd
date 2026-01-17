# Unfortunatly no good way to mytype this cuz of redundancy imports

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
    pass

class Adam(Optimizer):
    pass
