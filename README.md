<div align="center">
  <h1>monograd</h1>
  
  Something between [PyTorch](https://github.com/pytorch/pytorch) and [tinygrad](https://github.com/tinygrad/tinygrad)
</div>

---

### Monograd supports
- **Tensor library** and API
- Reverse-mode **autograd** differentiation with a dynamic graph (DAG)
- ~~JIT GPU Kernel compiler~~
- **nn / optim / datasets** for training

It is a lightweight, define-by-run deep learning framework built from scratch. Designed to be readable and hackable. It's a very small framework aimed at simplicity that can train real networks to high accuracy.

---

### Making a simple neural net in monograd:
```py
from monograd.tensor import Tensor
from monograd.nn import Module, Linear, optim

class MyNet(Module):
    def __init__(self):
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x

model = MyNet()
optim = optim.SGD(model.parameters(), lr=0.01)
# Training is the same as pytorch
```

---

### monograd vs Pytorch
```py
from monograd.tensor import Tensor

y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad) # dz/dx
print(y.grad) # dz/dy
```
Same thing but in PyTorch
```py
import torch

y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

---

Turns out you can build 95% of real world networks with **monograd**'s available framework

### Available Optimizers
- SGD(Stochastic Gradient Descent) with momentum
- Adam(Adaptive Momet Estimation)

### Available Network Layers
- Linear
- Conv2D
- MaxPool2D

### Available OPs 
- ADD
- SUM
- SUB
- MUL
- Transpose
- ReLu
- LeakyReLu
- Reshape
- LOG
- EXP

---

### Running tests
```bash
chmod +x test-all.sh
./test-all.sh # will run all available tests
```

### Run MNIST! (running on Conv2D and MaxPool2D)
```bash
PYTHONPATH=. python3 examples/mnist.py
```

---

### Roadmap
- [x] Conv2d (CPU)
- [x] MaxPool2D (CPU)
- [x] general refactoring with tinygrad-like file struct(nn, datasets) 
- [ ] ~~GPU Support (CUDA/HIP)~~
- [ ] ~~GPU kernel code gen (JIT)~~

--- 

### Installation & Dependencies
Only dependency is numpy!
```
git clone https://github.com/yourusername/monograd.git
cd monograd
pip install numpy
```
