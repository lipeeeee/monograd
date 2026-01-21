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
from monograd.nn import Module, Linear
from monograd.optimizer import SGD

class MyNet(Module):
    def __init__(self):
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x

model = MyNet()
optim = SGD(model.parameters(), lr=0.01)
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

### Roadmap
- [ ] Conv2d (CPU)
- [ ] general refactoring with tinygrad-like file struct(nn, datasets) 
- [ ] GPU Support (CUDA/HIP)
- [ ] GPU kernel code gen (JIT)

### Running tests
```bash
chmod +x test-all.sh
./test-all.sh # will run all available tests
```

--- 

### Installation & Dependencies
Only dependency is numpy!
```
git clone https://github.com/yourusername/monograd.git
cd monograd
pip install numpy
```

# junk
py flags
```
PYTHONPATH=. DEBUG=TRUE
```

code notes
```
MEMIMPROVEMENT: note about possible memory improvement
PERFIMPROVEMENT: note about possible performance improvement
REMOVE: remove ltr
```

todo
- Tests
- Check for refactoring optimizations
- Adam optimizer
- Save/Load weigths & bias 
- HIP/CUDA kernels (requires huge refactoring)
- Conv2d; MaxPool2d layers
