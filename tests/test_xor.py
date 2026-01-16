import numpy as np
from monograd.tensor import Tensor

# 1. Data
x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=False) # XOR input
y = Tensor([[0], [1], [1], [0]], requires_grad=False)             # XOR target

# 2. Weights (Random init)
w1 = Tensor(np.random.randn(2, 4), requires_grad=True)
b1 = Tensor(np.zeros((4,)), requires_grad=True)
w2 = Tensor(np.random.randn(4, 1), requires_grad=True)
b2 = Tensor(np.zeros((1,)), requires_grad=True)

# 3. Training Loop
for i in range(10):
    # Forward
    layer1 = (x @ w1 + b1).relu()
    pred = layer1 @ w2 + b2
    
    # Loss (MSE)
    diff = pred - y
    loss = (diff * diff).sum()
    
    # Backward
    loss.backward()
    
    # Update (SGD)
    # Note: We must wrap this in "no_grad" typically, but for now just subtract data
    w1.data -= 0.01 * w1.grad.data
    b1.data -= 0.01 * b1.grad.data
    w2.data -= 0.01 * w2.grad.data
    b2.data -= 0.01 * b2.grad.data
    
    # Zero Grads (Reset buckets!)
    w1.grad, b1.grad, w2.grad, b2.grad = None, None, None, None
    
    print(f"Step {i}, Loss: {loss.data}")
