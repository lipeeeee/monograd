import numpy as np
from monograd.tensor import Tensor
from monograd.nn import Module, Linear, Conv2d, MaxPool2d, optim
from monograd.nn.datasets import mnist

# --- Loss fn (Same as before) ---
def cross_entropy_loss(logits, targets):
    batch_size = logits.shape[0]
    max_logits = Tensor(logits.data.max(axis=1, keepdims=True), requires_grad=False)
    shifted = logits - max_logits
    exps = shifted.exp()
    sum_exps = exps.sum(axis=1).reshape((batch_size, 1))
    log_probs = shifted - sum_exps.log()
    y_true = np.zeros((batch_size, 10), dtype=np.float32)
    y_true[np.arange(batch_size), targets.data.astype(int)] = 1.0
    y_true = Tensor(y_true, requires_grad=False)
    picked_log_probs = (log_probs * y_true).sum(axis=1)
    loss = -picked_log_probs.sum() * (1.0 / batch_size)
    return loss

# --- CNN Model ---
class MNISTConvNet(Module):
    def __init__(self):
        # Input: 28x28 (1 channel)
        
        # Layer 1: Conv (1 -> 4 filters) + Pool
        # 28x28 -> 28x28 -> 14x14
        self.c1 = Conv2d(1, 4, kernel_size=3, padding=1)
        self.p1 = MaxPool2d(2) # Stride 2 by default
        
        # Layer 2: Conv (4 -> 8 filters) + Pool
        # 14x14 -> 14x14 -> 7x7
        self.c2 = Conv2d(4, 8, kernel_size=3, padding=1)
        self.p2 = MaxPool2d(2)
        
        # Layer 3: Linear
        # Input features: 8 channels * 7 * 7 pixels = 392
        self.l1 = Linear(8 * 7 * 7, 10)

    def forward(self, x):
        # x input is (Batch, 28, 28). We need (Batch, 1, 28, 28) for Conv2d
        x = x.reshape((-1, 1, 28, 28))
        
        x = self.c1(x).relu()
        x = self.p1(x)
        
        x = self.c2(x).relu()
        x = self.p2(x)
        
        # Flatten: (Batch, 8, 7, 7) -> (Batch, 392)
        x = x.reshape((-1, 8 * 7 * 7))
        x = self.l1(x)
        return x

# --- Training Setup ---
print("Loading Data...")
x_train, y_train, x_test, y_test = mnist()
print(f"Data Loaded: {x_train.shape}")

model = MNISTConvNet()
optim = optim.SGD(model.parameters(), lr=0.1, momentum=0.85)

BATCH_SIZE = 32
STEPS = 500

for step in range(STEPS):
    # A. Get Batch
    samp = np.random.randint(0, x_train.shape[0], size=(BATCH_SIZE))
    batch_x = Tensor(x_train[samp], requires_grad=False)
    batch_y = Tensor(y_train[samp], requires_grad=False)

    # B. Forward
    logits = model(batch_x)
    loss = cross_entropy_loss(logits, batch_y)

    # C. Backward
    optim.zero_grad()
    loss.backward()
    optim.step()

    # D. Logging
    if step % 20 == 0:
        preds = np.argmax(logits.data, axis=1)
        acc = (preds == batch_y.data).mean()
        print(f"Step {step:3d} | Loss: {loss.data:.4f} | Acc: {acc:.2%}")

# --- Eval ---
print("\nEvaluating on Test Set (Subset)...")
test_x = Tensor(x_test[:1000], requires_grad=False)
test_logits = model(test_x)
test_preds = np.argmax(test_logits.data, axis=1)
test_acc = (test_preds == y_test[:1000]).mean()

print(f"Test Accuracy: {test_acc:.2%}")
