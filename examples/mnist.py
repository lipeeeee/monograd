import numpy as np
from lib.mnist_loader import fetch_mnist
from monograd.tensor import Tensor
from monograd.nn import Module, Linear
from monograd.optimizer import SGD

# --- Loss fn() ---
def cross_entropy_loss(logits, targets):
    """
    logits: Tensor (Batch, 10)
    targets: Tensor (Batch,) - integers 0-9
    """
    batch_size = logits.shape[0]
    
    # A. Stable Softmax
    # Shift logits for stability (sub max) - detached from graph
    max_logits = Tensor(logits.data.max(axis=1, keepdims=True), requires_grad=False)
    shifted = logits - max_logits
    
    # Exponentiate
    exps = shifted.exp()
    sum_exps = exps.sum(axis=1).reshape((batch_size, 1))
    
    # LogSoftmax = shifted - log(sum_exp)
    log_probs = shifted - sum_exps.log()
    
    # B. NLL Loss (Negative Log Likelihood)
    # Create One-Hot Mask manually (since we lack advanced indexing ops)
    # targets.data must be integers
    y_true = np.zeros((batch_size, 10), dtype=np.float32)
    y_true[np.arange(batch_size), targets.data.astype(int)] = 1.0
    y_true = Tensor(y_true, requires_grad=False)
    
    # Select the correct class log-probabilities
    # Multiply by mask (0s kill wrong classes, 1 keeps correct class)
    picked_log_probs = (log_probs * y_true).sum(axis=1)
    
    # Average and Negate
    loss = -picked_log_probs.sum() * (1.0 / batch_size)
    return loss

# --- Model ---
class MNISTNet(Module):
    def __init__(self):
        # 784 inputs (28*28) -> 128 hidden -> 10 outputs (digits 0-9)
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)

    def forward(self, x):
        x = x.reshape((-1, 784))
        
        x = self.l1(x).relu()
        x = self.l2(x)

        return x

# --- Training ---
print("Loading Data...")
x_train, y_train, x_test, y_test = fetch_mnist()
print(f"Data Loaded: {x_train.shape}")

# Initialize
model = MNISTNet()
optim = SGD(model.parameters(), lr=0.01, momentum=0.9)

BATCH_SIZE = 64
STEPS = 10000

print("Starting Training...")
for step in range(STEPS):
    
    # A. Get Batch (Random sampling)
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
    if step % 100 == 0:
        preds = np.argmax(logits.data, axis=1)
        acc = (preds == batch_y.data).mean()
        print(f"Step {step:4d} | Loss: {loss.data:.4f} | Acc: {acc:.2%}")

# --- Final Evaluation ---
print("\nEvaluating on Test Set...")
test_x = Tensor(x_test, requires_grad=False)
test_logits = model(test_x)
test_preds = np.argmax(test_logits.data, axis=1)
test_acc = (test_preds == y_test).mean()

print(f"Final Test Accuracy: {test_acc:.2%}")

# sanity check
import random
idx = random.randint(0, 1000)
print(f"\nExample Prediction (Idx {idx}):")
print(f"Predicted: {test_preds[idx]}")
print(f"Actual:    {y_test[idx]}")
