import numpy as np
from monograd.tensor import Tensor
from monograd.nn import Module, Linear, Conv2d, MaxPool2d, optim
from monograd.nn.datasets import mnist
import matplotlib

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

def plot_prediction(model, x_test, y_test):
    import matplotlib.pyplot as plt

   # 1. Pick a random image
    idx = np.random.randint(0, len(x_test))
    img_tensor = Tensor(x_test[idx].reshape(1, 28, 28), requires_grad=False) # Add batch dim
    true_label = y_test[idx]
    
    # 2. Get Model Prediction
    logits = model(img_tensor)
    
    # Convert logits to probabilities (Softmax) using NumPy
    # exp(x) / sum(exp(x))
    exps = np.exp(logits.data.flatten())
    probs = exps / np.sum(exps)
    pred_label = np.argmax(probs)

    # 3. Plot
    plt.figure(figsize=(8, 4))

    # Subplot 1: The Image
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {true_label} | Pred: {pred_label}")
    plt.axis('off')

    # Subplot 2: The Probabilities
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(10), probs, color='gray')
    bars[pred_label].set_color('red')  # Highlight prediction
    bars[true_label].set_color('green') # Highlight truth (if different, you see both)
    
    plt.xlabel("Digit Class")
    plt.ylabel("Probability")
    plt.xticks(range(10))
    plt.ylim(0, 1.1)
    plt.title("Model Confidence")

    plt.tight_layout()
    plt.savefig('single_prediction.png')
    print(f"Saved prediction visualization to 'single_prediction.png' (Index {idx})")

# Call it
plot_prediction(model, x_test, y_test) 
import matplotlib.pyplot as plt
import numpy as np

def save_prediction_grid(model, x_test, y_test, filename='mnist_predictions_50.png'):
    print(f"Generating 50-image grid... saving to {filename}")
    
    # 1. Setup the figure grid
    rows, cols = 5, 10
    # Large figsize (18x10) ensures the text isn't cramped
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.6) # Add padding for titles
    axes = axes.flatten()

    # 2. Select 50 random indices
    indices = np.random.choice(len(x_test), rows * cols, replace=False)

    for i, idx in enumerate(indices):
        # Prepare data for model
        input_data = x_test[idx]
        if input_data.ndim == 2: 
            input_tensor_data = input_data[None, None, ...] 
        else: 
            input_tensor_data = input_data[None, ...]
            
        img_tensor = Tensor(input_tensor_data, requires_grad=False)
        true_label = y_test[idx]

        # Get prediction
        logits = model(img_tensor)
        pred_label = np.argmax(logits.data.flatten())

        # Plot onto the specific subplot axis
        ax = axes[i]
        ax.imshow(x_test[idx].squeeze(), cmap='gray')
        
        # Color coding: Green for correct, Red for wrong
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f"P:{pred_label} T:{true_label}", color=color, fontsize=10)
        ax.axis('off')

    # 3. Save and Close
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig) 
    print(f"Done! Successfully saved to {filename}")

# Execute
save_prediction_grid(model, x_test, y_test)
