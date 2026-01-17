from monograd.nn import Linear, Module
from monograd.optimizer import SGD
from monograd.tensor import Tensor 

class XORNet(Module):
    def __init__(self):
        # XOR is not linearly separable, so we need hidden dimensions.
        # Input: 2 (x, y) -> Hidden: 4 neurons -> Output: 1 (score)
        self.fc1 = Linear(2, 4)
        self.fc2 = Linear(4, 1)

    def forward(self, x):
        # x -> Linear -> ReLU -> Linear -> Output
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

# --- 2. Create Dataset (XOR) ---
# Input: (4 samples, 2 features)
x_train = Tensor([[0.0, 0.0], 
                  [0.0, 1.0], 
                  [1.0, 0.0], 
                  [1.0, 1.0]], requires_grad=False)

# Target: (4 samples, 1 label)
# 0^0=0, 0^1=1, 1^0=1, 1^1=0
y_train = Tensor([[0.0], 
                  [1.0], 
                  [1.0], 
                  [0.0]], requires_grad=False)

# --- 3. Initialize Model & Optimizer ---
model = XORNet()
# We use a high Learning Rate (0.1) because XOR is simple but has sharp gradients
optim = SGD(model.parameters(), lr=0.1)

print("--- Starting Training ---")

# --- 4. Training Loop ---
for epoch in range(500):
    
    # A. Forward Pass
    pred = model(x_train)
    
    # B. Loss Calculation (MSE: Mean Squared Error)
    # We use (pred - y) * (pred - y) since we haven't built Pow() yet
    diff = pred - y_train
    loss = (diff * diff).sum()
    
    # C. Backward Pass
    optim.zero_grad()
    loss.backward()  
    optim.step()     
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.data:.4f}")

print("\n--- Final Predictions ---")
final_pred = model(x_train)
for i in range(4):
    input_vals = x_train.data[i]
    prediction = final_pred.data[i][0]
    target = y_train.data[i][0]
    print(f"Input: {input_vals} | Target: {target} | Prediction: {prediction:.4f}")
