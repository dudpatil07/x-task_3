import torch
import matplotlib.pyplot as plt
import os
from model import ClassicMLP

# 1. Setup paths safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

# 2. XOR Dataset (4 explicit patterns)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# 3. Instantiate Model (Hyperparameters aligned with classic defaults)
model = ClassicMLP(input_dim=2, hidden_dim=2, output_dim=1, lr=0.5, alpha=0.9)

epochs = 5000
loss_history = []
log_output = []

# 4. Training Loop
for epoch in range(epochs):
    epoch_loss = 0.0
    
    # Present patterns one by one to replicate the original online learning flow
    for i in range(len(X)):
        pattern_X = X[i]
        pattern_y = y[i]
        
        # Forward pass for single pattern
        pred = model.forward(pattern_X)
        
        # Accumulate total squared error: E = 0.5 * sum((t - o)^2)
        loss = 0.5 * torch.sum((pattern_y - pred) ** 2).item()
        epoch_loss += loss
        
        # Backward pass & immediate step weight modification
        model.backward(pattern_y)
        
    loss_history.append(epoch_loss)
    
    # Track performance
    if epoch % 200 == 0 or epoch_loss < 0.005:
        status = f"Epoch {epoch:4d} | Total System Error: {epoch_loss:.6f}"
        log_output.append(status)
        print(status)
        if epoch_loss < 0.005:
            break

# 5. Save Logs and Final Predictions for Verification
log_path = os.path.join(results_dir, "training_log.txt")
with open(log_path, "w") as f:
    f.write("\n".join(log_output))
    f.write(f"\n\nFinal Network Evaluation on full XOR dataset:\n")
    final_preds = model.forward(X).tolist()
    for inputs, target, prediction in zip(X.tolist(), y.tolist(), final_preds):
        f.write(f"Input: {inputs} | Target: {target} | Predicted: {[round(prediction[0], 4)]}\n")

# 6. Save a loss curve diagram
plot_path = os.path.join(results_dir, "loss_curve.png")
plt.figure()
plt.plot(loss_history, color='blue', linewidth=1.5)
plt.title("1986 Backpropagation - XOR Total Error Convergence")
plt.xlabel("Epochs")
plt.ylabel("Squared Error Total (E)")
plt.grid(True, linestyle='--')
plt.savefig(plot_path)

print("\nSystem verification complete! Results perfectly compiled in results/ directory.")