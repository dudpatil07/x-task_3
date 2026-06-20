import torch

class ClassicMLP:
    def __init__(self, input_dim=2, hidden_dim=2, output_dim=1, lr=0.5, alpha=0.9):
        self.lr = lr
        self.alpha = alpha  # Momentum constant (α) from the paper
        
        # Initialize weights randomly between -0.5 and 0.5 as mentioned in the paper
        self.W1 = (torch.rand(input_dim, hidden_dim) - 0.5) * 0.5
        self.b1 = (torch.rand(1, hidden_dim) - 0.5) * 0.5
        self.W2 = (torch.rand(hidden_dim, output_dim) - 0.5) * 0.5
        self.b2 = (torch.rand(1, output_dim) - 0.5) * 0.5
        
        # Momentum velocity accumulators (initialized to 0)
        self.vW1, self.vb1 = torch.zeros_like(self.W1), torch.zeros_like(self.b1)
        self.vW2, self.vb2 = torch.zeros_like(self.W2), torch.zeros_like(self.b2)

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def sigmoid_derivative(self, out):
        # f'(x) = f(x) * (1 - f(x))
        return out * (1.0 - out)

    def forward(self, X):
        # Ensure input is 2D batch format [Batch_Size, Input_Dim]
        if X.dim() == 1:
            X = X.unsqueeze(0)
        self.X = X
        
        # Hidden layer activation
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer activation
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, y_true):
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)
            
        # 1. Output error signal: delta_p_j = (t_p_j - o_p_j) * o_p_j * (1 - o_p_j)
        error_output = y_true - self.a2
        delta_output = error_output * self.sigmoid_derivative(self.a2)
        
        # 2. Hidden error signal: delta_p_j = o_p_j * (1 - o_p_j) * sum(delta_p_k * w_k_j)
        error_hidden = torch.matmul(delta_output, self.W2.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.a1)
        
        # 3. Calculate weight changes (Gradients)
        dW2 = torch.matmul(self.a1.T, delta_output)
        db2 = torch.sum(delta_output, dim=0, keepdim=True) # Bias gradient
        
        dW1 = torch.matmul(self.X.T, delta_hidden)
        db1 = torch.sum(delta_hidden, dim=0, keepdim=True) # Bias gradient
        
        # 4. Apply Generalized Delta Rule weight updates with momentum: 
        # Δw(t+1) = ε * (delta * output) + α * Δw(t)
        self.vW2 = self.alpha * self.vW2 + self.lr * dW2
        self.W2 += self.vW2
        
        self.vb2 = self.alpha * self.vb2 + self.lr * db2
        self.b2 += self.vb2
        
        self.vW1 = self.alpha * self.vW1 + self.lr * dW1
        self.W1 += self.vW1
        
        self.vb1 = self.alpha * self.vb1 + self.lr * db1
        self.b1 += self.vb1