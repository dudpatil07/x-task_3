# PAPER_NOTES.md

## Research Paper Analysis: Learning representations by back-propagating errors
**Authors:** David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams (1986)  
**Replication Focus:** Automated Discovery of Internal Representations in the XOR Architecture  

---

## 1. The Central Claim

The core problem this paper addresses is the historic computational barrier identified by Minsky and Papert (1969): single-layer perceptrons are fundamentally incapable of learning non-linearly separable functions because they map inputs directly to outputs without any intermediate processing layers. 

The central claim of Rumelhart, Hinton, and Williams is that a multi-layer feedforward network can autonomously discover and engineer its own internal, hidden features to solve these complex, non-linear mappings. The authors assert that this can be achieved efficiently using a gradient descent error-propagation algorithm, which they formalize as the Generalized Delta Rule. 

### Why It Works Better
Unlike previous attempts that relied on heuristic or randomized hidden layer adjustments, backpropagation systematically distributes credit or blame for an error across every single weight in the system. By calculating the partial derivative of the total error function with respect to each individual weight ($\partial E / \partial w$), the network knows precisely how to shift its internal parameters to minimize global loss. 

The paper specifically emphasizes that adding a momentum term ($\alpha$) allows the weight updates to accumulate velocity down a gradient. This prevents the training process from stalling in local minima or getting trapped on shallow plateaus, which is the primary reason it works practically where standard gradient descent falters.

---

## 2. Core Architecture and Algorithm

To evaluate the validity of this claim, we must implement the exact mathematical framework laid out by the authors in the "Generalized Delta Rule" section of the paper.

   [ Output Layer ] (a2)
         ^
         |  Weights (W2) & Biases (b2)
         |
   [ Hidden Layer ] (a1)
         ^
         |  Weights (W1) & Biases (b1)
         |
   [ Input Layer ]  (X)


### Architectural Layout
The minimum viable configuration to test non-linear mapping is a 3-layer architecture configured as follows:
* **Input Layer:** 2 units corresponding to the binary features of the XOR problem.
* **Hidden Layer:** 2 processing units. The paper proves that 2 hidden units are mathematically sufficient to develop the internal abstractions required for XOR.
* **Output Layer:** 1 unit generating a continuous probability value between 0.0 and 1.0.

### Mathematical Components & Design Decisions

#### 1. Activation Function
Every hidden and output unit passes its net input through a non-linear, continuously differentiable logistic sigmoid function. The paper explicitly dictates this choice because its derivative can be expressed cleanly using its output, making it highly efficient for backpropagation:

$$y = \frac{1}{1 + e^{-x}}$$

$$f'(x) = y(1 - y)$$

#### 2. Forward Propagation
For any given input vector $X$, the hidden layer activations ($a_1$) and output layer activations ($a_2$) are calculated sequentially:

$$z_1 = X \cdot W_1 + b_1 \quad \rightarrow \quad a_1 = \text{sigmoid}(z_1)$$

$$z_2 = a_1 \cdot W_2 + b_2 \quad \rightarrow \quad a_2 = \text{sigmoid}(z_2)$$

*Note on Implementation Fidelity:* Biases ($b_1, b_2$) are explicitly handled as independent translation vectors rather than hardcoded hacks, perfectly mirroring the paper's treatment of biases as weights tied to a constant input of $1.0$.

#### 3. Error Backpropagation (The Math to Implement)
The objective function to minimize is the sum of squared errors ($E$). For a single pattern $p$:

$$E = \frac{1}{2} \sum_{j} (t_{pj} - a_{2j})^2$$

During the backward pass, we calculate error signals ($\delta$) layer by layer, moving from the output backward:
* **Output Layer Delta:** Measures the distance to target scaled by the gradient of the activation function.
    $$\delta_{\text{output}} = (t - a_2) \odot a_2 \odot (1 - a_2)$$
* **Hidden Layer Delta:** Backpropagates the output delta through the transposed weight matrix $W_2$, scaled by the hidden layer's activation gradient.
    $$\delta_{\text{hidden}} = (\delta_{\text{output}} \cdot W_2^T) \odot a_1 \odot (1 - a_1)$$

#### 4. Weight Updates with Momentum
Gradients are calculated via the outer product of incoming activations and outgoing deltas. Crucially, the updates are applied using the paper's momentum-based velocity tracking to damp oscillations:

$$\Delta W(t+1) = \alpha \cdot \Delta W(t) + \epsilon \cdot (\text{inputs}^T \cdot \delta)$$

Where $\epsilon$ is the learning rate (set to $0.5$) and $\alpha$ is the momentum constant (set to $0.9$). Biases are updated identically using their corresponding $\delta$ vectors.

---

## 3. Dataset, Metrics, and Baselines

### The Dataset
The benchmark used to verify the architecture is the **Exclusive-OR (XOR)** logic gate. It consists of exactly 4 distinct training patterns:
* `[0, 0] -> [0]`
* `[0, 1] -> [1]`
* `[1, 0] -> [1]`
* `[1, 1] -> [0]`

This dataset is structurally critical because the targets are symmetrical but non-linearly separable. If you plot these four points on a 2D plane, it is geometrically impossible to draw a single straight line that isolates the `1` outputs from the `0` outputs.

### Evaluation Metrics
We track two precise metrics to gauge successful replication:
1.  **Total System Error ($E$):** The accumulated squared error across all 4 patterns. The target convergence threshold is set at $E < 0.005$, indicating that the model has learned the mapping with near-perfect confidence.
2.  **Convergence Velocity (Epochs):** The number of complete iterations through the dataset required to break through the error plateaus and reach the threshold.

### The Baseline Comparison
The baseline for this experiment is the **Single-Layer Perceptron (Rosenblatt)**. A single-layer perceptron running on this exact dataset yields a permanent, irreducible total system error of $E \approx 0.50$, completely failing to resolve the logic table. 

Our implementation will be deemed successful if the multi-layer network passes the baseline, bypasses the inherent local plateaus, and drives the global system error down to near zero ($< 0.005$), proving the automatic formulation of hidden layer representations.

---

## 4. Pre-Code Architectural Reflection

### Expectations & Sandbox Strategy
Based on the text, we should expect the training curve to show long periods of stagnation where the error stays stuck at a plateau around $0.50$. This happens because the network spends early epochs adjusting weights to find the intermediate feature mappings. 

To ensure absolute implementation honesty, our code will avoid using high-level automatic differentiation abstraction tools like PyTorch's `loss.backward()` or `optimizer.step()`. Instead, we are treating PyTorch strictly as a matrix operations library, defining all tracking states, velocity accumulations, and matrix deltas explicitly by hand in raw code to mirror exactly what the authors calculated on paper in 1986.
