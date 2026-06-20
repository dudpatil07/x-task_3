# Paper Notes: Learning representations by back-propagating errors (1986)

## 1. Central Claim
The authors assert that a multi-layer network using a gradient descent error-propagation algorithm (Generalized Delta Rule) can automatically construct internal, hidden layer representations to map non-linearly separable inputs to targets, solving limitations of single-layer perceptrons.

## 2. Core Architecture & Algorithm
- **Architecture:** 3-layer feedforward network (Input -> Hidden -> Output).
- **Activation:** Logistic Sigmoid function: $y = 1 / (1 + e^{-x})$.
- **Loss:** Sum of squared errors: $E = 0.5 * \sum (t - y)^2$.
- **Update Mechanism:** Backpropagation of error signals ($\delta$) combined with a momentum constant ($\alpha$) to avoid local minima.

## 3. Dataset & Metrics
- **Dataset:** The 4-state XOR (Exclusive-OR) logic function.
- **Metrics:** Total error convergence ($E < 0.01$) and number of training epochs required to learn the mapping successfully.