# PAPER_NOTES.md

## Research Paper: Learning representations by back-propagating errors (1986)
**Done by:** Shubham Patil 
**Task:** Replicating the classic XOR experiment  

---

## 1. What is the central claim?

In this paper, Rumelhart, Hinton, and Williams are trying to solve a huge problem that older AI models had: single-layer perceptrons can't solve non-linear problems like XOR because they can only draw a single straight line to separate data points. 

I think the main claim here is that if you add a hidden layer between the input and output, the network can actually figure out its own internal features to solve these complex patterns automatically. To do this, they created the "Generalized Delta Rule" (which we call backpropagation today). It works way better because it calculates exactly how much every single weight in the network is responsible for the final error, and changes them in the right direction. 

Also, they claim that adding a momentum term ($\alpha$) is super important because it stops the network from getting stuck on flat plateaus or local minima where the error stops dropping.

---

## 2. Core Architecture and Algorithm

To test if their claim actually holds up, I am building a 3-layer feedforward network exactly like they described for the XOR problem:

* **Input Layer:** 2 units (for the binary inputs).
* **Hidden Layer:** 2 units (the authors proved 2 units is enough to learn the hidden representations).
* **Output Layer:** 1 unit (gives the final prediction between 0 and 1).

### The Math I am Implementing:
Instead of using PyTorch's automatic shortcuts like `loss.backward()`, I am coding the math formulas by hand to make sure the implementation is 100% honest to the 1986 paper.

1. **Forward Pass:** I pass the inputs through the first weights, add the bias, and use the sigmoid function: $y = 1 / (1 + e^{-x})$. Then I do the same from hidden to output.
2. **Output Delta:** I calculate the error at the end: $\delta_{\text{output}} = (t - a_2) \cdot a_2(1 - a_2)$.
3. **Hidden Delta:** I pass that error backward through the weights to find the hidden layer error signal.
4. **Weight & Bias Updates:** I update all weights AND biases using the momentum formula:

$$\Delta W(t+1) = \eta \cdot (\text{inputs}^T \cdot \delta) + \alpha \cdot \Delta W(t)$$

(Where $\eta$ is the learning rate, and $\alpha$ is the momentum constant). My code tracks this velocity over time just like the paper.

---

## 3. Dataset, Metrics, and Baseline

### The Dataset
I am using the standard 4-pattern XOR logic gate. The inputs are `[0,0]`, `[0,1]`, `[1,0]`, and `[1,1]`. It is hard for AI because the output is only `1` if the inputs are different.

### Metrics & Baseline
* **Baseline:** A single-layer network completely fails on XOR and gets stuck with a permanent high error of around $0.50$.
* **My Metrics:** I am measuring the **Total System Error** (sum of squared errors) and the number of **Epochs** it takes to converge. My target is to get the total error below $0.005$. 

### What to expect in the code
The paper notes that the network often gets stuck on a flat plateau for a long time before it suddenly figures out the hidden features and the error drops to zero. Also, depending on the random weights at the start, it might take 400 epochs or over 1000 epochs. My implementation will loop through the patterns one by one (online learning) to match their exact training flow.
