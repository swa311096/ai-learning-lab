# 05: How Neural Networks Learn

Before sequence models make sense, one concept needs to be in place: how a neural network actually learns from data.

This chapter covers three things:

1. what a neural network is
2. what gradient descent is
3. what backpropagation is

These are not optional background. Every model from this point forward — RNNs, LSTMs, transformers — learns using these exact mechanisms.

---

### What a Neural Network Is

A neural network is a function that maps an input to an output by passing data through layers of connected nodes.

Each node takes in some numbers, multiplies them by weights, adds them up, and passes the result to the next layer.

![A neural network with 4 inputs, 2 hidden layers, and 1 output](assets/neural-network.png)

This network has:

- **Input layer**: 4 nodes (a, b, c, d) — each receives one number from the input
- **Hidden layer 1**: 3 nodes — each takes all 4 inputs, multiplies each by a weight, sums them up
- **Hidden layer 2**: 3 nodes — same process, takes all outputs from hidden layer 1
- **Output layer**: 1 node — produces the final prediction

The connections between nodes have **weights** — numbers that control how much influence one node has on the next. Every arrow in the diagram is a weight.

The curved arrows on the right labeled "Adjustments of weights through back propagation" show what happens after a prediction is made — the error flows back through the network and each weight gets nudged.

For example, in a sentiment classifier:

- Input: the embedding vectors of each word in a sentence (4 numbers → 4 input nodes)
- Hidden layers: internal representations the model builds from those inputs
- Output: a single number (positive or negative)

At the start, all weights are random. The model knows nothing. It will make bad predictions.

Learning is the process of adjusting those weights so the predictions get better.

---

### What Gradient Descent Is

To improve the weights, the model needs two things:

1. a way to measure how wrong it is
2. a way to adjust the weights in the direction that makes it less wrong

**Measuring how wrong: the loss function**

After making a prediction, the model compares it to the correct answer. The difference is the **loss**. The goal of training is to reduce loss across the entire dataset.

Two loss functions appear constantly in NLP:

**Mean Squared Error (MSE)** — used when the output is a continuous number (like a rating).

```
Actual rating:    5
Predicted rating: 3
Error:            (3 − 5)² = 4

Actual rating:    4
Predicted rating: 4.5
Error:            (4.5 − 4)² = 0.25

MSE = average of all errors = (4 + 0.25) / 2 = 2.125
```

**Cross-Entropy Loss** — used for classification (like sentiment detection). It measures how confident the model was about the correct answer.

```
Correct label: positive

Model predicted 0.85 probability of positive:
  Loss = −log(0.85) = 0.16   ← low loss, confident and correct

Model predicted 0.20 probability of positive:
  Loss = −log(0.20) = 1.61   ← high loss, unconfident and wrong
```

Cross-entropy punishes the model not just for being wrong, but for being wrong confidently. A model that says "80% negative" when the answer is positive gets a much higher penalty than one that says "45% negative."

**Adjusting the weights: gradient descent**

Think of the loss as a landscape. The x-axis is a weight value. The y-axis is the resulting loss. The model is trying to find the lowest point.

```
Loss
  |
  |  .
  | / \
  |/   \       .
  |     \     / \
  |      \   /   \
  |       \_/     \___
  |         ↑
  |       minimum
  |
  +-------------------------> Weight value
```

At any point on that curve, the **gradient** tells you the slope — which direction is uphill. Moving in the opposite direction moves the weight downhill and reduces loss.

The update rule for every weight is:

```
new_weight = old_weight − (learning_rate × gradient)
```

A concrete example with one weight:

```
weight = 0.80
gradient at that point = 0.30  (loss increases as weight increases → move it down)
learning rate = 0.10

step 1:  w = 0.80 − (0.10 × 0.30) = 0.80 − 0.03 = 0.77
step 2:  w = 0.77 − (0.10 × 0.24) = 0.77 − 0.024 = 0.746
step 3:  w = 0.746 − (0.10 × 0.19) = 0.746 − 0.019 = 0.727
...
```

Each step the gradient gets smaller because the weight is getting closer to the minimum. The steps shrink automatically as the model improves.

A real network has thousands to billions of weights. This same calculation runs for every single one of them after every training example.

**Learning rate**

The size of each step is the **learning rate**.

```
Too large:  w = 0.80 − (1.0 × 0.30) = 0.50  → overshoots, bounces past minimum
Too small:  w = 0.80 − (0.001 × 0.30) = 0.7997  → barely moves, training takes forever
Just right: w = 0.80 − (0.10 × 0.30) = 0.77   → makes meaningful progress
```

Choosing the learning rate is one of the most practical challenges in training. Most modern systems use adaptive methods that adjust the learning rate automatically per weight.

---

### What Backpropagation Is

The network has many layers and thousands of weights. After computing the loss, the model needs to figure out: which weights were responsible for the error, and by how much?

This is what **backpropagation** does.

**Why this is not obvious**

Gradient descent needs the gradient for every weight. But a weight in layer 1 does not directly produce the output — it affects a hidden layer, which affects the next layer, which eventually affects the output. You cannot compute its gradient without tracing the full path.

This is where the **chain rule** comes in. If A affects B and B affects C, then the effect of A on C is:

```
∂C/∂A = (∂C/∂B) × (∂B/∂A)
```

Read as: "how does C change with A" = "how does C change with B" × "how does B change with A."

Backpropagation applies this chain rule from the output back through every layer to compute the gradient for every weight.

**A concrete example**

Network: one input → one hidden node → one output. Two weights: w1 and w2.

```
x = 2.0          (input)
w1 = 0.5         (weight: input → hidden)
w2 = 0.3         (weight: hidden → output)
target = 1.0     (correct answer)
learning rate = 0.1
```

**Step 1 — Forward pass:**

```
hidden = x × w1       = 2.0 × 0.5    = 1.0
output = hidden × w2  = 1.0 × 0.3    = 0.3

loss = (output − target)²
     = (0.3 − 1.0)²
     = (−0.7)²
     = 0.49
```

The model predicted 0.3 but the correct answer was 1.0. Loss is 0.49.

**Step 2 — Backward pass (chain rule):**

Start from the loss and work backward.

*Gradient of loss with respect to output:*
```
∂Loss/∂output = 2 × (output − target)
              = 2 × (0.3 − 1.0)
              = −1.4
```
The −1.4 means: increasing the output reduces the loss. The output needs to go up.

*Gradient for w2 (one step from output):*
```
∂output/∂w2 = hidden = 1.0

∂Loss/∂w2 = (∂Loss/∂output) × (∂output/∂w2)
           = −1.4 × 1.0
           = −1.4
```

*Gradient for w1 (two steps from output — chain rule through both layers):*
```
∂output/∂hidden = w2 = 0.3
∂hidden/∂w1     = x  = 2.0

∂Loss/∂w1 = (∂Loss/∂output) × (∂output/∂hidden) × (∂hidden/∂w1)
           = −1.4 × 0.3 × 2.0
           = −0.84
```

**Step 3 — Update both weights:**

```
w2_new = w2 − (lr × ∂Loss/∂w2) = 0.30 − (0.1 × −1.4) = 0.30 + 0.14 = 0.44
w1_new = w1 − (lr × ∂Loss/∂w1) = 0.50 − (0.1 × −0.84) = 0.50 + 0.084 = 0.584
```

Both weights increased because the output needed to go up (gradient was negative → step moved weights upward).

**Verify: did the loss go down?**

```
hidden = 2.0 × 0.584   = 1.168
output = 1.168 × 0.44  = 0.514

loss = (0.514 − 1.0)²  = 0.236
```

Before: loss = 0.49. After one step: loss = 0.236. The model is now closer to the correct answer.

---

![Forward and backward pass through a neural network](assets/backpropagation.png)

- **Forward pass** (blue): input flows through layers → prediction
- **Backward pass** (red): error flows backward → gradient for every weight

Once every weight has its gradient, gradient descent takes one step. Then the cycle repeats.

**What happens in deeper networks**

In a network with 10 layers, the gradient for a weight in layer 1 is computed by multiplying through 10 chain rule steps. Each multiplication can shrink the gradient — weights far from the output receive very small gradients and barely update. This is the **vanishing gradient problem**, which becomes critical in sequence models processing long input.

---

### Why This Matters for Everything That Follows

Every model from here forward — RNNs, LSTMs, transformers, language models — learns using this same loop:

1. make a prediction (forward pass)
2. measure the error (loss)
3. propagate the error backward (backpropagation)
4. adjust weights (gradient descent)
5. repeat

The architectures change. The learning mechanism does not.

When ch04 said "the model adjusts all the numbers slightly based on the error" during Word2Vec training — that was this. Gradient descent and backpropagation, applied to embedding vectors.

When we say an RNN "learns" to connect "it" to "cat" — that is this. The model made wrong predictions, the error propagated back through every step of the sequence, and the weights were updated.

The next chapter covers what happens when you build that loop into a model that processes sequences one word at a time.
