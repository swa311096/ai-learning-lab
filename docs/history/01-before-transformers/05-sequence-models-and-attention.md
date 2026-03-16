# 05: Sequence Models and Attention

Embeddings gave each word a meaningful representation. But they still processed each word in isolation.

What happens when meaning depends on what came earlier?

## Example: Why Sequence Matters

Consider:

"The cat sat on the mat because it was tired."

What does "it" refer to? The cat. Not the mat.

A model reading words in isolation has no way to connect "it" to "cat." It needs to carry information forward across the sentence.

That is what sequence models were built to do.

## What "Recurrent" Means

Sequence models are called **recurrent** because they loop — the output of one step feeds back as input to the next.

At each step, the model takes two things:

1. the current word's embedding
2. the hidden state from the previous step

And produces one thing:

- a new hidden state

```
Step 1:  embed("The")  + h_0 (empty) → h_1
Step 2:  embed("cat")  + h_1         → h_2
Step 3:  embed("sat")  + h_2         → h_3
...
```

The hidden state `h` is a fixed-size vector — a compressed summary of everything processed so far. Each new word updates it.

This is what makes the model sequential: you cannot process step 3 until you have the output of step 2. The model must read words one at a time, in order.

## RNNs and Their Problem

The basic version of this architecture is called an **RNN** (Recurrent Neural Network).

The problem: at each step, the new hidden state is computed from the current word and the previous hidden state — but the previous hidden state itself is already a compressed mixture of everything before it.

As the sequence gets longer, earlier information gets overwritten. By step 50, the hidden state carries very little of what happened at step 1.

There is also a deeper issue: training these models requires computing gradients backward through every step. In a long sequence, those gradients fade as they travel back through time — a problem called **vanishing gradients**. The model struggles to learn that something at step 1 is connected to something at step 40.

## LSTMs: Solving the Forgetting Problem

The **LSTM** (Long Short-Term Memory) was built to address this.

Instead of a single hidden state that gets rewritten at every step, LSTMs maintain two separate pieces:

- a **hidden state** (short-term memory, passed to the next step)
- a **cell state** (long-term memory, carried forward with less modification)

LSTMs also add gates — mechanisms that learn what to keep, what to forget, and what to expose:

- **forget gate**: how much of the existing long-term memory to discard
- **input gate**: how much new information to add to long-term memory
- **output gate**: what to pass to the next step as short-term memory

The result: LSTMs could maintain useful information across longer sequences than vanilla RNNs. They became the dominant architecture for sequence tasks through the 2010s.

**GRUs** (Gated Recurrent Units) were a simpler variant with fewer gates that achieved similar results with less computation.

## What Sequence Models Could Do

Compared to count-based and embedding-only methods, sequence models handled:

- machine translation
- text generation
- speech recognition
- named entity recognition and sequence labeling

## The Encoder-Decoder Architecture

One of the most important applications of sequence models was machine translation.

The standard setup: an **encoder-decoder** architecture.

The **encoder** reads the source sentence word by word and compresses the whole thing into a single vector — the final hidden state.

The **decoder** takes that single vector and generates the target language sentence word by word.

```
Source: "The cat sat on the mat."
         ↓ encoder reads all words
         → one vector [compressed representation]
         ↓ decoder reads that vector
Output: "Le chat était assis sur le tapis."
```

This worked. But it had a hard limit: the entire source sentence had to fit into one fixed-size vector before decoding could begin.

For short sentences, fine. For longer ones, the encoder had to cram too much into a single vector. Information was lost before the decoder ever started.

Consider translating a 40-word sentence. The final encoder hidden state is trying to represent the meaning of all 40 words. Words near the beginning are compressed through 40 steps of updates. By the end, their contribution to the vector is minimal.

## Attention: Letting the Decoder Look Back

Attention was introduced to solve the encoder bottleneck.

Instead of compressing the source sentence into one vector, the model keeps every encoder hidden state — one per source word.

When the decoder generates each output word, it is allowed to look at all the encoder hidden states and decide which source words are most relevant right now.

**How it computes this:**

1. For each encoder hidden state, compute a **score**: how relevant is this source word to the current decoding step?
2. Convert those scores to weights using softmax — they sum to 1.
3. Take a weighted sum of the encoder hidden states. This is the **context vector** for this step.
4. Use that context vector (instead of just the final encoder state) to predict the next output word.

```
Decoding step: generating "chat" (cat in French)

Scores against source words:
  "The"  → 0.02
  "cat"  → 0.91   ← highest weight
  "sat"  → 0.03
  "on"   → 0.02
  "the"  → 0.01
  "mat"  → 0.01

Context vector = weighted sum of encoder states
               = mostly the encoder state for "cat"
```

The model learned to focus on "cat" when generating the French word for cat. Not because anyone told it to — because the weights were learned from training data.

## Example: Attention in Action

In the sentence:

"The trophy did not fit in the suitcase because it was too large."

When the model reaches "it", it needs to decide whether "it" refers to "trophy" or "suitcase."

Without attention, the model relies on its compressed hidden state — which may have lost the distinction between the two nouns across that many steps.

With attention, the model computes a score for every earlier word. "Large" connects more naturally to something that does not fit (the trophy, not the container), so the model learns to weight "trophy" more heavily when resolving "it."

A model relying only on a compressed hidden state is much more likely to get this wrong.

## Why Attention Mattered Beyond Sequence Models

Attention was originally added to patch the encoder-decoder bottleneck. But it revealed something deeper.

The step-by-step processing of RNNs and LSTMs was not just slow — it was architecturally the wrong way to handle long-range relationships. The hidden state was a bottleneck by design.

Attention gave any position in a sequence direct access to any other position. No compression. No sequential dependency.

This pointed toward a new idea: what if you removed the recurrent structure entirely and built an architecture around attention from the start?

That is where the next part begins.
